from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from playsound import playsound
import os
import logging
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
beep_thread = None
stop_beep = threading.Event()

def load_model():
    global model
    try:
        if not os.path.exists("best.pt"):
            logger.error("Model file 'best.pt' not found!")
            return False
        model = YOLO("best.pt")
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def continuous_beep():
    while not stop_beep.is_set():
        try:
            if os.path.exists('beep.mp3'):
                playsound('beep.mp3')
                time.sleep(1)
            else:
                logger.error("beep.mp3 not found in static directory")
                break
        except Exception as e:
            logger.error(f"Error playing sound: {e}")
            break

@app.route('/')
def index():
    return send_from_directory('.', 'dummy.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/detect', methods=['POST'])
def detect_drowsiness():
    global model, beep_thread
    
    if model is None:
        if not load_model():
            return jsonify({'error': 'Model not initialized'}), 500
    
    if 'frame' not in request.files:
        logger.error("No frame received in request")
        return jsonify({'error': 'No frame part'}), 400
    
    try:
        file = request.files['frame']
        file_bytes = file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Run detection
        results = model(frame, verbose=False)
        drowsy_detected = False
        detections = []
        
        # Process detection results
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                    
                    # Draw bounding box and label on frame
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{model.names[class_id]}: {confidence:.2f}"
                    
                    # Store detection info
                    detections.append({
                        'class': model.names[class_id],
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    if model.names[class_id] == "Drowsy":
                        drowsy_detected = True
                        # Draw red rectangle for drowsy
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # Draw green rectangle for alert
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert processed frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        # Handle beeping
        if drowsy_detected and (beep_thread is None or not beep_thread.is_alive()):
            stop_beep.clear()
            beep_thread = threading.Thread(target=continuous_beep, daemon=True)
            beep_thread.start()
        elif not drowsy_detected:
            stop_beep.set()
        
        return jsonify({
            'drowsy': drowsy_detected,
            'detections': detections,
            'processed_image': processed_image,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in detection: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/stop_beep', methods=['POST'])
def stop_beep_handler():
    global beep_thread
    stop_beep.set()
    if beep_thread and beep_thread.is_alive():
        beep_thread.join(timeout=1)
    beep_thread = None
    return jsonify({'message': 'Beep stopped'})

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    if not load_model():
        logger.error("Failed to load model. Please ensure 'best.pt' exists in the correct location.")
    app.run(debug=True, port=5500)