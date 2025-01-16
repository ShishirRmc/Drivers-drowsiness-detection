const videoElement = document.getElementById('videoElement');
const startButton = document.getElementById('startDetection');
const stopButton = document.getElementById('stopDetection');
const stopBeepButton = document.getElementById('stopBeep');
const alertMessage = document.getElementById('alertMessage');

let stream;
let intervalId;
let isDetectionRunning = false;
let detectionCanvas;

function setupDetectionCanvas() {
    // Create canvas overlay for drawing detection results
    detectionCanvas = document.createElement('canvas');
    detectionCanvas.style.position = 'absolute';
    detectionCanvas.style.left = '0';
    detectionCanvas.style.top = '0';
    detectionCanvas.style.width = '100%';
    detectionCanvas.style.height = '100%';

    // Add canvas to video container
    const videoContainer = document.querySelector('.video-container');
    videoContainer.style.position = 'relative';
    videoContainer.appendChild(detectionCanvas);
}

async function startVideo() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });
        videoElement.srcObject = stream;

        await new Promise((resolve) => {
            videoElement.onloadedmetadata = resolve;
        });

        videoElement.play();

        // Set up canvas after video dimensions are known
        detectionCanvas.width = videoElement.videoWidth;
        detectionCanvas.height = videoElement.videoHeight;

        return true;
    } catch (err) {
        console.error("Error accessing the camera:", err);
        alert("Failed to access camera. Please ensure camera permissions are granted.");
        return false;
    }
}

function stopVideo() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    videoElement.srcObject = null;
    hideAlert();

    // Clear detection canvas
    if (detectionCanvas) {
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
    }
}

async function startDetection() {
    if (isDetectionRunning) return;

    setupDetectionCanvas();
    const success = await startVideo();
    if (!success) return;

    isDetectionRunning = true;
    startButton.disabled = true;
    stopButton.disabled = false;

    await new Promise(resolve => setTimeout(resolve, 1000));

    intervalId = setInterval(sendFrame, 1000);
}

function stopDetection() {
    isDetectionRunning = false;
    clearInterval(intervalId);
    stopVideo();
    hideAlert();
    startButton.disabled = false;
    stopButton.disabled = true;
    stopBeep();
}

function updateDetectionDisplay(processedImageBase64) {
    const img = new Image();
    img.onload = () => {
        const ctx = detectionCanvas.getContext('2d');
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        ctx.drawImage(img, 0, 0, detectionCanvas.width, detectionCanvas.height);
    };
    img.src = 'data:image/jpeg;base64,' + processedImageBase64;
}

function sendFrame() {
    if (!videoElement.videoWidth) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);

    canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const data = await response.json();

            if (data.processed_image) {
                updateDetectionDisplay(data.processed_image);
            }

            if (data.drowsy) {
                showAlert();
            } else {
                hideAlert();
            }
        } catch (error) {
            console.error('Error in detection:', error);
            if (isDetectionRunning) {
                stopDetection();
                alert('Detection error occurred. Please try again.');
            }
        }
    }, 'image/jpeg', 0.8);
}

function showAlert() {
    alertMessage.classList.remove('hidden');
    document.body.style.backgroundColor = '#ff0000';
}

function hideAlert() {
    alertMessage.classList.add('hidden');
    document.body.style.backgroundColor = '';
}

async function stopBeep() {
    try {
        const response = await fetch('/stop_beep', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) throw new Error('Failed to stop beep');

    } catch (error) {
        console.error('Error stopping beep:', error);
    }
}

// Event Listeners
startButton.addEventListener('click', startDetection);
stopButton.addEventListener('click', stopDetection);
stopBeepButton.addEventListener('click', stopBeep);

// Initialize button states
stopButton.disabled = true;

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (isDetectionRunning) {
        stopDetection();
    }
});