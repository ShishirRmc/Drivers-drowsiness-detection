# Drowsiness Detection System 🚗💤

This is a real-time **Drowsiness Detection System** using **YOLO**, **OpenCV**, and **Streamlit**. The application monitors a video stream to detect drowsy drivers and alerts them using a beep sound.

## Features
- 🔍 **Real-time drowsiness detection** using YOLO.
- 🔊 **Beep sound alert** when drowsiness is detected.
- 🔄 **Fullscreen Mode**: Automatically starts in fullscreen and can be toggled with the `F` key.


---

## Installation 🛠️

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- OpenCV
- Streamlit
- Pygame
- Pynput
- Ultralytics YOLO
- Screeninfo

### Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/ShishirRmc/Drivers-drowsiness-detection.git
   cd drowsiness-detection
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```sh
   streamlit run web_app.py
   ```

---

## Usage 🚀
- Click **Start Detection** to begin monitoring.
- The app will switch to **fullscreen mode**.
- If drowsiness is detected, a **red label** and **beep alert** will be triggered.
- Press **F** to toggle fullscreen.
- Click **Stop Detection** to exit monitoring.

---

## Troubleshooting 🔧
### Beep Sound Not Working
- Ensure **beep.mp3** is available in the project folder.
- Verify `pygame.mixer.init()` is successfully executed.


### Keyboard Error: `keyboard.Key.f` Not Found
- Replace `keyboard.Key.f` with `keyboard.KeyCode.from_char('f')`.

---

## License 📜
This project is licensed under the MIT License.

---
