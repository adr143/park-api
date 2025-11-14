from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO
import cv2
import tempfile
import shutil
import time

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    # Simple HTML player to test stream in browser
    return HTMLResponse("""
    <html>
    <body style="text-align:center;">
      <h2>YOLO Live Stream</h2>
      <img src="/stream" style="width:80%; border:2px solid #ccc;">
    </body>
    </html>
    """)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video and begin live YOLO stream."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        input_path = tmp.name
    return StreamingResponse(process_video(input_path),
                             media_type="multipart/x-mixed-replace; boundary=frame")

def process_video(path: str):
    """Stream YOLO processed frames in MJPEG format."""
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated = results[0].plot()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)  # ~30fps
    cap.release()
