import cv2
import requests
import numpy as np

VIDEO_PATH = "input.mp4"  # change to your local video file
API_URL = "http://127.0.0.1:5000/upload"

def stream_yolo_tracking():
    with open(VIDEO_PATH, "rb") as f:
        files = {"file": ("input.mp4", f, "video/mp4")}
        response = requests.post(API_URL, files=files, stream=True)

        if response.status_code != 200:
            print("Error:", response.status_code)
            print(response.text)
            return

        bytes_buffer = b""
        for chunk in response.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b"\xff\xd8")  # JPEG start
            b = bytes_buffer.find(b"\xff\xd9")  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imshow("YOLO Tracking Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_yolo_tracking()
