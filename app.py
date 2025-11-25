from flask import Flask, request, send_file, jsonify, url_for
import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO
from supabase import create_client, Client

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
VIDEO_NAME = "processed.mp4"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

SUPABASE_URL = "https://tyiiawylacwgxproemzc.supabase.co"
SUPABASE_KEY = "sb_secret_FojsYWzigjRrieh0k6VAEw_LgjFTw-6"
SUPABASE_BUCKET = "video_violation"

NGROK_URL = "https://redfish-fancy-solely.ngrok-free.app"

# Load YOLOv8 model
model = YOLO("vehicle_model.pt")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Time threshold in seconds ----
STAY_THRESHOLD = 20


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']

        input_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
        output_path = os.path.join(PROCESSED_FOLDER, VIDEO_NAME)

        video_file.save(input_path)

        # ---- Initialize Video I/O ----
        cap = cv2.VideoCapture(input_path)
        current_frame = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        object_times = {}
        registered_objects = set()

        while True:
            ret, frame = cap.read()
            current_frame += 1
            progress = (current_frame / total_frames) * 100.0
            print(f"Processing progress: {progress:.2f}%")
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False)

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                current_timestamp = time.time()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)

                    if track_id not in object_times:
                        object_times[track_id] = {
                            "first_seen": current_timestamp,
                            "last_seen": current_timestamp
                        }
                    else:
                        object_times[track_id]["last_seen"] = current_timestamp

                    duration = object_times[track_id]["last_seen"] - object_times[track_id]["first_seen"]

                    if duration >= STAY_THRESHOLD:
                        color = (0, 0, 255)
                        registered_objects.add(track_id)
                    else:
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {track_id} | {duration:.1f}s", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out.write(frame)

        cap.release()
        out.release()

        # --- Upload to Supabase ---
        print("Uploading processed video to Supabase...")
        # Delete existing
        existing_files = supabase.storage.from_(SUPABASE_BUCKET).list()
        for file in existing_files:
            if file['name'] == VIDEO_NAME:
                supabase.storage.from_(SUPABASE_BUCKET).remove([VIDEO_NAME])
                print("Deleted previous video from Supabase.")
                break

        with open(output_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(VIDEO_NAME, f.read())
        print("Upload complete!")

        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(VIDEO_NAME)

        return jsonify({
            "tracked_objects": len(registered_objects),
            "video_url": public_url
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/processed_video')
def get_processed_video():
    # Just return the public URL of the latest uploaded video
    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(VIDEO_NAME)
    return jsonify({"video_url": public_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
