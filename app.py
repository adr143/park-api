from flask import Flask, request, send_file, jsonify, url_for
import os
import cv2
import time
import json
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("vehicle_model.pt")

# ---- Time threshold in seconds ----
STAY_THRESHOLD = 30


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']

        input_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
        output_path = os.path.join(PROCESSED_FOLDER, 'processed.mp4')

        video_file.save(input_path)

        # ---- Initialize Video I/O ----
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ---- Tracking time dictionary ----
        # Dictionary: { track_id : { "first_seen": timestamp, "last_seen": timestamp } }
        object_times = {}

        # ---- Final registered objects ----
        registered_objects = set()

        start_processing_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Tracking
            results = model.track(frame, persist=True, verbose=False)

            if results and results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                current_timestamp = time.time()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)

                    # Initialize tracking entry if new ID
                    if track_id not in object_times:
                        object_times[track_id] = {
                            "first_seen": current_timestamp,
                            "last_seen": current_timestamp
                        }
                    else:
                        object_times[track_id]["last_seen"] = current_timestamp

                    # Calculate duration
                    duration = object_times[track_id]["last_seen"] - object_times[track_id]["first_seen"]

                    # If duration above threshold → red box + register object
                    if duration >= STAY_THRESHOLD:
                        color = (0, 0, 255)  # RED
                        registered_objects.add(track_id)
                    else:
                        color = (0, 255, 0)  # GREEN

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw text
                    cv2.putText(
                        frame,
                        f"ID {track_id} | {duration:.1f}s",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )

            out.write(frame)

        cap.release()
        out.release()

        # Final count = number of objects exceeding threshold
        tracked_count = len(registered_objects)

        video_url = url_for('get_processed_video', _external=True)

        return jsonify({
            "tracked_objects": tracked_count,
            "video_url": video_url
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/processed_video')
def get_processed_video():
    output_path = os.path.join(PROCESSED_FOLDER, 'processed.mp4')
    if not os.path.exists(output_path):
        return jsonify({"error": "Processed video not found"}), 404

    return send_file(output_path, mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
