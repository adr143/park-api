from flask import Flask, request, send_file, jsonify, url_for
import os
import cv2
import json
import numpy as np
import time
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = YOLO("vehicle_model.pt")

ILLEGAL_PARKING_TIME = 20

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        roi_json = request.form.get('coords', None)

        input_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
        output_path = os.path.join(PROCESSED_FOLDER, 'processed.mp4')
        video_file.save(input_path)

        roi_points = None
        if roi_json:
            roi_points = json.loads(roi_json)
            roi_points = np.array([[int(p['x']), int(p['y'])] for p in roi_points], np.int32)

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 1.0 / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        vehicle_entry_time = {}
        illegal_parking_ids = set()
        last_seen = {}

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            if roi_points is not None:
                x, y, w, h = cv2.boundingRect(roi_points)
                cropped_frame = frame[y:y+h, x:x+w]
                results = model.track(cropped_frame, persist=True, verbose=False)

                # Draw ROI polygon
                cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

                if results and results[0].boxes.id is not None:
                    for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                             results[0].boxes.id.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        x1 += x
                        x2 += x
                        y1 += y
                        y2 += y

                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        inside_roi = cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0

                        if inside_roi:
                            if track_id not in vehicle_entry_time:
                                vehicle_entry_time[track_id] = current_time
                            last_seen[track_id] = current_time

                            time_in_roi = current_time - vehicle_entry_time[track_id]

                            if time_in_roi >= ILLEGAL_PARKING_TIME:
                                illegal_parking_ids.add(track_id)
                                color = (0, 0, 255)  # 🔴 Red for illegal
                            else:
                                color = (0, 255, 0)  # 🟢 Green for still waiting
                        else:
                            if track_id in vehicle_entry_time:
                                del vehicle_entry_time[track_id]
                            color = (255, 255, 0)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"ID:{int(track_id)}"
                        if track_id in vehicle_entry_time:
                            elapsed = int(current_time - vehicle_entry_time[track_id])
                            label += f" ({elapsed}s)"
                        cv2.putText(frame, label, (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            else:
                results = model.track(frame, persist=True, verbose=False)
                if results and results[0].boxes.id is not None:
                    for track_id in results[0].boxes.id.cpu().numpy():
                        vehicle_entry_time.setdefault(track_id, current_time)
                frame = results[0].plot()

            out.write(frame)

        cap.release()
        out.release()

        total_illegal = len(illegal_parking_ids)
        video_url = url_for('get_processed_video', _external=True)

        return jsonify({
            "total_illegal_parking": total_illegal,
            "illegal_ids": list(illegal_parking_ids),
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
    app.run(host='0.0.0.0', port=5000)
