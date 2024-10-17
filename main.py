import cv2
import torch
from flask import Flask, request, jsonify

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Define the Flask app
app = Flask(__name__)


@app.route("/detect_motion", methods=["POST"])
def detect_motion():
    data = request.json
    video_path = data["video_path"]
    roi_coordinates = data["roi_coordinates"]  # [x_min, y_min, x_max, y_max]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 400

    x_min, y_min, x_max, y_max = roi_coordinates

    detected_objects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection with YOLOV5
        results = model(frame)
        # Convert results to DataFra,e amd get objects
        for obj in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = obj[:6]
            if conf > 0.5:  # Confidence threshold
                # Check if the detected object falls within the region of interest(ROI)
                if x_min < x1 < x_max and y_min < y1 < y_max:
                    detected_objects.append(
                        {
                            "object": model.names[int(cls)],
                            "coordinates": [int(x1), int(y1), int(x2), int(y2)],
                            "Confidence": float(conf),
                        }
                    )

    cap.release()

    if not detected_objects:
        return jsonify({"message": "No motion detected in the specified area"})

    return jsonify({"detected_objects": detected_objects})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
