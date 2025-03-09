from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Initialize YOLO model
model = YOLO('yolov9c.pt')

# Define vehicle classes
VEHICLE_CLASSES = {
    "car": 2,      # Class index for car
    "truck": 7,    # Class index for truck
    "bus": 5,      # Class index for bus
    "motorbike": 3,# Class index for motorbike
    "bicycle": 1   # Class index for bicycle
}

def count_vehicles_by_type(results):
    vehicle_counts = {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES.keys()}

    for box in results[0].boxes.data:
        class_id = int(box[5])
        for vehicle_type, idx in VEHICLE_CLASSES.items():
            if class_id == idx:
                vehicle_counts[vehicle_type] += 1
                break

    return vehicle_counts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        lane_count = request.form.get('lane_count', type=int)
        if not lane_count:
            return render_template('index.html', error="Please enter the number of lanes.")

        lane_vehicle_counts = [0] * lane_count
        lane_vehicle_types = [{}] * lane_count

        if 'files' not in request.files:
            return render_template('index.html', error="No files were selected for upload.")

        files = request.files.getlist('files')
        if len(files) != lane_count:
            return render_template('index.html', error=f"Please upload exactly {lane_count} files (one per lane).")

        detection_outputs = []

        for i, file in enumerate(files):
            if file and file.filename != '':
                filepath = os.path.join(UPLOAD_FOLDER, f"lane_{i+1}_{file.filename}")
                file.save(filepath)

                file_extension = file.filename.rsplit('.', 1)[1].lower()

                if file_extension in ['jpg', 'jpeg', 'png']:
                    img = cv2.imread(filepath)
                    results = model(img)

                    # Count vehicles by type
                    vehicle_counts = count_vehicles_by_type(results)
                    lane_vehicle_types[i] = vehicle_counts
                    total_vehicles = sum(vehicle_counts.values())

                    # Filter and draw detections
                    img_with_vehicles = img.copy()
                    for box in results[0].boxes.data:
                        class_id = int(box[5])
                        if class_id in VEHICLE_CLASSES.values():
                            x1, y1, x2, y2 = map(int, box[:4])
                            class_name = results[0].names[class_id]
                            cv2.rectangle(img_with_vehicles, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_with_vehicles, class_name, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    detection_path = os.path.join(DETECTION_FOLDER, f"lane_{i+1}_{file.filename}")
                    cv2.imwrite(detection_path, img_with_vehicles)

                    lane_vehicle_counts[i] = total_vehicles

                    detection_outputs.append({
                        'lane': i+1,
                        'file_type': 'image',
                        'original_file': f"lane_{i+1}_{file.filename}",
                        'detection_file': f"lane_{i+1}_{file.filename}",
                        'vehicle_count': total_vehicles,
                        'vehicle_types': vehicle_counts
                    })

                elif file_extension in ['mp4', 'avi']:
                    video_path = filepath
                    output_path = os.path.join(DETECTION_FOLDER, f"lane_{i+1}_output.mp4")

                    cap = cv2.VideoCapture(video_path)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

                    frame_count = 0
                    total_vehicle_counts = {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES.keys()}

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame)
                        frame_vehicle_counts = count_vehicles_by_type(results)

                        for vehicle_type, count in frame_vehicle_counts.items():
                            total_vehicle_counts[vehicle_type] += count

                        # Draw detections
                        frame_with_vehicles = frame.copy()
                        for box in results[0].boxes.data:
                            class_id = int(box[5])
                            if class_id in VEHICLE_CLASSES.values():
                                x1, y1, x2, y2 = map(int, box[:4])
                                class_name = results[0].names[class_id]
                                cv2.rectangle(frame_with_vehicles, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame_with_vehicles, class_name, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        out.write(frame_with_vehicles)
                        frame_count += 1

                    cap.release()
                    out.release()

                    # Calculate averages
                    average_vehicle_counts = {
                        vehicle_type: count // frame_count if frame_count > 0 else 0
                        for vehicle_type, count in total_vehicle_counts.items()
                    }
                    total_average_vehicles = sum(average_vehicle_counts.values())

                    lane_vehicle_counts[i] = total_average_vehicles
                    lane_vehicle_types[i] = average_vehicle_counts

                    detection_outputs.append({
                        'lane': i+1,
                        'file_type': 'video',
                        'original_file': f"lane_{i+1}_{file.filename}",
                        'detection_file': f"lane_{i+1}_output.mp4",
                        'vehicle_count': total_average_vehicles,
                        'vehicle_types': average_vehicle_counts
                    })

                else:
                    return render_template('index.html', error="Unsupported file format. Use .jpg, .png, or .mp4 only.")
            else:
                return render_template('index.html', error="One of the files has no filename.")

        total_vehicles = sum(lane_vehicle_counts)
        green_times = [round((count / total_vehicles) * 60.0, 2) if total_vehicles else round(60.0 / lane_count, 2) for count in lane_vehicle_counts]

        lane_info = [{
            'lane_number': i+1,
            'vehicle_count': lane_vehicle_counts[i],
            'green_time': green_times[i],
            'vehicle_types': lane_vehicle_types[i]
        } for i in range(lane_count)]

        return render_template('index.html',
                             detection_outputs=detection_outputs,
                             lane_info=lane_info,
                             total_vehicles=total_vehicles,
                             show_results=True)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
