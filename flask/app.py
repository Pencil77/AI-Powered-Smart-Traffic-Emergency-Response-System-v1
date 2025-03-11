from flask import Flask, render_template, request, jsonify
import os
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi'}

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, DETECTION_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Initialize YOLO model
try:
    model = YOLO('yolov9c.pt')
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

# Define vehicle classes
VEHICLE_CLASSES = {
    "car": 2,      # Class index for car
    "truck": 7,    # Class index for truck
    "bus": 5,      # Class index for bus
    "motorbike": 3,# Class index for motorbike
    "bicycle": 1   # Class index for bicycle
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add cleanup function
def cleanup_old_files():
    """Clean up files older than 1 hour"""
    current_time = datetime.now()
    for folder in [UPLOAD_FOLDER, DETECTION_FOLDER]:
        for filename in os.listdir(folder):
            if filename == '.gitkeep':
                continue
            filepath = os.path.join(folder, filename)
            try:
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                if (current_time - file_time).total_seconds() > 3600:  # 1 hour
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {e}")

def count_vehicles_by_type(results):
    """Count vehicles by type from detection results"""
    try:
        vehicle_counts = {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES.keys()}

        for box in results[0].boxes.data:
            class_id = int(box[5])
            for vehicle_type, idx in VEHICLE_CLASSES.items():
                if class_id == idx:
                    vehicle_counts[vehicle_type] += 1
                    break

        return vehicle_counts
    except Exception as e:
        logger.error(f"Error counting vehicles: {e}")
        return {vehicle_type: 0 for vehicle_type in VEHICLE_CLASSES.keys()}

def process_image(image_path):
    """Process a single image and return detection results"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")

        results = model(img)
        vehicle_counts = count_vehicles_by_type(results)

        # Draw detections
        img_with_vehicles = img.copy()
        for box in results[0].boxes.data:
            class_id = int(box[5])
            if class_id in VEHICLE_CLASSES.values():
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = results[0].names[class_id]
                cv2.rectangle(img_with_vehicles, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_vehicles, class_name, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img_with_vehicles, vehicle_counts, sum(vehicle_counts.values())
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None, None, 0

def process_video(video_path):
    """Process a video and return detection results"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = video_path.replace(UPLOAD_FOLDER, DETECTION_FOLDER).replace('.mp4', '_output.mp4')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

        return output_path, average_vehicle_counts, total_average_vehicles
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")
        return None, None, 0

# Add health check endpoint
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route handler"""
    cleanup_old_files()  # Add cleanup at start of route
    if request.method == "POST":
        try:
            # Get form data
            lane_count = request.form.get('lane_count', type=int)
            cycle_time = request.form.get('cycle_time', type=int, default=60)

            # Validate inputs
            if not lane_count:
                return render_template('index.html', error="Please select the number of lanes.")
            if not 30 <= cycle_time <= 180:
                return render_template('index.html', error="Cycle time must be between 30 and 180 seconds.")

            # Initialize arrays
            lane_vehicle_counts = [0] * lane_count
            lane_vehicle_types = [{}] * lane_count
            detection_outputs = []

            # Process files
            files = request.files.getlist('files')
            if len(files) != lane_count:
                return render_template('index.html',
                                     error=f"Please upload exactly {lane_count} files (one per lane).")

            for i, file in enumerate(files):
                if not file or not file.filename:
                    return render_template('index.html',
                                         error=f"Missing file for lane {i+1}")

                if not allowed_file(file.filename):
                    return render_template('index.html',
                                         error=f"Invalid file format for lane {i+1}. Use jpg, png, or mp4 only.")

                # Save file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lane_{i+1}_{timestamp}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Process file based on type
                file_extension = file.filename.rsplit('.', 1)[1].lower()

                if file_extension in ['jpg', 'jpeg', 'png']:
                    # Process image
                    img_with_vehicles, vehicle_counts, total_vehicles = process_image(filepath)
                    if img_with_vehicles is None:
                        return render_template('index.html',
                                             error=f"Error processing image for lane {i+1}")

                    detection_path = os.path.join(DETECTION_FOLDER, filename)
                    cv2.imwrite(detection_path, img_with_vehicles)

                    lane_vehicle_counts[i] = total_vehicles
                    lane_vehicle_types[i] = vehicle_counts

                    detection_outputs.append({
                        'lane': i+1,
                        'file_type': 'image',
                        'original_file': filename,
                        'detection_file': filename,
                        'vehicle_count': total_vehicles,
                        'vehicle_types': vehicle_counts
                    })

                else:  # video file
                    # Process video
                    output_path, vehicle_counts, total_vehicles = process_video(filepath)
                    if output_path is None:
                        return render_template('index.html',
                                             error=f"Error processing video for lane {i+1}")

                    lane_vehicle_counts[i] = total_vehicles
                    lane_vehicle_types[i] = vehicle_counts

                    detection_outputs.append({
                        'lane': i+1,
                        'file_type': 'video',
                        'original_file': filename,
                        'detection_file': os.path.basename(output_path),
                        'vehicle_count': total_vehicles,
                        'vehicle_types': vehicle_counts
                    })

            # Calculate total vehicles and green times
            total_vehicles = sum(lane_vehicle_counts)
            if total_vehicles == 0:
                green_times = [cycle_time / lane_count] * lane_count
            else:
                green_times = [round((count / total_vehicles) * cycle_time, 2)
                             for count in lane_vehicle_counts]

            # Prepare lane info
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
                                 cycle_time=cycle_time,
                                 show_results=True)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return render_template('index.html',
                                 error="An error occurred while processing your request. Please try again.")

    return render_template('index.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
