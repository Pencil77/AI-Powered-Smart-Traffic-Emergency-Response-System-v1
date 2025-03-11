import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import logging
import tempfile
import base64
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define vehicle classes
VEHICLE_CLASSES = {
    "car": 2,      # Class index for car
    "truck": 7,    # Class index for truck
    "bus": 5,      # Class index for bus
    "motorbike": 3,# Class index for motorbike
    "bicycle": 1   # Class index for bicycle
}

# Function to load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov9c.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

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

def process_image(image_path, model):
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

        # Convert BGR to RGB for displaying with Streamlit
        rgb_img = cv2.cvtColor(img_with_vehicles, cv2.COLOR_BGR2RGB)
        return rgb_img, vehicle_counts, sum(vehicle_counts.values())
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None, None, 0

def create_traffic_light_css():
    """Create CSS for traffic light visualization"""
    css = """
    <style>
        .traffic-light-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 20px;
        }
        .traffic-light {
            width: 80px;
            height: 200px;
            background: #333;
            border-radius: 10px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            margin: 0 10px;
        }
        .lane-label {
            text-align: center;
            font-weight: bold;
            margin-bottom: 5px;
            color: white;
        }
        .light {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            margin: 5px 0;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        .red { background: #ff0000; opacity: 0.3; }
        .yellow { background: #ffff00; opacity: 0.3; }
        .green { background: #00ff00; opacity: 0.3; }
        .active {
            opacity: 1;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        }
        .timer-display {
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
            font-size: 16px;
            font-family: monospace;
            color: white;
        }
        .timer-text {
            position: absolute;
            color: white;
            font-weight: bold;
            font-size: 14px;
            font-family: monospace;
        }
    </style>
    """
    return css

def create_traffic_light_html(lane_count):
    """Create HTML for traffic light visualization"""
    traffic_lights_html = '<div class="traffic-light-container">'

    for i in range(lane_count):
        traffic_lights_html += f"""
        <div>
            <div class="lane-label">Lane {i+1}</div>
            <div class="traffic-light" id="traffic-light-{i+1}">
                <div class="light red" id="red-{i+1}">
                    <span class="timer-text" id="red-timer-{i+1}"></span>
                </div>
                <div class="light yellow" id="yellow-{i+1}"></div>
                <div class="light green" id="green-{i+1}">
                    <span class="timer-text" id="green-timer-{i+1}"></span>
                </div>
            </div>
            <div class="timer-display" id="timer-{i+1}">00:00</div>
        </div>
        """

    traffic_lights_html += '</div>'
    return traffic_lights_html

def create_traffic_light_js(lane_info, cycle_time):
    """Create JavaScript for traffic light simulation"""
    # Convert lane_info to JSON string for JavaScript
    import json
    lane_info_json = json.dumps(lane_info)

    js = f"""
    <script>
        // Traffic Light Simulation
        class TrafficLightSimulation {{
            constructor(laneInfo, cycleTime) {{
                this.laneInfo = laneInfo;
                this.cycleTime = cycleTime;
                this.currentLane = 0;
                this.totalLanes = laneInfo.length;
                this.isSimulationRunning = true;
                this.timerId = null;
                this.allLaneTimers = [];
            }}

            calculateRemainingTime(currentLaneIndex) {{
                // Calculate remaining time until this lane gets green light
                let timeToWait = 0;
                for (let i = 0; i < this.totalLanes; i++) {{
                    const checkLane = (this.currentLane + i) % this.totalLanes;
                    if (checkLane === currentLaneIndex) break;
                    timeToWait += this.laneInfo[checkLane].green_time + 2; // Add 2 seconds for yellow light
                }}
                return timeToWait;
            }}

            updateTrafficLights() {{
                if (!this.isSimulationRunning) return;

                // Clear any existing timers
                for (let timerId of this.allLaneTimers) {{
                    clearInterval(timerId);
                }}
                this.allLaneTimers = [];

                // Reset all lights
                for(let i = 0; i < this.totalLanes; i++) {{
                    const redLight = document.getElementById(`red-${{i+1}}`);
                    const yellowLight = document.getElementById(`yellow-${{i+1}}`);
                    const greenLight = document.getElementById(`green-${{i+1}}`);
                    const greenTimer = document.getElementById(`green-timer-${{i+1}}`);
                    const redTimer = document.getElementById(`red-timer-${{i+1}}`);
                    const timer = document.getElementById(`timer-${{i+1}}`);

                    if (redLight && yellowLight && greenLight && timer) {{
                        redLight.classList.remove('active');
                        yellowLight.classList.remove('active');
                        greenLight.classList.remove('active');
                        timer.textContent = "00:00";
                        if (greenTimer) greenTimer.textContent = "";
                        if (redTimer) redTimer.textContent = "";

                        if(i !== this.currentLane) {{
                            redLight.classList.add('active');

                            // Calculate and display waiting time for red lights
                            if (i !== this.currentLane) {{
                                let timeToWait = this.calculateRemainingTime(i);

                                // Create a timer for this red light
                                const redTimerId = setInterval(() => {{
                                    if (!this.isSimulationRunning) {{
                                        clearInterval(redTimerId);
                                        return;
                                    }}

                                    if (timeToWait > 0) {{
                                        if (redTimer) {{
                                            redTimer.textContent = Math.ceil(timeToWait);
                                        }}

                                        const minutes = Math.floor(timeToWait/60);
                                        const seconds = Math.floor(timeToWait % 60);
                                        const decimal = Math.floor((timeToWait % 1) * 10);

                                        const timerElement = document.getElementById(`timer-${{i+1}}`);
                                        if (timerElement) {{
                                            timerElement.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}.${{decimal}}`;
                                        }}

                                        timeToWait -= 0.1;
                                    }}
                                }}, 100);

                                this.allLaneTimers.push(redTimerId);
                            }}
                        }}
                    }}
                }}

                // Set green for current lane
                const currentGreenLight = document.getElementById(`green-${{this.currentLane+1}}`);
                const currentGreenTimer = document.getElementById(`green-timer-${{this.currentLane+1}}`);

                if (currentGreenLight) {{
                    currentGreenLight.classList.add('active');
                }}

                // Start timer for current lane
                let timeLeft = this.laneInfo[this.currentLane].green_time;
                const currentTimerId = setInterval(() => {{
                    if (!this.isSimulationRunning) {{
                        clearInterval(currentTimerId);
                        return;
                    }}

                    if(timeLeft >= 0) {{
                        const minutes = Math.floor(timeLeft/60);
                        const seconds = Math.floor(timeLeft % 60);
                        const decimal = Math.floor((timeLeft % 1) * 10);

                        const timerElement = document.getElementById(`timer-${{this.currentLane+1}}`);
                        if (timerElement) {{
                            timerElement.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}.${{decimal}}`;
                        }}

                        // Update in-light timer
                        if (currentGreenTimer) {{
                            currentGreenTimer.textContent = Math.ceil(timeLeft);
                        }}
                    }}
                    timeLeft -= 0.1; // Update every 100ms for smoother countdown
                }}, 100);

                this.allLaneTimers.push(currentTimerId);

                // Schedule next lane
                setTimeout(() => {{
                    if (!this.isSimulationRunning) {{
                        clearInterval(currentTimerId);
                        return;
                    }}

                    // Yellow light transition
                    const currentGreenLight = document.getElementById(`green-${{this.currentLane+1}}`);
                    const currentYellowLight = document.getElementById(`yellow-${{this.currentLane+1}}`);

                    if (currentGreenLight && currentYellowLight) {{
                        currentGreenLight.classList.remove('active');
                        currentYellowLight.classList.add('active');
                    }}

                    setTimeout(() => {{
                        if (!this.isSimulationRunning) {{
                            for (let timerId of this.allLaneTimers) {{
                                clearInterval(timerId);
                            }}
                            return;
                        }}

                        this.currentLane = (this.currentLane + 1) % this.totalLanes;
                        this.updateTrafficLights();
                    }}, 2000); // Yellow light duration

                }}, this.laneInfo[this.currentLane].green_time * 1000);
            }}

            start() {{
                this.updateTrafficLights();
            }}
        }}

        // Initialize simulation
        // Use DOMContentLoaded for the initial load
        document.addEventListener('DOMContentLoaded', function() {{
            initializeSimulation();
        }});

        // This will be executed immediately as well (for when Streamlit reruns the script)
        function initializeSimulation() {{
            const laneInfo = {lane_info_json};
            const cycleTime = {cycle_time};
            const simulation = new TrafficLightSimulation(laneInfo, cycleTime);
            simulation.start();
        }}

        // Execute immediately for Streamlit's dynamic content loading
        initializeSimulation();
    </script>
    """
    return js

def main():
    # App title and description
    st.set_page_config(page_title="Smart Traffic Signal Management", layout="wide")

    st.title("Smart Traffic Signal Management")
    st.markdown("Upload traffic images to analyze vehicle flow and optimize signal timing.")

    # Load YOLO model
    model = load_model()
    if model is None:
        st.error("Failed to load YOLO model. Please check if the model file exists.")
        return

    # Sample images button
    if st.button("Download Sample Traffic Lane Images"):
        st.markdown("[Click here to download sample traffic lane images](https://github.com/suma-vatturi/ai-powered-smart-traffic-management-system/tree/main/test_images)")

    # Configuration section
    st.header("Traffic Analysis Configuration")

    col1, col2 = st.columns(2)

    with col1:
        lane_count = st.selectbox("Number of Traffic Lanes",
                                 options=[1, 2, 3, 4],
                                 format_func=lambda x: f"{x} Lane{'s' if x > 1 else ''}")

    with col2:
        cycle_time = st.slider("Cycle Time (seconds)",
                              min_value=30,
                              max_value=180,
                              value=60,
                              help="Enter cycle time between 30-180 seconds")

    # File upload section
    st.header("Upload Traffic Media")
    st.markdown("Upload one image per lane")

    # Create file uploaders based on lane count
    uploaded_files = []
    for i in range(lane_count):
        uploaded_file = st.file_uploader(f"Lane {i+1}",
                                        type=["jpg", "jpeg", "png"],
                                        key=f"lane_{i+1}")
        uploaded_files.append(uploaded_file)

    # Process button
    if st.button("Analyze Traffic", type="primary"):
        # Validate if all files are uploaded
        if None in uploaded_files:
            st.error("Please upload files for all lanes")
            return

        # Process files
        lane_vehicle_counts = [0] * lane_count
        lane_vehicle_types = [{} for _ in range(lane_count)]
        detection_outputs = []

        with st.spinner("Processing traffic data..."):
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file is None:
                    continue

                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
                temp_file.close()

                # Process image
                img_with_vehicles, vehicle_counts, total_vehicles = process_image(temp_path, model)

                if img_with_vehicles is None:
                    st.error(f"Error processing image for lane {i+1}")
                    os.unlink(temp_path)
                    continue

                lane_vehicle_counts[i] = total_vehicles
                lane_vehicle_types[i] = vehicle_counts

                detection_outputs.append({
                    'lane': i+1,
                    'file_type': 'image',
                    'detection_image': img_with_vehicles,
                    'vehicle_count': total_vehicles,
                    'vehicle_types': vehicle_counts
                })

                # Remove temporary file
                os.unlink(temp_path)

        # Calculate total vehicles and green times
        total_vehicles = sum(lane_vehicle_counts)
        if total_vehicles == 0:
            green_times = [cycle_time / lane_count] * lane_count
        else:
            green_times = [round((count / total_vehicles) * cycle_time, 2)
                         for count in lane_vehicle_counts]

        # Prepare lane info for visualization
        lane_info = [{
            'lane_number': i+1,
            'vehicle_count': lane_vehicle_counts[i],
            'green_time': green_times[i],
            'vehicle_types': lane_vehicle_types[i]
        } for i in range(lane_count)]

        # Display results
        st.header("Traffic Analysis Results")

        # Traffic Light Visualization
        st.subheader("Real-time Signal Status")

        # Create HTML components for traffic light simulation
        traffic_light_css = create_traffic_light_css()
        traffic_light_html = create_traffic_light_html(lane_count)
        traffic_light_js = create_traffic_light_js(lane_info, cycle_time)

        # Combine and display
        traffic_light_component = f"{traffic_light_css}{traffic_light_html}{traffic_light_js}"
        st.components.v1.html(traffic_light_component, height=300)

        # Display detection results
        st.subheader("Detection Results")

        detection_cols = st.columns(lane_count)

        for i, output in enumerate(detection_outputs):
            with detection_cols[i]:
                st.write(f"Lane {output['lane']}")
                st.image(output['detection_image'], caption=f"Lane {output['lane']} - Detection", use_container_width=True)

        # Display signal timing analysis
        st.subheader("Signal Timing Analysis")

        # Display summary in a table
        timing_data = []
        for lane in lane_info:
            vehicle_types_str = ", ".join([f"{vehicle_type.title()}: {count}" for vehicle_type, count in lane['vehicle_types'].items() if count > 0])
            timing_data.append({
                "Lane": f"Lane {lane['lane_number']}",
                "Vehicle Distribution": vehicle_types_str,
                "Total Count": lane['vehicle_count'],
                "Green Time (s)": f"{lane['green_time']:.2f}"
            })

        st.table(timing_data)

        # Display summary cards
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Vehicles", total_vehicles)

        with col2:
            st.metric("Cycle Time", f"{cycle_time} seconds")

        # Display green time distribution as a bar chart
        st.subheader("Green Time Distribution")

        chart_data = {
            "Lane": [f"Lane {lane['lane_number']}" for lane in lane_info],
            "Green Time (seconds)": [lane['green_time'] for lane in lane_info]
        }

        st.bar_chart(chart_data, x="Lane", y="Green Time (seconds)")

if __name__ == "__main__":
    main()
