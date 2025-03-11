## INTRODUCTION
This project develops an AI-based dynamic traffic signal system using object detection to optimize traffic flow. It analyzes real-time traffic data and adjusts signal timings based on vehicle density. It also describes the accompanying Flask-based web application that demonstrates how to use the trained model for lane-wise vehicle detection and green-time calculation.
### Deployment Link
You can try the live demo of the system at the following URL:  (Takes a lot of time for processing as it is deployed in free instance in AWS, Refer to Youtube Video below. Will update to a better instance later)
[AI-Powered Smart Traffic Management System](http://ec2-18-226-222-74.us-east-2.compute.amazonaws.com:5000/)

### YouTube Video Demo
Check out the demonstration of the project on YouTube:  
[YouTube Demo Video](https://www.youtube.com/watch?v=gQNsI5qtbt4)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Web Application Overview](#web-application-overview)
3. [Vehicle Class Mapping](#vehicle-class-mapping)
4. [Screenshots](#screenshots)
5. [How to Use This Repository](#how-to-use-this-repository)
6. [Dataset & Model](#dataset--model)
7. [Training Process](#training-process)
8. [Training Logs & Results](#training-logs--results)
9. [Training Metrics & Plots](#training-metrics--plots)
10. [Future Scope](#future-scope)
11. [References](#references)

---
## WEB APPLICATION OVERVIEW
We developed a Flask-based web application to demonstrate real-time lane-wise vehicle counting and dynamic green-time calculation using the trained model.

### Key Features
- **Multiple Lane Input:** Users can specify how many lanes they want to analyze.
- **Automatic Detection:** The YOLOv9 model (`best.pt`) is loaded onto the Flask server.
- **Image & Video Support:** Each lane can have an uploaded image or video file.
- **Vehicle Counting:** Counts the number of recognized vehicles by type.
- **Dynamic Signal Timing:** Calculates a proportional green-time for each lane based on total vehicle count.

### Workflow Summary
1. The user enters the number of lanes.
2. For each lane, the user uploads an image or video.
3. The application runs inference with the YOLOv9 model.
4. Detections are drawn, and total vehicle counts are aggregated.
5. A dynamic green-time (in seconds) is computed for each lane.
6. Results are displayed with detection outputs and recommended signal times.

---

## VEHICLE CLASS MAPPING

| Vehicle Type | Class Index |
|-------------|------------|
| Car         | 2          |
| Truck       | 7          |
| Bus         | 5          |
| Motorbike   | 3          |
| Bicycle     | 1          |

---

## SCREENSHOTS

<img src="https://github.com/user-attachments/assets/c87e1133-2dfc-4445-9565-6df8ca71e35e" width="400">

---

## HOW TO USE THIS REPOSITORY

1. Clone the repository and navigate to the project directory:
   ```sh
   git clone ai-powered-smart-traffic-management-system
   cd ai-powered-smart-traffic-management-system/app
   ```
2. Ensure you have a Python 3.x environment set up.
3. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Place your trained YOLO weights (`best.pt`) in the designated location.
5. Launch the Flask app:
   ```sh
   python app.py
   ```
6. Open your web browser and go to the indicated localhost address.
7. Select the number of lanes, upload images/videos for each lane, and click **“Analyze Traffic.”**

---

## DATASET & MODEL

### Dataset: "Street View Dataset (Roboflow Universe)"
- **Total Images:** 8,693

#### Dataset Split:
| Set         | Images  | 
|------------|--------|
| Train Set  | 7,566  |
| Validation | 805    | 
| Test Set   | 322    | 

#### Preprocessing:
- **Auto-Orient:** Applied
- **Resize:** Fill (with center crop) to 640×640

#### Augmentations:
- **Outputs per training example:** 3
- **Crop:** 0% Minimum Zoom, 28% Maximum Zoom
- **Exposure:** Between -10% and +10%

### Model: YOLOv9
- Balances speed and accuracy, suitable for real-time traffic control
- Enhanced feature extraction layers for both small and large vehicles
- Scalable for deployment on edge devices or the cloud

---

## TRAINING PROCESS

- Link of the YOLOv9 training notebook: https://github.com/suma-vatturi/ai-powered-smart-traffic-management-system/blob/main/training_notebook.ipynb

### Data Preparation & Annotation
- Employed an open-source labeling tool (or Roboflow annotation interface).
- Maintained uniform class naming conventions (e.g., car, bus, truck, etc.).
- Ensured accurate bounding boxes for robust vehicle detection.

### Data Augmentation
- **Techniques:** Random flipping, rotation, scaling, hue/saturation adjustments.
- **Purpose:** Increases training sample diversity and reduces overfitting.

### Fine-Tuning
- Adjusted learning rate, batch size, and IoU thresholds to optimize mAP.
- Focused on balancing precision and recall.

### Model Evaluation & Key Metrics
- **Validation & Testing:**
  - mAP(0.5), mAP(0.5:0.95), precision, recall, and F1-score.
  - YOLOv9’s speed (FPS) ensures near real-time detection.

---

## TRAINING LOGS & RESULTS

### Best YOLOv9 Model on Validation Set:
- **mAP_0.5:** 0.944
- **mAP_0.5:0.95:** 0.749
- **Precision:** 0.906
- **Recall:** 0.910

### Class-specific performance (mAP_0.5 Accuracy):
| Vehicle Type | mAP_0.5:0.95 |
|-------------|-------------|
| All         | 0.944       |
| Bicycle     | 0.942       |
| Bus         | 0.978       |
| Car         | 0.968       |
| Motorbike   | 0.901       |
| Person      | 0.88        |
| Truck       | 0.995       |

---

## TRAINING METRICS & PLOTS

- **F1 Score Curve:**  
  <img src="https://github.com/user-attachments/assets/a3c2cf42-6897-45ef-a916-fbceafe5ee98" width="400">

- **Precision-Recall Curve:**  
  <img src="https://github.com/user-attachments/assets/851bb2d9-3ba3-41b0-b0cb-6e0bd4b74f78" width="400">

- **Confusion Matrix:**  
  <img src="https://github.com/user-attachments/assets/86fcedd8-2b06-4cb3-864c-8188fe4e12b4" width="400">

- **Training Results:**  
  <img src="https://github.com/user-attachments/assets/842a9023-ef12-4264-9492-47c033217281" width="600">

---


## FUTURE SCOPE
- **Priority-Based Traffic Control:** Automatically detect and prioritize ambulances, police vehicles, etc.
- **Load Balancing:** Differentiate large vehicles (trucks/buses) vs. smaller vehicles to reduce congestion.
- **Predictive Analytics:** Use historical data and forecasting models to preemptively adjust signals.

---
## REFERENCES
- FSMVU, **“Street View Dataset,”** Roboflow Universe, 2023.  
  - [Dataset URL](https://universe.roboflow.com/fsmvu/street-view-gdogo)  

- Redmon, J., Farhadi, A., **“YOLO Series,”** arXiv.  
  - [YOLO Paper](https://arxiv.org/abs/1506.02640)  

- **YOLOv9 Colab Custom Training Notebook by Roboflow**  
  - [Notebook URL](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov9-object-detection-on-custom-dataset.ipynb#scrollTo=pixgo4qnjdoU)  

- **SCATS** (Sydney Coordinated Adaptive Traffic System)  

- **SCOOT** (Split Cycle Offset Optimization Technique)  

- **TrafNet: A Deep Neural Network for Real-Time Traffic Signal Control**  
  - Wei Wei, Xueting Wang, et al., IEEE Transactions on Intelligent Transportation Systems, 2019.  
  - [Paper URL](https://ieeexplore.ieee.org/document/8869792)  

- **A Survey on Traffic Light Control Methods for Efficient Urban Traffic Management**  
  - Abhishek Gupta, Rajesh Sharma, et al., Journal of Advanced Transportation, 2021.  
  - [Paper URL](https://www.hindawi.com/journals/jat/2021/6691675/)  

- **YOLOv9: State-of-the-Art Object Detection**  
  - Ultralytics, 2024.  
  - [YOLOv9 GitHub](https://github.com/WongKinYiu/yolov9)

- **Real-time Traffic Congestion Detection Using Computer Vision and Deep Learning**  
  - Lin, C.-H., and Chen, L.-H., 2022.  
  - [Paper URL](https://www.sciencedirect.com/science/article/abs/pii/S2352146522000985)  

- **OpenTraffic Dataset: Real-World Traffic Flow Data for AI-Based Traffic Management**  
  - OpenTraffic, 2022.  
  - [Dataset URL](https://www.opentraffic.io/)  


---
