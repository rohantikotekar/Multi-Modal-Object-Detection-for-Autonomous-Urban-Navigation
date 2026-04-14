## **Multi-Modal Object Detection & Spatial Inference for Autonomous Urban Navigation**

This repository featurues a robust **3D Perception Stack** designed for autonomous navigation within the CARLA simulation environment. The architecture transitions from ground-truth dependency to an end-to-end inference pipeline that interprets high-dimensional sensor streams to localize traffic participants in real-time.

---

### **System Architecture & Design**

The perception module serves as the primary environmental interpreter for the autonomous agent. By replacing idealized ground-truth data with a multi-modal pipeline, the system processes raw sensor inputs to generate a structured 3D spatial map of the ego-vehicle's surroundings.

* **Objective**: To mitigate the domain gap between simulation and real-world deployment, the agent must process unrefined sensor data to identify objects under varying conditions and occlusions.
* **Inference Pipeline**: An asynchronous stack that ingests synchronized RGB and LiDAR data, executes deep learning inference, and projects detections into global coordinates for precise spatial reasoning.

---

### **Technical Implementation**

#### **1. Sensor Configuration & Modalities**
The vehicle is equipped with a comprehensive sensor suite to ensure high-resolution environmental awareness:
* **Stereo RGB Cameras**: Dual-perspective cameras (Left/Right) configured for wide-field semantic understanding.
* **64-Channel LiDAR**: High-resolution ray-cast sensor providing dense spatial mapping with a 50-meter effective range.
* **GNSS Integration**: Global navigation satellite system for coordinate frame alignment between local detections and the world map.

#### **2. Inference Engine (`detect()`)**
The `detect()` function manages the central intelligence of the module, handling complex data transformations and neural network execution:
* **Input Aggregation**: Processes a high-dimensional dictionary containing frame-synchronized RGBA imagery and LiDAR point clouds.
* **Architecture Support**: Optimized for state-of-the-art 3D backbones including **PointPillars**, **CenterFormer**, and **DETR3D** via **mmdetection3d** and **OpenPCDet** frameworks.
* **Output Specifications**: Generates 3D bounding boxes with associated semantic class labels (Vehicle, Pedestrian, Cyclist) and confidence scores.

#### **3. Evaluation & Benchmarking**
Detection performance is rigorously validated using **Average Precision (AP)** across multiple **Intersection over Union (IoU)** thresholds:
* **Precision Metrics**: AP is calculated at IoU 0.3, 0.5, and 0.7 to quantify localization accuracy and classification reliability.
* **Geometric Alignment**: Bounding box coordinates are re-indexed into valid polygons via `convert_format()` and `box_2_polygon()` to ensure mathematical correctness during evaluation against CARLA ground-truth data.

---

### **Deployment Instructions**

#### **Prerequisites**
* **CARLA Simulator 0.9.x**
* **Python 3.8+**
* **Deep Learning Environment**: PyTorch or TensorFlow (Docker recommended for dependency isolation)

#### **Execution Flow**
1.  **Initialize CARLA Server**: Establish the simulation environment.
2.  **Generate Traffic**: Populate the environment with background actors.
3.  **Launch Perception Engine**:
    ```bash
    python3 automatic_control.py -sync
    ```
    *The `-sync` flag is utilized to ensure deterministic execution and temporal alignment between the server and the agent.*

---
### Output
1. Visualization in Carla
<img width="1033" height="606" alt="image" src="https://github.com/user-attachments/assets/14b41ab7-9c2d-4f07-9e7d-49d4997bed53" />

2. GNSS
<img width="1036" height="605" alt="image" src="https://github.com/user-attachments/assets/61c31cc5-fa36-4b69-98f4-ad6f5ffb182f" />


### **Key Performance Features**
* **Real-Time Visualization**: Implements a Pygame interface for visual feedback, overlaying model detections against ground-truth participants.
* **Domain Adaptation**: Implementation paths for fine-tuning pre-trained models on simulator-specific data to minimize performance degradation caused by domain shift.

---

### **Conclusion**
This implementation provides a robust framework for 3D object detection by successfully transitioning from simplified simulator data to raw sensor-based inference. By integrating multi-modal fusion and high-resolution LiDAR processing, the system achieves the spatial accuracy and classification reliability required for complex autonomous navigation.
