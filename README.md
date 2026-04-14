## **Advancing Autonomous Perception: 3D Object Detection in Urban Environments**

[cite_start]This repository features the development and integration of a high-fidelity **3D Perception Engine** for autonomous driving within the CARLA simulator[cite: 4, 5]. [cite_start]The core objective is to move beyond idealized simulation data by architecting an inference pipeline that interprets raw sensor streams to identify and localize traffic participants in real-time[cite: 6, 28].

---

### **Project Overview**

The perception module serves as an autonomous vehicle's vision, converting raw sensor inputs into a clear, structured map of its surroundings. [cite_start]This project replaces simplified ground-truth data with a realistic pipeline that detects 3D objects using camera and LiDAR sensors[cite: 5, 6].

* [cite_start]**Problem Statement**: Standard autonomous systems often rely on "oracle" data from the simulator[cite: 19, 20]. [cite_start]To prepare for real-world deployment, agents must instead process high-dimensional sensor data to recognize objects under varying conditions and occlusions[cite: 228, 229, 231].
* [cite_start]**The Solution**: A multi-modal perception stack that ingests synchronized RGB and LiDAR data, processes it through deep learning models, and outputs 3D bounding boxes in global coordinates for precise navigation[cite: 6, 196, 202].

---

### **Technical Architecture**

#### **1. Sensor Fusion & Configuration**
[cite_start]The system is equipped with a versatile sensor suite to ensure comprehensive environmental awareness[cite: 60, 61]:
* [cite_start]**Stereo RGB Cameras**: Dual-perspective cameras (Left/Right) configured for wide-field semantic understanding[cite: 44, 45, 48].
* [cite_start]**64-Channel LiDAR**: High-resolution ray-cast sensor providing dense spatial mapping up to 50 meters[cite: 51, 53, 54].
* [cite_start]**GNSS Integration**: Global positioning for accurate coordinate frame alignment between local detections and the world map[cite: 57, 217].

#### **2. Inference Engine & Pipeline**
[cite_start]The `detect()` function serves as the central intelligence of the module, handling complex data transformations[cite: 196, 197]:
* [cite_start]**Input Aggregation**: Processes a high-dimensional dictionary containing frame-synchronized images and LiDAR point clouds[cite: 199, 202].
* [cite_start]**Deep Learning Integration**: Built to support state-of-the-art architectures from **OpenMMLab (mmdetection3d)** and **OpenPCDet**, such as PointPillars, CenterFormer, and DETR3D[cite: 211, 212, 213].
* [cite_start]**Output Specifications**: Generates 3D bounding boxes with associated class labels (Vehicle, Pedestrian, Cyclist) and confidence scores[cite: 202, 203, 205, 207].

#### **3. Evaluation & Benchmarking**
[cite_start]Performance is rigorously validated using Average Precision (AP) metrics across multiple Intersection over Union (IoU) thresholds[cite: 29]:
* [cite_start]**Precision Metrics**: AP is calculated at IoU 0.3, 0.5, and 0.7 to quantify localization accuracy[cite: 33, 34].
* [cite_start]**Spatial Alignment**: Coordinates are reordered into valid polygons to ensure mathematical correctness during evaluation against CARLA ground truth[cite: 38, 39, 227].

---

### **Deployment Instructions**

#### **Prerequisites**
* [cite_start]CARLA Simulator 0.9.x [cite: 7]
* Python 3.8+
* [cite_start]PyTorch / TensorFlow Environment (Docker recommended for dependency management) [cite: 214, 221]

#### **Execution Flow**
1.  [cite_start]**Initialize CARLA Server**: Start the simulation environment[cite: 22].
2.  [cite_start]**Generate Traffic**: Populate the map with background actors[cite: 23, 24].
3.  **Launch Perception Engine**:
    ```bash
    python3 automatic_control.py -sync
    ```
    [cite_start]*The `-sync` flag is utilized to ensure deterministic execution and data alignment between the server and the agent[cite: 27].*

---

### **Key Performance Features**
* [cite_start]**Multi-Modal Visualization**: Real-time feedback via a Pygame interface, allowing for side-by-side comparison of model detections against ground-truth participants[cite: 63, 64, 65].
* [cite_start]**Domain Adaptation**: Implementation paths for fine-tuning pre-trained models on simulator-specific data to bridge the gap between general training sets and the CARLA environment[cite: 230, 231, 233].
