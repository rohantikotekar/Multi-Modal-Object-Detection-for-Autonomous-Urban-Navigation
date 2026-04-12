# Vision-Centric 3D Object Detection Pipeline for Autonomous Urban Navigation

This repository contains the implementation of a 3D perception module designed for autonomous driving agents within the CARLA simulator. The project focuses on real-time detection of traffic participants (vehicles, pedestrians, and cyclists) using multi-modal sensor fusion and pre-trained deep learning architectures.

---

## Project Overview

The perception module serves as an autonomous vehicle's vision, converting raw sensor inputs into a clear, structured map of its surroundings. This project replaces simplified ground-truth data with a realistic pipeline that detects 3D objects using camera and LiDAR sensors.

### Problem Statement
Standard autonomous agents often rely on ground-truth data directly from the simulator. To bridge the gap between simulation and real-world deployment, we must implement a detection system that can identify objects in 3D space, handle occlusions, and operate under varying environmental conditions while maintaining high average precision (AP) across different Intersection over Union (IoU) thresholds.

### Solution
I have implemented a perception pipeline within the `Detector` class that:
* **Ingests Multi-modal Data:** Processes synchronized RGBA camera feeds and LiDAR point clouds.
* **Inference Engine:** Integrates state-of-the-art pre-trained models (via OpenMMLab/OpenPCDet) to perform 3D object detection.
* **Spatial Transformation:** Converts local sensor-frame detections into global coordinates for downstream path planning and control.
* **Evaluation:** Benchmarks performance against ground truth using Average Precision (AP) at IoUs of 0.3, 0.5, and 0.7.

### Ego vehicle data processing workflow
<img width="668" height="744" alt="image" src="https://github.com/user-attachments/assets/3d2a28c7-d7f9-4c4a-b9ee-31160408720b" />

---

## Technical Implementation

### 1. Sensor Configuration
The system is designed with a flexible sensor suite. The default configuration includes:
* **Dual RGB Cameras:** Left and Right cameras for wide-angle visual coverage.
* **64-Channel LiDAR:** A ray-cast sensor with a 50-meter range for precise spatial mapping.
* **GNSS:** For global positioning and coordinate frame alignment.

### 2. Perception Pipeline
The core logic resides in the `detect()` function, which follows these stages:
* **Data Pre-processing:** Normalizes sensor input and aligns frames.
* **Model Inference:** Utilizes models such as **PointPillar** or **CenterNet** to extract bounding boxes.
* **Output Formatting:** Generates detection boxes in $(N, 8, 3)$ or $(N, 4, 2)$ shapes, including class labels (Vehicle, Pedestrian, Cyclist) and confidence scores.

### 3. Coordinate Alignment
To ensure the `automatic_control.py` agent reacts correctly, detected objects are mapped from the camera/LiDAR frame to the global simulation frame. This involves reordering bounding box corner coordinates into valid polygons to ensure accurate IoU calculation by the evaluation module.

---

## Getting Started

### Prerequisites
* CARLA Simulator (0.9.x)
* Python 3.8+
* PyTorch or TensorFlow (depending on the chosen model backbone)
* OpenMMLab / mmdetection3d (recommended)

### Installation & Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rohantikotekar/perception_lab1.git
    cd perception_lab1
    ```
2.  **Environment Setup:**
    Follow the provided Docker instructions or install dependencies manually to ensure consistency with the pre-trained model environment.

### Execution Flow
1.  **Start CARLA Server:**
    Ensure the CARLA simulator is running.
2.  **Generate Traffic:**
    ```bash
    python3 generate_traffic.py
    ```
3.  **Run Perception & Control:**
    ```bash
    python3 automatic_control.py -sync
    ```

---

## Performance Evaluation

The module is evaluated based on its ability to detect objects within a 50-meter radius. Performance metrics are printed upon termination, displaying:
* Total object instances processed.
* Average Precision (AP) at IoU 0.3, 0.5, and 0.7.

*Note: Due to occlusions and the "all-encompassing" nature of the ground truth, AP scores reflect the model's ability to detect visible objects vs. theoretical ground truth.*

---

## Repository Structure
* `detector.py`: Main perception logic and sensor configuration.
* `automatic_control.py`: The autonomous agent relying on the detector's output.
* `eval.py`: Evaluation script for calculating IoU and AP.
* `generate_traffic.py`: Script to populate the CARLA environment.
