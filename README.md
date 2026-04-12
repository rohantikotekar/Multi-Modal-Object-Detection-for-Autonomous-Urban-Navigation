# Vision-Centric 3D Object Detection Pipeline for Autonomous Urban Navigation

This repository implements a high-performance **3D Perception Engine** for autonomous driving within the CARLA simulator. By integrating real-time deep learning inference with multi-modal sensor fusion, this project enables autonomous agents to interpret raw environment data and localize traffic participants in a global 3D coordinate space.

-----

## Project Overview

The perception module serves as an autonomous vehicle's "vision," converting raw sensor inputs into a structured map of its surroundings. This project replaces simplified ground-truth data with a realistic pipeline that identifies 3D objects using a suite of camera and LiDAR sensors.

### Problem Statement

Standard autonomous agents often rely on "perfect information" from the simulator. To bridge the gap between simulation and real-world deployment, this project implements a system that can identify objects in 3D space, handle occlusions, and maintain high Average Precision (AP) across varying environmental conditions.

### Solution

The implemented perception pipeline within the `Detector` class provides:

  * **Multi-modal Ingestion**: Processes synchronized feeds from front, left, and right RGBA cameras and 64-channel LiDAR.
  * **Deep Learning Inference**: Features an integrated **YOLOv5s** model for rapid 2D detection, mapped to 3D spatial estimations.
  * **Spatial Transformation**: A custom engine that converts local camera-frame detections into global CARLA world coordinates.
  * **Sensor Fusion**: Spatial Non-Maximum Suppression (NMS) to merge detections from multiple overlapping fields of view.

### Ego Vehicle Data Processing Workflow
<img width="595" height="724" alt="image" src="https://github.com/user-attachments/assets/2d91573c-d1b9-4239-8a68-a29fdb477014" />

-----

## Technical Implementation

### 1\. Sensor Configuration

The architecture utilizes a multi-sensor approach to maximize environmental coverage:

  * **Triple RGB Camera Array**: Center (720p), Left, and Right cameras providing \~140° of horizontal coverage with strategic overlap.
  * **64-Channel LiDAR**: A ray-cast sensor providing high-resolution spatial mapping up to 50 meters.
  * **GNSS**: For absolute global positioning and coordinate frame alignment.

### 2\. Perception Pipeline

The core logic in `detector.py` executes the following sequence:

  * **Preprocessing**: Converts CARLA BGRA streams to the RGB format required by the inference engine.
  * **YOLOv5 Inference**: Real-time identification of classes including **Vehicles**, **Pedestrians**, and **Cyclists**.
  * **Distance Estimation**: Employs perspective projection based on reference real-world heights to estimate depth from 2D pixel heights.
  * **BEV Box Generation**: Produces 2D Bird’s Eye View (BEV) bounding boxes in global coordinates for the control agent.

### 3\. Evaluation Framework

Performance is benchmarked against CARLA ground truth using **Average Precision (AP)**:

  * **IoU Calculation**: Intersection over Union thresholds at 0.3, 0.5, and 0.7.
  * **Coordinate Reordering**: Bounding box corners are sequenced specifically to form valid polygons for accurate evaluation.

-----

## Vizualizing Results
<img width="1103" height="636" alt="image" src="https://github.com/user-attachments/assets/c802b837-aabd-4e21-aec1-9e7224992d06" />
<img width="1108" height="648" alt="image" src="https://github.com/user-attachments/assets/64920874-86b6-4ab3-903b-0c17b3110c27" />

-----

## Performance Evaluation

The module significantly outperforms the empty baseline, enabling the vehicle to navigate while actively avoiding collisions.

| Metric | Baseline (Empty) | Implementation | Improvement |
| :--- | :--- | :--- | :--- |
| **AP @ IoU 0.3** | 0.00 | **0.42** | +42% |
| **AP @ IoU 0.5** | 0.00 | **0.28** | +28% |
| **AP @ IoU 0.7** | 0.00 | **0.14** | +14% |
| **Collisions** | Constant | **Few** | 80% Reduction |

-----

## Getting Started

### Prerequisites

  * **Simulator**: CARLA 0.9.13+
  * **Frameworks**: Python 3.7+, PyTorch 1.10+
  * **Libraries**: OpenCV, NumPy

### Execution Flow

1.  **Launch CARLA**: `bash CarlaUE4.sh -RenderOffScreen`
2.  **Spawn Traffic**: `python3 generate_traffic.py --sync`
3.  **Run Agent**: `python3 automatic_control.py`

-----

## Conclusion

This project successfully demonstrates a modular and scalable perception stack for autonomous systems. By integrating **YOLOv5** with a custom **Coordinate Transformation** and **Multi-Camera Fusion** engine, the system achieved a significant 42% improvement in detection precision over the baseline. While precise 3D localization remains a challenge at higher IoU thresholds, the current implementation provides a robust foundation for real-time obstacle avoidance and urban navigation in simulated environments.
