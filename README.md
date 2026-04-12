# Visual-Inertial Odometry and Pose Estimation for Autonomous Vehicles

This repository implements a robust **Visual-Inertial Odometry (VIO) Pipeline** for autonomous vehicle localization within the CARLA simulator. By fusing high-frequency inertial data with visual feature tracking and depth information, this project enables precise 6-DoF pose estimation in environments where GPS may be unreliable.

-----

## Project Overview

The odometry module serves as the vehicle's internal positioning system, estimating motion by analyzing how the world moves relative to the onboard sensors. This project implements a comprehensive VIO system that addresses common visual odometry failures like scale drift and motion jitter.

### Problem Statement

Pure Visual Odometry (VO) often suffers from scale ambiguity and cumulative drift over long trajectories. To achieve reliable autonomous navigation, the system must accurately recover metric scale and maintain a smooth, continuous estimate of the vehicle’s position ($x, y, z$) and orientation ($roll, pitch, yaw$).

### Solution

The implemented VIO pipeline features:

  * **Multi-Sensor Fusion**: Synchronized processing of RGB cameras, Depth sensors, and IMU data.
  * **Feature-Based Tracking**: Utilizes OpenCV ORB features and RANSAC-based Essential Matrix estimation for motion tracking.
  * **Depth-Assisted Scale Recovery**: A median-depth calculation engine that anchors monocular visual motion to real-world metric units.
  * **EKF State Estimation**: An Extended Kalman Filter that fuses visual "measurements" with IMU "predictions" to filter noise and handle rapid maneuvers.

### Trajectory Comparison and Results

\<img width="775" height="560" alt="image" src="[https://github.com/user-attachments/assets/07ba3fd2-fd54-461e-8f92-22427590a419](https://github.com/user-attachments/assets/07ba3fd2-fd54-461e-8f92-22427590a419)" /\>

-----

## Technical Implementation

### 1\. Sensor Configuration

The architecture utilizes a rigidly attached sensor suite to maximize data fusion accuracy:

  * **RGB Frontal Camera**: Primary visual input for feature detection and tracking.
  * **Depth Camera**: Co-located with the RGB sensor to provide per-pixel distance maps for scale correction.
  * **IMU (Inertial Measurement Unit)**: Provides high-frequency linear acceleration and angular velocity readings.
  * **Lidar**: Integrated to enhance environmental mapping and verification.

### 2\. Localization Pipeline

The core logic executes the following sequence:

  * **Visual Odometry**: Detects and matches ORB keypoints between consecutive frames to recover relative rotation ($R$) and translation ($t$).
  * **Scale Correction**: Decodes CARLA depth maps to calculate the median scene depth, scaling the visual translation vector to meters.
  * **Inertial Propagation**: Integrates IMU data to predict the vehicle's state between camera frames and remove gravity vectors.
  * **Global Alignment**: Transforms camera-frame coordinates into the global CARLA world frame using initial sensor transforms.

### 3\. Evaluation Framework

Performance is benchmarked against CARLA ground truth using **Absolute Trajectory Error (ATE)**:

  * **Position Metrics**: RMSE, Mean, and Median error calculation in meters.
  * **Rotation Metrics**: RMSE and Mean orientation error in radians.
  * **Synchronous Execution**: Utilizes CARLA's sync mode to ensure fixed time steps ($dt$) for stable EKF integration.

-----

## Performance Evaluation

The module provides smooth and continuous pose estimates, significantly reducing the jitter associated with raw visual tracking.

| Metric | Result |
| :--- | :--- |
| **RMSE Position Error** | 40.98 meters |
| **RMSE Rotation Error** | 0.52 radians |
| **Mean Position Error** | 32.68 meters |
| **Median Position Error** | 31.29 meters |

-----

## Getting Started

### Prerequisites

  * **Simulator**: CARLA 0.9.15
  * **Languages**: Python 3.x
  * **Libraries**: OpenCV, NumPy, SciPy

### Execution Flow

1.  **Launch CARLA**: `./CarlaUE4.sh -RenderOffScreen`
2.  **Spawn Traffic**: `python generate_traffic.py`
3.  **Run Odometry**: `python automatic_control.py -sync`

-----

## Conclusion

This project successfully demonstrates a modular and scalable Visual-Inertial Odometry stack for autonomous systems. By integrating ORB-based visual tracking with an Extended Kalman Filter and depth-assisted scale recovery, the system provides a reliable foundation for real-time localization. While challenges like Z-axis coordinate drift and sensitivity to textureless scenes were observed, the application of world-frame anchoring and sensor fusion effectively minimized total trajectory error, proving the system's viability for urban navigation in simulated environments.
