## **3D Object Detection pipeline for Autonomous Urban Navigation**

This project focuses on building a high-tech **3D Perception Engine** for self-driving cars using the CARLA simulator. Instead of relying on "perfect" data from the simulator, this system teaches the car to "see" and understand its surroundings by processing raw data from cameras and sensors in real-time.

---

### **Project Overview**

Think of the perception module as the car's eyes and brain. It takes raw information and turns it into a map of nearby objects.

* **The Challenge**: Most simulation tests use "cheat" data where the computer already knows where everything is. To work in the real world, a car must be able to spot pedestrians and other vehicles even when they are partially hidden or in difficult weather.
* **The Solution**: I built a system that combines camera images and LiDAR (laser scanning). It uses deep learning to draw 3D boxes around objects, telling the car exactly where they are and what they are.
  
---

### **How It Works**

#### **1. The Sensors**
The car uses three main "senses" to understand the world:
* **Cameras**: Two cameras provide a wide view to help the car recognize shapes and colors.
* **LiDAR**: A laser sensor that scans the area up to 50 meters away, creating a detailed 3D map of distances.
* **GPS (GNSS)**: Keeps track of exactly where the car is on the global map.

#### **2. The Brain (Inference Engine)**
The system uses a function called `detect()` to process all this data:
* **Data Gathering**: It syncs the camera photos and laser scans so they match perfectly in time.
* **AI Processing**: It uses advanced AI models (like PointPillars and CenterFormer) to analyze the data.
* **Results**: It outputs 3D boxes labeled "Vehicle," "Pedestrian," or "Cyclist" with a confidence score.

#### **3. Measuring Success**
I checked the car's accuracy by comparing its "guesses" to the actual truth in the simulator. We look at how well the 3D boxes overlap with the real objects to ensure the car isn't just seeing ghosts or missing real hazards.

---

### **How to Run It**

#### **What You Need**
* CARLA Simulator
* Python 3.8 or higher
* PyTorch or TensorFlow

#### **Steps to Start**
1.  **Start CARLA**: Launch the simulator window.
2.  **Add Traffic**: Fill the city with other cars and walkers.
3.  **Run the Engine**:
    ```bash
    python3 automatic_control.py -sync
    ```
    *The `-sync` flag keeps the car and the simulator perfectly timed.*

---

### **Main Features**
* **Live Dashboard**: A screen shows you exactly what the car sees in real-time.
* **Smart Training**: The system is designed to bridge the gap between "simulation" and "reality," making the AI more reliable in different environments.

---

### **Conclusion**
This project successfully moves autonomous driving away from "perfect" simulator data and toward realistic sensor-based navigation. By combining LiDAR and Camera data through an advanced AI pipeline, the system can accurately identify and track traffic in complex urban settings. This is a vital step toward creating self-driving cars that are safe enough for real-world city streets.
