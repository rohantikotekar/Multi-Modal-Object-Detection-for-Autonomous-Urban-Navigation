#!/usr/bin/env python

"""
CARLA Object Detector - Working Version
Simplified approach with 2D Bird's Eye View detection
Converts camera detections to proper 2D bounding boxes in global coordinates
"""

import numpy as np
import cv2
import torch
import sys
import os
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import carla
except ImportError:
    print("Warning: CARLA module not found. Some features may not work.")


class Detector:
    """
    Simplified object detector for CARLA that outputs 2D BEV boxes
    """
    
    def __init__(self, carla_world):
        """
        Initialize the detector
        
        Args:
            carla_world: CARLA world instance
        """
        self.world = carla_world
        self.model = None
        self.device = None
        
        # Detection parameters
        self.confidence_threshold = 0.30
        self.iou_threshold = 0.45
        self.max_detection_range = 50.0
        
        # Load model
        self._load_model()
        
        # COCO to CARLA class mapping
        self.coco_to_carla = {
            0: 1,   # person -> pedestrian
            1: 2,   # bicycle -> cyclist  
            2: 0,   # car -> vehicle
            3: 2,   # motorcycle -> cyclist
            5: 0,   # bus -> vehicle
            7: 0,   # truck -> vehicle
        }
        
        # Default object sizes (length, width) in meters for BEV
        self.default_sizes = {
            0: (4.5, 2.0),   # vehicle
            1: (0.6, 0.6),   # pedestrian
            2: (1.8, 0.6),   # cyclist
        }
        
        print("Detector initialized")
        
    def _load_model(self):
        """Load YOLOv5 model"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            print("Loading YOLOv5s model...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            print("YOLOv5 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def sensors(self):
        """
        Define sensor configuration
        Using a simple front-facing camera setup
        """
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0, 'z': 1.8,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1280, 'height': 720, 'fov': 100,
                'id': 'Center'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.5, 'y': -0.4, 'z': 1.8,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -35.0,
                'width': 800, 'height': 600, 'fov': 100,
                'id': 'Left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.5, 'y': 0.4, 'z': 1.8,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 35.0,
                'width': 800, 'height': 600, 'fov': 100,
                'id': 'Right'
            },
        ]
        
        return sensors
    
    def detect(self, sensor_data, ego_vehicle):
        """
        Main detection function
        
        Args:
            sensor_data: Dictionary with sensor data
            ego_vehicle: Ego vehicle actor
            
        Returns:
            Dictionary with:
                - det_boxes: (N, 4, 2) for 2D BEV boxes OR (N, 8, 3) for 3D boxes
                - det_class: (N,) object classes
                - det_score: (N,) confidence scores
        """
        if self.model is None:
            return self._empty_detection()
        
        all_detections = []
        
        # Get ego vehicle info
        ego_transform = ego_vehicle.get_transform()
        
        # Process each camera
        for sensor_id in ['Center', 'Left', 'Right']:
            if sensor_id not in sensor_data:
                continue
                
            frame_id, sensor_img = sensor_data[sensor_id]
            
            # Preprocess image
            img = self._preprocess_image(sensor_img)
            if img is None:
                continue
            
            # Get sensor config
            sensor_config = self._get_sensor_config(sensor_id)
            
            # Run detection
            detections = self._detect_objects(img, sensor_id, sensor_config, ego_transform)
            
            if detections:
                all_detections.extend(detections)
        
        # Merge detections
        if not all_detections:
            return self._empty_detection()
        
        return self._merge_and_format(all_detections)
    
    def _empty_detection(self):
        """Return empty detection"""
        return {
            'det_boxes': np.zeros((0, 4, 2)),  # 2D BEV format
            'det_class': np.zeros((0,), dtype=np.int32),
            'det_score': np.zeros((0,))
        }
    
    def _get_sensor_config(self, sensor_id):
        """Get sensor configuration"""
        configs = {
            'Center': {'x': 2.0, 'y': 0.0, 'z': 1.8, 'yaw': 0.0, 'fov': 100},
            'Left': {'x': 1.5, 'y': -0.4, 'z': 1.8, 'yaw': -35.0, 'fov': 100},
            'Right': {'x': 1.5, 'y': 0.4, 'z': 1.8, 'yaw': 35.0, 'fov': 100},
        }
        return configs.get(sensor_id, configs['Center'])
    
    def _preprocess_image(self, carla_image):
        """Convert CARLA image to RGB"""
        try:
            if carla_image is None or not isinstance(carla_image, np.ndarray):
                return None
            
            # Convert to RGB
            if len(carla_image.shape) == 3:
                if carla_image.shape[2] == 4:
                    img = cv2.cvtColor(carla_image, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(carla_image, cv2.COLOR_BGR2RGB)
            else:
                return None
            
            return img
            
        except Exception as e:
            print(f"Preprocess error: {e}")
            return None
    
    def _detect_objects(self, img, sensor_id, sensor_config, ego_transform):
        """
        Detect objects in image and convert to global coordinates
        """
        try:
            # Run YOLOv5
            results = self.model(img, size=640)
            pred = results.xyxy[0].cpu().numpy()
            
            detections = []
            img_h, img_w = img.shape[:2]
            
            for det in pred:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                
                if cls not in self.coco_to_carla:
                    continue
                
                carla_class = self.coco_to_carla[cls]
                
                # Calculate object position
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                box_h = y2 - y1
                
                # Estimate distance from box height
                distance = self._estimate_distance(box_h, img_h, carla_class)
                
                if distance > self.max_detection_range:
                    continue
                
                # Calculate horizontal angle
                norm_x = (center_x / img_w - 0.5) * 2
                h_angle = norm_x * (sensor_config['fov'] / 2)
                total_angle = sensor_config['yaw'] + h_angle
                
                # Convert to global position
                global_pos = self._to_global_position(
                    distance, total_angle, 
                    sensor_config, ego_transform
                )
                
                if global_pos is None:
                    continue
                
                # Create 2D BEV bounding box
                bbox_2d = self._create_bev_bbox(
                    global_pos, total_angle + ego_transform.rotation.yaw,
                    carla_class
                )
                
                detections.append({
                    'bbox': bbox_2d,
                    'class': carla_class,
                    'score': float(conf),
                    'position': global_pos
                })
            
            return detections
            
        except Exception as e:
            print(f"Detection error on {sensor_id}: {e}")
            return []
    
    def _estimate_distance(self, box_height, img_height, obj_class):
        """
        Estimate distance based on box height
        """
        # Reference heights in meters
        ref_heights = {0: 1.6, 1: 1.7, 2: 1.7}  # vehicle, pedestrian, cyclist
        
        ref_h = ref_heights.get(obj_class, 1.6)
        
        # Simple perspective formula
        if box_height > 0:
            distance = (ref_h * img_height) / (box_height * 2)
        else:
            distance = 20.0
        
        return np.clip(distance, 3.0, 50.0)
    
    def _to_global_position(self, distance, angle_deg, sensor_config, ego_transform):
        """
        Convert sensor-relative position to global coordinates
        """
        try:
            # Ego vehicle position and orientation
            ego_x = ego_transform.location.x
            ego_y = ego_transform.location.y
            ego_yaw = ego_transform.rotation.yaw
            
            # Sensor offset in ego vehicle frame
            sensor_x = sensor_config['x']
            sensor_y = sensor_config['y']
            
            # Convert angle to radians
            ego_yaw_rad = np.deg2rad(ego_yaw)
            total_angle_rad = np.deg2rad(ego_yaw + angle_deg)
            
            # Sensor position in global frame
            sensor_global_x = ego_x + sensor_x * np.cos(ego_yaw_rad) - sensor_y * np.sin(ego_yaw_rad)
            sensor_global_y = ego_y + sensor_x * np.sin(ego_yaw_rad) + sensor_y * np.cos(ego_yaw_rad)
            
            # Object position relative to sensor
            obj_dx = distance * np.cos(total_angle_rad)
            obj_dy = distance * np.sin(total_angle_rad)
            
            # Object global position
            obj_x = sensor_global_x + obj_dx
            obj_y = sensor_global_y + obj_dy
            
            return (obj_x, obj_y)
            
        except Exception as e:
            print(f"Position conversion error: {e}")
            return None
    
    def _create_bev_bbox(self, center_pos, heading_deg, obj_class):
        """
        Create 2D BEV bounding box (4 corners)
        Returns box in format that can form a polygon when connected in order
        """
        cx, cy = center_pos
        length, width = self.default_sizes[obj_class]
        
        # Half dimensions
        hl = length / 2
        hw = width / 2
        
        # Create corners in local frame (front-right-back-left order for polygon)
        corners_local = np.array([
            [ hl, -hw],  # front-right
            [ hl,  hw],  # front-left
            [-hl,  hw],  # back-left
            [-hl, -hw],  # back-right
        ])
        
        # Rotate by heading
        heading_rad = np.deg2rad(heading_deg)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)
        
        rotation = np.array([
            [cos_h, -sin_h],
            [sin_h,  cos_h]
        ])
        
        corners_rotated = corners_local @ rotation.T
        
        # Translate to center
        corners_global = corners_rotated + np.array([cx, cy])
        
        return corners_global
    
    def _merge_and_format(self, all_detections):
        """
        Merge detections and format output
        """
        if not all_detections:
            return self._empty_detection()
        
        # Sort by confidence
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # Simple NMS based on position
        merged = []
        
        for det in all_detections:
            pos = det['position']
            
            # Check for duplicates
            is_dup = False
            for m in merged:
                m_pos = m['position']
                dist = np.sqrt((pos[0] - m_pos[0])**2 + (pos[1] - m_pos[1])**2)
                
                threshold = 2.5 if det['class'] == 0 else 1.5
                
                if det['class'] == m['class'] and dist < threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                merged.append(det)
        
        if not merged:
            return self._empty_detection()
        
        # Format output
        det_boxes = np.array([d['bbox'] for d in merged])
        det_class = np.array([d['class'] for d in merged], dtype=np.int32)
        det_score = np.array([d['score'] for d in merged])
        
        print(f"Detected: {len(merged)} objects "
              f"(V:{np.sum(det_class==0)} P:{np.sum(det_class==1)} C:{np.sum(det_class==2)})")
        
        return {
            'det_boxes': det_boxes,
            'det_class': det_class,
            'det_score': det_score
        }


def create_detector(carla_world):
    """Factory function"""
    return Detector(carla_world)


if __name__ == "__main__":
    print("CARLA Detector Module - Simplified 2D BEV Version")
    print("Use with automatic_control.py")
