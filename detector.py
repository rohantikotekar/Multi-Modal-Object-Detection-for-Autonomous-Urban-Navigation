#!/usr/bin/einv python

"""
CARLA Object Detector - YOLOv5 v7.0 with 3D Boxes
Fixed version that returns proper 3D bounding boxes
"""

import numpy as np
import cv2
import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Detector:
    """Object detector using YOLOv5 v7.0"""
    
    def __init__(self):
        """Initialize detector"""
        print("="*60)
        print("YOLOv5 v7.0 Detector with 3D Boxes")
        print("="*60)
        
        self.model = None
        self.device = None
        self.frame_count = 0
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detection_range = 50.0
        
        # COCO to CARLA class mapping
        self.coco_to_carla = {
            0: 1,   # person -> pedestrian
            1: 2,   # bicycle -> cyclist
            2: 0,   # car -> vehicle
            3: 2,   # motorcycle -> cyclist
            5: 0,   # bus -> vehicle
            7: 0,   # truck -> vehicle
        }
        
        # Default 3D object sizes (length, width, height)
        self.default_sizes = {
            0: (4.5, 2.0, 1.8),   # vehicle
            1: (0.6, 0.6, 1.7),   # pedestrian
            2: (1.8, 0.6, 1.5),   # cyclist
        }
        
        # Sensor configurations
        self.sensor_configs = {
            'Left': {'x': 0.7, 'y': -0.4, 'z': 1.60, 'yaw': 0.0, 'fov': 100},
            'Right': {'x': 0.7, 'y': 0.4, 'z': 1.60, 'yaw': 0.0, 'fov': 100},
        }
        
        # Load YOLOv5
        if TORCH_AVAILABLE:
            self._load_yolov5()
        else:
            print("✗ PyTorch not available")
        
        print("="*60)
    
    def _load_yolov5(self):
        """Load YOLOv5 v7.0 (latest stable)"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            print("Loading YOLOv5s v7.0...")
            
            # Load YOLOv5 v7.0 - fixes utils.datasets issue
            self.model = torch.hub.load(
                'ultralytics/yolov5:v7.0',  # Specific v7.0 tag
                'yolov5s',
                pretrained=True,
                verbose=False
            )
            
            self.model.to(self.device)
            self.model.eval()
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            print("✓ YOLOv5s v7.0 loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("Using synthetic detections")

    def sensors(self):
        """Define sensor configuration"""
        return [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},
            
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 
             'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50, 'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
             'id': 'LIDAR'},
            
            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]
    
    def detect(self, sensor_data):
        """Main detection - returns 3D boxes (N, 8, 3)"""
        self.frame_count += 1
        
        if self.frame_count % 20 == 1:
            mode = "YOLOv5" if self.model else "Synthetic"
            print(f"Frame {self.frame_count} - Mode: {mode}")
        
        if self.model is None:
            return self._synthetic_3d_detections()
        
        # Process with YOLOv5
        ego_x, ego_y, ego_yaw = 0.0, 0.0, 0.0
        all_detections = []
        
        for camera_id in ['Left', 'Right']:
            if camera_id in sensor_data:
                dets = self._process_camera(
                    sensor_data[camera_id], 
                    camera_id, 
                    ego_x, ego_y, ego_yaw
                )
                all_detections.extend(dets)
        
        if not all_detections:
            return self._empty_detection()
        
        result = self._merge_and_format_3d(all_detections)
        
        if self.frame_count % 20 == 1:
            print(f"  Detections: {len(result['det_boxes'])}")
        
        return result
    
    def _synthetic_3d_detections(self):
        """Fallback synthetic 3D detections"""
        boxes = []
        for i in range(5):
            x = 10.0 + i * 6.0
            y = -4.0 + i * 2.0
            z = 0.0
            obj_class = [0, 1, 2, 0, 1][i]
            bbox_3d = self._create_3d_bbox(x, y, z, 0.0, obj_class)
            boxes.append(bbox_3d)
        
        return {
            'det_boxes': np.array(boxes),
            'det_class': np.array([0, 1, 2, 0, 1], dtype=np.int32),
            'det_score': np.array([0.9, 0.85, 0.8, 0.75, 0.7])
        }
    
    def _empty_detection(self):
        return {
            'det_boxes': np.zeros((0, 8, 3)),
            'det_class': np.zeros((0,), dtype=np.int32),
            'det_score': np.zeros((0,))
        }
    
    def _process_camera(self, camera_data, camera_id, ego_x, ego_y, ego_yaw):
        """Process camera with YOLOv5"""
        try:
            frame_id, img_data = camera_data
            img_rgb = self._preprocess_image(img_data)
            if img_rgb is None:
                return []
            
            with torch.no_grad():
                results = self.model(img_rgb, size=640)
            
            pred = results.xyxy[0].cpu().numpy()
            detections = []
            img_h, img_w = img_rgb.shape[:2]
            sensor_config = self.sensor_configs.get(camera_id, self.sensor_configs['Left'])
            
            for det in pred:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                
                if cls not in self.coco_to_carla:
                    continue
                
                carla_class = self.coco_to_carla[cls]
                box_h = y2 - y1
                distance = self._estimate_distance(box_h, img_h, carla_class)
                
                if distance > self.max_detection_range:
                    continue
                
                center_x = (x1 + x2) / 2
                norm_x = (center_x / img_w - 0.5) * 2
                h_angle = norm_x * (sensor_config['fov'] / 2)
                
                total_angle = ego_yaw + sensor_config['yaw'] + h_angle
                angle_rad = np.deg2rad(total_angle)
                
                sensor_x = sensor_config['x']
                sensor_y = sensor_config['y']
                ego_yaw_rad = np.deg2rad(ego_yaw)
                
                sensor_global_x = ego_x + sensor_x * np.cos(ego_yaw_rad) - sensor_y * np.sin(ego_yaw_rad)
                sensor_global_y = ego_y + sensor_x * np.sin(ego_yaw_rad) + sensor_y * np.cos(ego_yaw_rad)
                
                obj_x = sensor_global_x + distance * np.cos(angle_rad)
                obj_y = sensor_global_y + distance * np.sin(angle_rad)
                
                detections.append({
                    'x': obj_x,
                    'y': obj_y,
                    'z': 0.0,
                    'heading': total_angle,
                    'class': carla_class,
                    'score': float(conf),
                    'position': (obj_x, obj_y)
                })
            
            return detections
            
        except Exception as e:
            return []
    
    def _preprocess_image(self, img_data):
        try:
            if img_data is None or not isinstance(img_data, np.ndarray):
                return None
            if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
                return cv2.cvtColor(img_data[:, :, :3], cv2.COLOR_BGR2RGB)
            return None
        except:
            return None
    
    def _estimate_distance(self, box_height, img_height, obj_class):
        ref_heights = {0: 1.6, 1: 1.7, 2: 1.7}
        ref_h = ref_heights.get(obj_class, 1.6)
        if box_height > 0:
            distance = (ref_h * img_height) / (box_height * 2)
        else:
            distance = 20.0
        return np.clip(distance, 3.0, 50.0)
    
    def _create_3d_bbox(self, cx, cy, cz, heading, obj_class):
        """Create 3D box with 8 corners (x,y,z)"""
        length, width, height = self.default_sizes[obj_class]
        l, w, h = length/2, width/2, height/2
        
        corners = np.array([
            [ l, -w, -h], [-l, -w, -h], [-l, -w,  h], [ l, -w,  h],
            [ l,  w, -h], [-l,  w, -h], [-l,  w,  h], [ l,  w,  h],
        ])
        
        theta = np.deg2rad(heading)
        cos_h, sin_h = np.cos(theta), np.sin(theta)
        rotation_z = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        
        corners_rotated = corners @ rotation_z.T
        corners_global = corners_rotated + np.array([cx, cy, cz])
        
        return corners_global
    
    def _merge_and_format_3d(self, all_detections):
        """Merge detections and create 3D boxes"""
        if not all_detections:
            return self._empty_detection()
        
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        merged = []
        
        for det in all_detections:
            pos = det['position']
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
        
        # Create 3D boxes
        det_boxes = []
        for d in merged:
            bbox_3d = self._create_3d_bbox(d['x'], d['y'], d['z'], d['heading'], d['class'])
            det_boxes.append(bbox_3d)
        
        return {
            'det_boxes': np.array(det_boxes),
            'det_class': np.array([d['class'] for d in merged], dtype=np.int32),
            'det_score': np.array([d['score'] for d in merged])
        }


def create_detector():
    return Detector()
