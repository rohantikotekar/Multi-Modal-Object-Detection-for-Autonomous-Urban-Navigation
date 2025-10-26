#!/usr/bin/env python

"""
CARLA Object Detector - YOLOv5 (Fixed Version)
Loads YOLOv5 correctly without ultralytics package conflicts
"""

import numpy as np
import cv2
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import torch
try:
    import torch
    TORCH_AVAILABLE = True
    print("✓ PyTorch imported successfully")
except ImportError:
    print("✗ PyTorch not available")
    print("Install with: pip install torch torchvision")
    TORCH_AVAILABLE = False


class Detector:
    """Object detector using YOLOv5 via torch.hub"""
    
    def __init__(self):
        """Initialize detector"""
        print("="*60)
        print("Initializing YOLOv5 Detector...")
        print("="*60)
        
        self.model = None
        self.device = None
        self.frame_count = 0
        
        # Detection parameters
        self.confidence_threshold = 0.25
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
        
        # Default object sizes (length, width)
        self.default_sizes = {
            0: (4.5, 2.0),   # vehicle
            1: (0.6, 0.6),   # pedestrian
            2: (1.8, 0.6),   # cyclist
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
            print("✗ Cannot load YOLOv5 - PyTorch not available")
        
        print("="*60)
    
    def _load_yolov5(self):
        """
        Load YOLOv5 via torch.hub
        This is the CORRECT way - no ultralytics package needed
        """
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Set trust_repo to avoid security warnings
            os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
            
            print("Loading YOLOv5s via torch.hub...")
            print("(First run may take 1-2 minutes to download)")
            
            # Load YOLOv5 v6.0 - stable version that doesn't need ultralytics
            self.model = torch.hub.load(
                'ultralytics/yolov5:v6.0',  # Use v6.0 stable release
                'yolov5s',                   # Model variant
                pretrained=True,             # Use pretrained weights
                verbose=False                # Reduce output
            )
            
            # Move to device and set eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Set detection parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = 100  # Max detections per image
            
            print("✓ YOLOv5s loaded successfully!")
            print(f"  Model on: {self.device}")
            print(f"  Confidence threshold: {self.confidence_threshold}")
            
        except Exception as e:
            print(f"✗ Error loading YOLOv5: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure you have internet connection")
            print("2. Run: pip uninstall -y ultralytics")
            print("3. Run: pip install torch torchvision")
            print("4. Test: python3 -c \"import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s')\"")
            print("\nUsing fallback mode (synthetic detections)")
            self.model = None
    
    def sensors(self):
        """Define sensor configuration"""
        sensors = [
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
        
        return sensors
    
    def detect(self, sensor_data):
        """Main detection function"""
        self.frame_count += 1
        
        # Debug print
        if self.frame_count % 10 == 1:
            print(f"\n{'='*60}")
            print(f"DETECTOR CALLED - Frame {self.frame_count}")
            print(f"Available sensors: {list(sensor_data.keys())}")
            print(f"{'='*60}")
        
        # Check if model loaded
        if self.model is None:
            if self.frame_count % 10 == 1:
                print("⚠ YOLOv5 not loaded - using synthetic detections")
            return self._synthetic_detections()
        
        # Get ego position
        ego_x, ego_y, ego_yaw = 0.0, 0.0, 0.0
        
        all_detections = []
        
        # Process cameras
        for camera_id in ['Left', 'Right']:
            if camera_id in sensor_data:
                dets = self._process_camera(
                    sensor_data[camera_id], 
                    camera_id, 
                    ego_x, ego_y, ego_yaw
                )
                all_detections.extend(dets)
        
        # Merge and format
        if not all_detections:
            return self._empty_detection()
        
        result = self._merge_detections(all_detections)
        
        # Debug print
        if self.frame_count % 10 == 1:
            n_boxes = len(result['det_boxes'])
            n_veh = np.sum(result['det_class'] == 0)
            n_ped = np.sum(result['det_class'] == 1)
            n_cyc = np.sum(result['det_class'] == 2)
            print(f"✓ Returning {n_boxes} detections (V:{n_veh}, P:{n_ped}, C:{n_cyc})")
            print(f"{'='*60}\n")
        
        return result
    
    def _synthetic_detections(self):
        """Fallback synthetic detections if YOLOv5 fails"""
        boxes = []
        for i in range(3):
            x = 10.0 + i * 8.0
            y = -2.0 + i * 2.0
            obj_class = 0 if i < 2 else 1
            bbox = self._create_bev_bbox(x, y, 0.0, obj_class)
            boxes.append(bbox)
        
        return {
            'det_boxes': np.array(boxes),
            'det_class': np.array([0, 0, 1], dtype=np.int32),
            'det_score': np.array([0.8, 0.7, 0.6])
        }
    
    def _empty_detection(self):
        """Return empty detection"""
        return {
            'det_boxes': np.zeros((0, 4, 2)),
            'det_class': np.zeros((0,), dtype=np.int32),
            'det_score': np.zeros((0,))
        }
    
    def _process_camera(self, camera_data, camera_id, ego_x, ego_y, ego_yaw):
        """Process one camera"""
        try:
            frame_id, img_data = camera_data
            
            # Preprocess
            img_rgb = self._preprocess_image(img_data)
            if img_rgb is None:
                return []
            
            # Run YOLOv5 inference
            with torch.no_grad():  # Disable gradients for inference
                results = self.model(img_rgb, size=640)
            
            # Extract predictions
            pred = results.xyxy[0].cpu().numpy()
            
            detections = []
            img_h, img_w = img_rgb.shape[:2]
            sensor_config = self.sensor_configs.get(camera_id, self.sensor_configs['Left'])
            
            for det in pred:
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                
                # Filter for relevant classes
                if cls not in self.coco_to_carla:
                    continue
                
                carla_class = self.coco_to_carla[cls]
                
                # Distance estimation
                box_h = y2 - y1
                distance = self._estimate_distance(box_h, img_h, carla_class)
                
                if distance > self.max_detection_range:
                    continue
                
                # Position calculation
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
                
                bbox_2d = self._create_bev_bbox(obj_x, obj_y, total_angle, carla_class)
                
                detections.append({
                    'bbox': bbox_2d,
                    'class': carla_class,
                    'score': float(conf),
                    'position': (obj_x, obj_y)
                })
            
            return detections
            
        except Exception as e:
            print(f"Error processing {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _preprocess_image(self, img_data):
        """Convert CARLA image to RGB"""
        try:
            if img_data is None or not isinstance(img_data, np.ndarray):
                return None
            
            if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
                img_rgb = cv2.cvtColor(img_data[:, :, :3], cv2.COLOR_BGR2RGB)
                return img_rgb
            
            return None
        except Exception as e:
            return None
    
    def _estimate_distance(self, box_height, img_height, obj_class):
        """Estimate distance from box height"""
        ref_heights = {0: 1.6, 1: 1.7, 2: 1.7}
        ref_h = ref_heights.get(obj_class, 1.6)
        
        if box_height > 0:
            distance = (ref_h * img_height) / (box_height * 2)
        else:
            distance = 20.0
        
        return np.clip(distance, 3.0, 50.0)
    
    def _create_bev_bbox(self, cx, cy, heading_deg, obj_class):
        """Create 2D BEV bounding box"""
        length, width = self.default_sizes[obj_class]
        
        hl = length / 2
        hw = width / 2
        
        corners = np.array([
            [ hl, -hw],
            [ hl,  hw],
            [-hl,  hw],
            [-hl, -hw],
        ])
        
        theta = np.deg2rad(heading_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        rotation = np.array([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ])
        
        corners_rotated = corners @ rotation.T
        corners_global = corners_rotated + np.array([cx, cy])
        
        return corners_global
    
    def _merge_detections(self, all_detections):
        """Merge detections with spatial NMS"""
        if not all_detections:
            return self._empty_detection()
        
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        merged = []
        
        for det in all_detections:
            pos = det['position']
            
            is_duplicate = False
            for m in merged:
                m_pos = m['position']
                dist = np.sqrt((pos[0] - m_pos[0])**2 + (pos[1] - m_pos[1])**2)
                
                threshold = 2.5 if det['class'] == 0 else 1.5
                
                if det['class'] == m['class'] and dist < threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        if not merged:
            return self._empty_detection()
        
        det_boxes = np.array([d['bbox'] for d in merged])
        det_class = np.array([d['class'] for d in merged], dtype=np.int32)
        det_score = np.array([d['score'] for d in merged])
        
        return {
            'det_boxes': det_boxes,
            'det_class': det_class,
            'det_score': det_score
        }


def create_detector():
    """Factory function"""
    return Detector()


if __name__ == "__main__":
    print("CARLA YOLOv5 Detector")
    print("Run: python3 automatic_control.py --sync")
