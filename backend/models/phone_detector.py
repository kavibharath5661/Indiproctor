"""
Phone Detector - PRODUCTION READY
Optimized for real-time detection with balanced sensitivity
"""

import cv2
import numpy as np
from collections import deque
import time

try:
    from ultralytics import YOLO
    import os
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class PhoneDetector:
    """Production phone detector with balanced settings"""
    
    def __init__(self):
        # Unauthorized gadgets
        self.unauthorized_objects = {
            'cell phone': {'name': 'Mobile Phone', 'penalty': 10.0, 'emoji': ''},
            'laptop': {'name': 'Laptop', 'penalty': 10.0, 'emoji': ''},
            'remote': {'name': 'Device/Remote', 'penalty': 10.0, 'emoji': ''},
            'tv': {'name': 'Screen/Monitor', 'penalty': 10.0, 'emoji': ''},
            'book': {'name': 'Book/Notes', 'penalty': 10.0, 'emoji': ''},
            'apple': {'name': 'Apple Device', 'penalty': 10.0, 'emoji': ''}, # YOLO sometimes confuses back of iPhone
        }
        
        # PRODUCTION SETTINGS (balanced)
        self.confidence_threshold = 0.55  # Increased to prevent false positives
        self.detection_cooldown = 3.0    # 3 seconds between violations
        self.min_detections = 3          # Require multiple consecutive frames
        
        # Tracking
        self.last_detections = {}
        self.detection_buffer = deque(maxlen=10)
        self.consecutive_detections = 0
        
        # Statistics
        self.total_detections = 0
        self.confirmed_detections = 0
        self.frame_count = 0
        
        try:
            if YOLO_AVAILABLE:
                model_path = 'yolov8s.pt'  # Upgraded to small model for better accuracy
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f" Phone Detector: confidence={self.confidence_threshold}, min_frames={self.min_detections}")
            else:
                self.model_loaded = False
                print("  YOLO unavailable - phone detection disabled")
        except Exception as e:
            print(f" Phone detector init error: {e}")
            self.model_loaded = False
    
    def detect_gadgets(self, frame):
        """Detect unauthorized gadgets"""
        self.frame_count += 1
        
        if not self.model_loaded:
            return {
                'detected': False,
                'objects': [],
                'penalties': {},
                'total_penalty': 0.0,
                'confidence': 0.0
            }
        
        try:
            current_time = time.time()
            
            # Run YOLO
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            frame_has_phone = False
            detected_objects = []
            
            # Check detections
            persons_in_frame = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name == 'person':
                        persons_in_frame += 1
                    
                    if class_name in self.unauthorized_objects:
                        frame_has_phone = True
                        obj_info = self.unauthorized_objects[class_name]
                        detected_objects.append({
                            'name': obj_info['name'],
                            'confidence': confidence
                        })
            
            # Multi-frame confirmation
            self.detection_buffer.append(frame_has_phone)
            
            if frame_has_phone:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = 0
            
            # Confirm after min_detections frames
            if self.consecutive_detections >= self.min_detections:
                # Check cooldown
                last_time = self.last_detections.get('phone', 0)
                if current_time - last_time < self.detection_cooldown:
                    return {
                        'detected': False,
                        'objects': [],
                        'penalties': {},
                        'total_penalty': 0.0,
                        'confidence': 0.0
                    }
                
                # CONFIRMED DETECTION
                self.last_detections['phone'] = current_time
                self.confirmed_detections += 1
                
                penalties = {}
                total_penalty = 0.0
                object_names = []
                
                for obj in detected_objects:
                    penalties[obj['name']] = 10.0
                    total_penalty += 10.0
                    object_names.append(obj['name'])
                
                return {
                    'detected': True,
                    'objects': object_names,
                    'penalties': penalties,
                    'total_penalty': total_penalty,
                    'confidence': detected_objects[0]['confidence'] if detected_objects else 0.0,
                    'person_count': persons_in_frame
                }
            else:
                return {
                    'detected': False,
                    'objects': [],
                    'penalties': {},
                    'total_penalty': 0.0,
                    'confidence': 0.0,
                    'person_count': persons_in_frame
                }
        
        except Exception as e:
            return {
                'detected': False,
                'objects': [],
                'penalties': {},
                'total_penalty': 0.0,
                'confidence': 0.0,
                'person_count': 0,
                'error': str(e)
            }
    
    def get_detection_stats(self):
        """Get statistics"""
        return {
            'total_detections': self.total_detections,
            'confirmed_detections': self.confirmed_detections,
            'consecutive_frames': self.consecutive_detections,
            'frames_processed': self.frame_count
        }
    
    def reset(self):
        """Reset detector"""
        self.detection_buffer.clear()
        self.consecutive_detections = 0
        self.last_detections = {}
        self.frame_count = 0
