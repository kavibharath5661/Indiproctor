"""
FINAL CALIBRATED Lip Movement Detector
Optimized for SILENT READING scenarios
Only detects ACTUAL AUDIBLE SPEECH
"""

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class LipMovementDetector:
    """
    FINAL CALIBRATED version for silent reading
    - Ignores silent reading lip movements
    - Only detects actual audible speech
    - Uses audio correlation for confirmation
    """
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe required")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Higher confidence
            min_tracking_confidence=0.7
        )
        
        # Lip landmarks
        self.upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        #  FINAL THRESHOLDS - Optimized for silent reading
        self.min_distance = 0.015      # HIGH - ignores silent reading
        self.moderate_threshold = 0.030 # Only clear speech
        self.severe_threshold = 0.050   # Loud speech only
        
        # Smoothing (longer window for stability)
        self.lip_distance_history = deque(maxlen=15)  # Increased
        self.detection_buffer = deque(maxlen=10)       # Increased
        
        # Baseline with moving average
        self.baseline_distance = None
        self.baseline_samples = []
        self.baseline_frames = 60  # 2 seconds at 30fps
        self.baseline_update_counter = 0
        
        # Audio correlation (NEW!)
        self.recent_audio_levels = deque(maxlen=10)
        self.audio_speech_detected = False
        
        # Statistics
        self.total_detections = 0
        self.confirmed_detections = 0
        self.frame_count = 0
        self.false_positive_blocks = 0
        
        print(" FINAL CALIBRATED lip detector")
        print("   - Optimized for SILENT READING")
        print("   - High thresholds: min=0.018, mod=0.035, sev=0.055")
        print("   - Audio correlation enabled")
        print("   - 60-frame baseline with updates")
    
    def update_audio_status(self, audio_level, is_speech):
        """Update audio status for correlation"""
        self.recent_audio_levels.append(audio_level)
        self.audio_speech_detected = is_speech
    
    def detect_lip_movement(self, frame):
        """
        FINAL detection with audio correlation
        Only confirms lip movement if audio also detects speech
        """
        if frame is None or not MEDIAPIPE_AVAILABLE:
            return {'detected': False}
        
        try:
            self.frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.lip_distance_history.clear()
                self.detection_buffer.clear()
                return {'detected': False, 'reason': 'No face'}
            
            face_landmarks = results.multi_face_landmarks[0]
            
            # Calculate lip distance
            upper_y = np.mean([face_landmarks.landmark[i].y for i in self.upper_lip])
            lower_y = np.mean([face_landmarks.landmark[i].y for i in self.lower_lip])
            raw_distance = abs(lower_y - upper_y)
            
            #  BASELINE CALIBRATION (60 frames = 2 seconds)
            if len(self.baseline_samples) < self.baseline_frames:
                self.baseline_samples.append(raw_distance)
                if len(self.baseline_samples) == self.baseline_frames:
                    # Use median for baseline (more robust)
                    self.baseline_distance = np.median(self.baseline_samples)
                    print(f" Baseline set: {self.baseline_distance:.4f} (median of 60 frames)")
                return {
                    'detected': False, 
                    'reason': 'Calibrating baseline', 
                    'progress': f"{len(self.baseline_samples)}/{self.baseline_frames}"
                }
            
            #  ADAPTIVE BASELINE UPDATE (every 300 frames = 10 seconds)
            self.baseline_update_counter += 1
            if self.baseline_update_counter >= 300:
                # Update baseline with recent stable measurements
                if len(self.lip_distance_history) >= 10:
                    recent_median = np.median(list(self.lip_distance_history))
                    # Slowly adjust baseline (0.9 old + 0.1 new)
                    self.baseline_distance = 0.9 * self.baseline_distance + 0.1 * recent_median
                self.baseline_update_counter = 0
            
            self.lip_distance_history.append(raw_distance)
            
            #  AGGRESSIVE SMOOTHING (median of last 15 frames)
            if len(self.lip_distance_history) >= 10:
                smoothed_distance = np.median(list(self.lip_distance_history))
            else:
                smoothed_distance = raw_distance
            
            # Deviation from baseline
            baseline_deviation = smoothed_distance - self.baseline_distance
            
            #  HIGH THRESHOLD DETECTION
            if baseline_deviation > self.min_distance:
                
                #  AUDIO CORRELATION CHECK (NEW!)
                # Only proceed if audio also indicates potential speech
                avg_audio = np.mean(list(self.recent_audio_levels)) if self.recent_audio_levels else 0
                
                # Require BOTH lip movement AND audio activity
                if avg_audio < 5 and not self.audio_speech_detected:
                    # No audio activity - likely silent reading
                    self.detection_buffer.append(False)
                    self.false_positive_blocks += 1
                    if self.false_positive_blocks % 30 == 0:
                        print(f" Blocked {self.false_positive_blocks} false positives (silent reading)")
                    return {
                        'detected': False,
                        'reason': 'No audio correlation',
                        'deviation': float(baseline_deviation),
                        'audio_level': float(avg_audio)
                    }
                
                # Classify severity (only if audio confirms)
                if baseline_deviation > self.severe_threshold:
                    warning_level = 1
                    penalty = 2.0
                    message = " SEVERE speech"
                    confidence = 0.95
                elif baseline_deviation > self.moderate_threshold:
                    warning_level = 2
                    penalty = 1.0
                    message = " MODERATE speech"
                    confidence = 0.85
                else:
                    warning_level = 3
                    penalty = 0.5
                    message = " Minor speech"
                    confidence = 0.75
                
                self.detection_buffer.append(True)
                
                #  STRICT CONFIRMATION: 7 of 10 frames
                if len(self.detection_buffer) >= 10:
                    recent_detections = sum(list(self.detection_buffer))
                    confirmed = recent_detections >= 7
                    confidence *= (recent_detections / 10)
                else:
                    confirmed = False
                
                if confirmed:
                    self.total_detections += 1
                    self.confirmed_detections += 1
                    
                    if self.confirmed_detections % 5 == 0:
                        print(f" Speech #{self.confirmed_detections}: dev={baseline_deviation:.4f}, audio={avg_audio:.1f}")
                    
                    return {
                        'detected': True,
                        'distance': float(smoothed_distance),
                        'baseline_distance': float(self.baseline_distance),
                        'deviation': float(baseline_deviation),
                        'audio_level': float(avg_audio),
                        'warning_level': warning_level,
                        'penalty': penalty,
                        'message': message,
                        'confidence': confidence,
                        'total_detections': self.confirmed_detections
                    }
                else:
                    return {
                        'detected': False, 
                        'reason': f'Confirming ({sum(list(self.detection_buffer))}/10)',
                        'deviation': float(baseline_deviation)
                    }
            else:
                self.detection_buffer.append(False)
                return {
                    'detected': False,
                    'baseline_distance': float(self.baseline_distance),
                    'current_distance': float(smoothed_distance),
                    'deviation': float(baseline_deviation)
                }
        
        except Exception as e:
            print(f" Lip error: {e}")
            return {'detected': False, 'error': str(e)}
    
    def reset(self):
        """Reset detector"""
        self.lip_distance_history.clear()
        self.detection_buffer.clear()
        self.baseline_samples = []
        self.baseline_distance = None
        self.baseline_update_counter = 0
        self.recent_audio_levels.clear()
        self.total_detections = 0
        self.confirmed_detections = 0
        self.frame_count = 0
        self.false_positive_blocks = 0
    
    def get_statistics(self):
        """Get statistics"""
        return {
            'total_detections': self.total_detections,
            'confirmed_detections': self.confirmed_detections,
            'false_positives_blocked': self.false_positive_blocks,
            'frame_count': self.frame_count,
            'baseline': float(self.baseline_distance) if self.baseline_distance else 0
        }
    
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except:
            pass
