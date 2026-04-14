"""
Eye Gaze Tracker - WITH WARM-UP PERIOD
Prevents score from dropping in the first 30 seconds of exam
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time


class EyeGazeTracker:
    """Eye gaze tracking with warm-up period to prevent early penalties"""
    
    def __init__(self, 
                 max_history=300,
                 look_away_threshold=0.18,
                 blink_threshold=0.21,
                 warmup_frames=100):  # Reduced to ~3.3 seconds
        """Initialize with warm-up period"""
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.look_away_threshold = look_away_threshold
        self.blink_threshold = blink_threshold
        self.max_history = max_history
        
        #  NEW: Warm-up period
        self.warmup_frames = warmup_frames
        self.is_warmed_up = False
        
        # History tracking
        self.gaze_history = deque(maxlen=max_history)
        self.blink_history = deque(maxlen=max_history)
        self.looking_away_history = deque(maxlen=max_history)
        self.displacement_history = deque(maxlen=5)  # For smoothing sensor data
        
        # Counters
        self.total_frames = 0
        self.looking_away_count = 0
        self.blink_count = 0
        self.no_face_count = 0
        
        # Time tracking
        self.start_time = time.time()
        self.last_blink_time = time.time()
        
        print(f" Eye Tracker with {warmup_frames} frame warm-up (threshold={look_away_threshold})")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate EAR for blink detection"""
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C)
    
    def get_iris_displacement(self, iris_landmarks, eye_landmarks):
        """Calculate iris displacement ratio"""
        iris_center = np.mean(iris_landmarks, axis=0)
        eye_center = np.mean(eye_landmarks, axis=0)
        
        eye_width = np.max(eye_landmarks[:, 0]) - np.min(eye_landmarks[:, 0])
        eye_height = np.max(eye_landmarks[:, 1]) - np.min(eye_landmarks[:, 1])
        
        if eye_width < 1e-6 or eye_height < 1e-6:
            return 0.0, 0.0, 0.0
        
        dx = (iris_center[0] - eye_center[0]) / eye_width
        dy = (iris_center[1] - eye_center[1]) / eye_height
        displacement = np.sqrt(dx**2 + dy**2)
        
        return displacement, dx, dy
    
    def process_frame(self, frame):
        """Process frame with warm-up period"""
        self.total_frames += 1
        
        #  Check if warm-up is complete
        if not self.is_warmed_up and self.total_frames >= self.warmup_frames:
            self.is_warmed_up = True
            print(f" Eye gaze warm-up complete! Now tracking violations.")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        analysis = {
            'face_detected': False,
            'gaze_direction': None,
            'is_looking_away': False,
            'is_blinking': False,
            'eye_aspect_ratio': 0,
            'displacement': 0.0,
            'violations': [],
            'annotated_frame': frame.copy(),
            'is_warmup': not self.is_warmed_up  #  NEW
        }
        
        if not results.multi_face_landmarks:
            self.no_face_count += 1
            analysis['violations'].append('no_face_detected')
            #  Only count during warm-up if already warmed up
            if self.is_warmed_up:
                self.looking_away_history.append(1)
            else:
                self.looking_away_history.append(0)  # Don't penalize during warm-up
            return analysis
        
        analysis['face_detected'] = True
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        # Extract landmarks
        left_eye = np.array([(face_landmarks.landmark[i].x * w, 
                             face_landmarks.landmark[i].y * h) 
                            for i in self.LEFT_EYE])
        right_eye = np.array([(face_landmarks.landmark[i].x * w, 
                              face_landmarks.landmark[i].y * h) 
                             for i in self.RIGHT_EYE])
        left_iris = np.array([(face_landmarks.landmark[i].x * w, 
                              face_landmarks.landmark[i].y * h) 
                             for i in self.LEFT_IRIS])
        right_iris = np.array([(face_landmarks.landmark[i].x * w, 
                               face_landmarks.landmark[i].y * h) 
                              for i in self.RIGHT_IRIS])
        
        # Calculate EAR
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        analysis['eye_aspect_ratio'] = avg_ear
        
        # Blink detection
        if avg_ear < self.blink_threshold:
            analysis['is_blinking'] = True
            current_time = time.time()
            if current_time - self.last_blink_time > 0.3:
                self.blink_count += 1
                self.last_blink_time = current_time
        
        # Iris displacement (gaze detection)
        left_disp, left_dx, left_dy = self.get_iris_displacement(left_iris, left_eye)
        right_disp, right_dx, right_dy = self.get_iris_displacement(right_iris, right_eye)
        
        raw_displacement = (left_disp + right_disp) / 2.0
        
        # Smooth displacement to avoid jitter
        self.displacement_history.append(raw_displacement)
        avg_displacement = sum(self.displacement_history) / len(self.displacement_history)
        
        analysis['displacement'] = avg_displacement
        
        #  Only penalize if warmed up
        if avg_displacement >= self.look_away_threshold:
            analysis['is_looking_away'] = True
            analysis['gaze_direction'] = 'away'
            
            if self.is_warmed_up:
                self.looking_away_count += 1
                analysis['violations'].append('looking_away')
                self.looking_away_history.append(1)
            else:
                # During warm-up, detect but don't penalize
                self.looking_away_history.append(0)
        else:
            analysis['gaze_direction'] = 'center'
            self.looking_away_history.append(0)
        
        # Update history
        self.gaze_history.append({
            'timestamp': time.time() - self.start_time,
            'displacement': avg_displacement,
            'direction': analysis['gaze_direction'],
            'ear': avg_ear
        })
        self.blink_history.append(avg_ear)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(
            frame, left_eye, right_eye, left_iris, right_iris, 
            analysis['gaze_direction']
        )
        analysis['annotated_frame'] = annotated_frame
        
        return analysis
    
    def draw_annotations(self, frame, left_eye, right_eye, 
                        left_iris, right_iris, direction):
        """Draw gaze annotations"""
        annotated = frame.copy()
        
        # Draw eyes
        cv2.polylines(annotated, [left_eye.astype(np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(annotated, [right_eye.astype(np.int32)], True, (0, 255, 0), 1)
        
        # Draw iris
        left_center = left_iris.mean(axis=0).astype(np.int32)
        right_center = right_iris.mean(axis=0).astype(np.int32)
        cv2.circle(annotated, tuple(left_center), 2, (255, 0, 0), -1)
        cv2.circle(annotated, tuple(right_center), 2, (255, 0, 0), -1)
        
        # Status text
        color = (0, 255, 0) if direction == "center" else (0, 0, 255)
        status = "WARM-UP" if not self.is_warmed_up else direction
        cv2.putText(annotated, f"Gaze: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
    
    def get_statistics(self):
        """Get detailed statistics"""
        if self.total_frames == 0:
            return {}
        
        #  Calculate based on frames AFTER warm-up
        trackable_frames = max(1, self.total_frames - self.warmup_frames)
        
        looking_away_pct = (self.looking_away_count / trackable_frames) * 100
        no_face_pct = (self.no_face_count / trackable_frames) * 100
        
        elapsed = time.time() - self.start_time
        blink_rate = (self.blink_count / (elapsed / 60)) if elapsed > 0 else 0
        
        # Calculate max consecutive looking away
        consecutive = 0
        max_consecutive = 0
        for is_away in self.looking_away_history:
            if is_away:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return {
            'total_frames': self.total_frames,
            'trackable_frames': trackable_frames,
            'is_warmup': not self.is_warmed_up,
            'looking_away_count': self.looking_away_count,
            'looking_away_percentage': round(looking_away_pct, 2),
            'no_face_count': self.no_face_count,
            'no_face_percentage': round(no_face_pct, 2),
            'blink_count': self.blink_count,
            'blink_rate_per_minute': round(blink_rate, 2),
            'max_consecutive_looking_away_frames': max_consecutive,
            'avg_eye_aspect_ratio': round(np.mean(self.blink_history) if self.blink_history else 0, 3),
            'session_duration_seconds': round(elapsed, 2)
        }
    
    def calculate_attention_score(self):
        """
        Calculate attention score with warm-up protection
        Score stays at 100 during warm-up period
        """
        stats = self.get_statistics()
        
        #  During warm-up, always return 100
        if stats.get('is_warmup', False):
            return 100.0
        
        if stats.get('trackable_frames', 0) == 0:
            return 100.0
        
        score = 100.0
        
        # Penalty for looking away
        looking_away_penalty = stats.get('looking_away_percentage', 0) * 1.5
        score -= looking_away_penalty
        
        # Heavy penalty for no face
        no_face_penalty = stats.get('no_face_percentage', 0) * 1.5
        score -= no_face_penalty
        
        # Penalty for sustained looking away
        max_consecutive = stats.get('max_consecutive_looking_away_frames', 0)
        if max_consecutive > 30:
            score -= min((max_consecutive - 30) * 0.2, 30)
        
        return round(max(0, min(100, score)), 2)
    
    def reset(self):
        """Reset all tracking"""
        self.gaze_history.clear()
        self.blink_history.clear()
        self.looking_away_history.clear()
        self.total_frames = 0
        self.looking_away_count = 0
        self.blink_count = 0
        self.no_face_count = 0
        self.is_warmed_up = False
        self.start_time = time.time()
        self.last_blink_time = time.time()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
