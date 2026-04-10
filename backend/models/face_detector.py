"""
Face Detection Module
Detects multiple faces, unauthorized persons, and face absence
Uses MediaPipe and OpenCV for robust face detection
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# MediaPipe availability handling
MP_AVAILABLE = False
mp_face_detection_module = None
try:
    from mediapipe import solutions as mp_solutions
    mp_face_detection_module = mp_solutions.face_detection
    MP_AVAILABLE = True
except Exception:
    try:
        from mediapipe.python.solutions import face_detection as mp_face_detection_module
        MP_AVAILABLE = True
    except Exception:
        MP_AVAILABLE = False


class FaceDetector:
    """
    Advanced face detection system for monitoring exam integrity
    Detects multiple faces, face absence, and suspicious activities
    """
    
    def __init__(self, 
                 max_history=300,
                 min_detection_confidence=0.15,
                 allowed_faces=1):
        """
        Initialize the face detector
        
        Args:
            max_history: Maximum number of frames to keep in history
            min_detection_confidence: Minimum confidence for face detection
            allowed_faces: Number of allowed faces (default: 1)
        """
        # Initialize MediaPipe Face Detection (fallback to Haar if unavailable)
        self.face_detection = None
        if MP_AVAILABLE and mp_face_detection_module is not None:
            self.mp_face_detection = mp_face_detection_module
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=1  # 1 for full range, 0 for short range
            )
        else:
            self.face_detection = None
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # Parameters
        self.allowed_faces = allowed_faces
        self.max_history = max_history
        
        # Tracking history
        self.face_count_history = deque(maxlen=max_history)
        self.violation_history = deque(maxlen=max_history)
        
        # Violation counters
        self.total_frames = 0
        self.no_face_frames = 0
        self.multiple_faces_frames = 0
        self.total_violations = 0
        
        # Face tracking
        self.consecutive_no_face = 0
        self.consecutive_multiple_faces = 0
        self.max_consecutive_no_face = 0
        self.max_consecutive_multiple_faces = 0
        
        # Time tracking
        self.start_time = time.time()
        self.violation_timestamps = []
        
        # Face recognition (for authorized person verification)
        self.authorized_face_encoding = None
        self.face_recognition_enabled = False
    
    def detect_faces(self, frame):
        """
        Detect faces in the given frame
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            list: List of detected face bounding boxes and confidence scores
        """
        faces = []
        if self.face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to absolute coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    confidence = detection.score[0]
                    
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'detection': detection
                    })
        else:
            # OpenCV fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w, h) in detections:
                faces.append({
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'confidence': 0.6,
                    'detection': None
                })
        
        return faces
    
    def process_frame(self, frame):
        """
        Process a single video frame for face detection analysis
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            dict: Analysis results including face count, violations, etc.
        """
        self.total_frames += 1
        
        # Detect faces
        faces = self.detect_faces(frame)
        num_faces = len(faces)
        
        # Initialize analysis results
        analysis = {
            'num_faces': num_faces,
            'faces': faces,
            'violations': [],
            'is_violation': False,
            'violation_type': None,
            'annotated_frame': frame.copy()
        }
        
        # Check for violations
        if num_faces == 0:
            # No face detected
            self.no_face_frames += 1
            self.consecutive_no_face += 1
            self.consecutive_multiple_faces = 0
            self.max_consecutive_no_face = max(
                self.max_consecutive_no_face,
                self.consecutive_no_face
            )
            
            analysis['is_violation'] = True
            analysis['violation_type'] = 'no_face'
            analysis['violations'].append('no_face_detected')
            self.violation_history.append(1)
            self.violation_timestamps.append(time.time() - self.start_time)
            
        elif num_faces > self.allowed_faces:
            # Multiple faces detected
            self.multiple_faces_frames += 1
            self.consecutive_multiple_faces += 1
            self.consecutive_no_face = 0
            self.max_consecutive_multiple_faces = max(
                self.max_consecutive_multiple_faces,
                self.consecutive_multiple_faces
            )
            
            analysis['is_violation'] = True
            analysis['violation_type'] = 'multiple_faces'
            analysis['violations'].append(f'multiple_faces_detected_{num_faces}')
            self.violation_history.append(1)
            self.violation_timestamps.append(time.time() - self.start_time)
            
        else:
            # Valid state (1 face)
            self.consecutive_no_face = 0
            self.consecutive_multiple_faces = 0
            self.violation_history.append(0)
        
        # Update history
        self.face_count_history.append(num_faces)
        
        # Draw annotations
        annotated_frame = self.draw_annotations(frame, faces, analysis['violation_type'])
        analysis['annotated_frame'] = annotated_frame
        
        return analysis
    
    def draw_annotations(self, frame, faces, violation_type):
        """
        Draw face bounding boxes and violation warnings on frame
        
        Args:
            frame: Input frame
            faces: List of detected faces with bounding boxes
            violation_type: Type of violation (None, 'no_face', 'multiple_faces')
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w, _ = frame.shape
        
        # Determine color based on violation
        if violation_type == 'no_face':
            status_color = (0, 0, 255)  # Red
            status_text = "WARNING: No Face Detected!"
        elif violation_type == 'multiple_faces':
            status_color = (0, 0, 255)  # Red
            status_text = f"WARNING: {len(faces)} Faces Detected!"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "Status: OK"
        
        # Draw status banner
        cv2.rectangle(annotated, (0, 0), (w, 50), status_color, -1)
        cv2.putText(annotated, status_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw face bounding boxes
        for face in faces:
            x, y, width, height = face['bbox']
            confidence = face['confidence']
            
            # Determine box color
            box_color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + width, y + height), box_color, 2)
            
            # Draw confidence score
            label = f"{confidence:.2f}"
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw face count
        cv2.putText(annotated, f"Faces: {len(faces)}", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_statistics(self):
        """
        Get comprehensive statistics about face detection
        
        Returns:
            dict: Statistical summary
        """
        if self.total_frames == 0:
            return {}
        
        # Calculate percentages
        no_face_percentage = (self.no_face_frames / self.total_frames) * 100
        multiple_faces_percentage = (self.multiple_faces_frames / self.total_frames) * 100
        valid_frames_percentage = 100 - no_face_percentage - multiple_faces_percentage
        
        # Calculate violation rate
        total_violations = self.no_face_frames + self.multiple_faces_frames
        violation_percentage = (total_violations / self.total_frames) * 100
        
        # Calculate average faces per frame
        avg_faces = np.mean(self.face_count_history) if self.face_count_history else 0
        
        # Time statistics
        elapsed_time = time.time() - self.start_time
        
        return {
            'total_frames': self.total_frames,
            'no_face_frames': self.no_face_frames,
            'no_face_percentage': round(no_face_percentage, 2),
            'multiple_faces_frames': self.multiple_faces_frames,
            'multiple_faces_percentage': round(multiple_faces_percentage, 2),
            'valid_frames_percentage': round(valid_frames_percentage, 2),
            'total_violations': total_violations,
            'violation_percentage': round(violation_percentage, 2),
            'max_consecutive_no_face': self.max_consecutive_no_face,
            'max_consecutive_multiple_faces': self.max_consecutive_multiple_faces,
            'avg_faces_detected': round(avg_faces, 2),
            'session_duration_seconds': round(elapsed_time, 2),
            'violations_per_minute': round((total_violations / (elapsed_time / 60)) if elapsed_time > 0 else 0, 2)
        }
    
    def calculate_face_integrity_score(self):
        """
        Calculate face detection integrity score (0-100)
        Higher score means better compliance
        
        Returns:
            float: Integrity score
        """
        stats = self.get_statistics()
        
        if stats.get('total_frames', 0) == 0:
            return 100.0
        
        # Start with perfect score
        score = 100.0
        
        # Penalize for no face detection (severe penalty)
        no_face_penalty = stats.get('no_face_percentage', 0) * 0.9
        score -= no_face_penalty
        
        # Penalize for multiple faces (very severe penalty)
        multiple_faces_penalty = stats.get('multiple_faces_percentage', 0) * 1.0
        score -= multiple_faces_penalty
        
        # Penalize for long consecutive violations
        max_no_face = stats.get('max_consecutive_no_face', 0)
        if max_no_face > 60:  # More than 2 seconds at 30fps
            score -= min((max_no_face - 60) * 0.15, 20)
        
        max_multiple = stats.get('max_consecutive_multiple_faces', 0)
        if max_multiple > 30:  # More than 1 second at 30fps
            score -= min((max_multiple - 30) * 0.2, 25)
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return round(score, 2)
    
    def get_violation_timeline(self):
        """
        Get timeline of violations for reporting
        
        Returns:
            list: List of violation timestamps
        """
        return self.violation_timestamps
    
    def reset(self):
        """Reset all tracking data"""
        self.face_count_history.clear()
        self.violation_history.clear()
        self.total_frames = 0
        self.no_face_frames = 0
        self.multiple_faces_frames = 0
        self.total_violations = 0
        self.consecutive_no_face = 0
        self.consecutive_multiple_faces = 0
        self.max_consecutive_no_face = 0
        self.max_consecutive_multiple_faces = 0
        self.start_time = time.time()
        self.violation_timestamps.clear()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


class FaceVerifier:
    """
    Face verification system to ensure the same authorized person
    throughout the exam
    """
    
    def __init__(self):
        """Initialize face verifier"""
        try:
            import face_recognition
            self.face_recognition = face_recognition
            self.enabled = True
        except ImportError:
            print("Warning: face_recognition library not available")
            self.enabled = False
        
        self.authorized_encoding = None
        self.verification_threshold = 0.6
    
    def set_authorized_face(self, frame):
        """
        Set the authorized face from initial frame
        
        Args:
            frame: Frame containing the authorized person's face
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = self.face_recognition.face_encodings(rgb_frame)
        
        if len(face_encodings) == 1:
            self.authorized_encoding = face_encodings[0]
            return True
        return False
    
    def verify_face(self, frame):
        """
        Verify if the face in frame matches authorized person
        
        Args:
            frame: Current frame to verify
            
        Returns:
            dict: Verification results
        """
        if not self.enabled or self.authorized_encoding is None:
            return {'verified': True, 'confidence': 1.0}
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = self.face_recognition.face_encodings(rgb_frame)
        
        if len(face_encodings) == 0:
            return {'verified': False, 'confidence': 0.0, 'reason': 'no_face'}
        
        if len(face_encodings) > 1:
            return {'verified': False, 'confidence': 0.0, 'reason': 'multiple_faces'}
        
        # Compare with authorized face
        face_distances = self.face_recognition.face_distance(
            [self.authorized_encoding], 
            face_encodings[0]
        )
        
        distance = face_distances[0]
        verified = distance < self.verification_threshold
        confidence = 1 - distance
        
        return {
            'verified': verified,
            'confidence': round(confidence, 3),
            'distance': round(distance, 3)
        }


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FaceDetector(allowed_faces=1)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Face Detector Started. Press 'q' to quit, 's' for statistics.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        analysis = detector.process_frame(frame)
        
        # Display annotated frame
        cv2.imshow('Face Detection', analysis['annotated_frame'])
        
        # Display real-time info
        print(f"\rFaces: {analysis['num_faces']}, "
              f"Violations: {len(analysis['violations'])}, "
              f"Score: {detector.calculate_face_integrity_score():.1f}", end='')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n\n=== Statistics ===")
            stats = detector.get_statistics()
            for key, value in stats.items():
                print(f"{key}: {value}")
            print(f"Integrity Score: {detector.calculate_face_integrity_score()}")
            print("================\n")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n\n=== Final Report ===")
    final_stats = detector.get_statistics()
    for key, value in final_stats.items():
        print(f"{key}: {value}")
    print(f"Final Integrity Score: {detector.calculate_face_integrity_score()}")
