"""
MediaPipe Fix and Verification Script
Fixes the MediaPipe import error and verifies all modules
"""

import subprocess
import sys

print("\n" + "="*80)
print("FIXING MEDIAPIPE AND VERIFYING ALL IMPORTS")
print("="*80)

# 1. Uninstall and reinstall MediaPipe
print("\n1. Fixing MediaPipe installation...")
try:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "mediapipe"], 
                   check=False, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "pip", "install", "mediapipe==0.10.9"], 
                          check=True, capture_output=True, text=True)
    print(" MediaPipe 0.10.9 installed")
except Exception as e:
    print(f"  MediaPipe installation issue: {e}")

# 2. Verify MediaPipe
print("\n2. Verifying MediaPipe...")
try:
    import mediapipe as mp
    print(f" MediaPipe version: {mp.__version__}")
    
    # Test solutions attribute
    if hasattr(mp, 'solutions'):
        print(" MediaPipe.solutions available")
        
        # Test FaceMesh
        face_mesh = mp.solutions.face_mesh
        print(" FaceMesh available")
    else:
        print(" MediaPipe.solutions NOT available")
        print("   This means MediaPipe is not installed correctly")
except ImportError as e:
    print(f" MediaPipe import error: {e}")
except Exception as e:
    print(f"  MediaPipe verification error: {e}")

# 3. Verify other dependencies
print("\n3. Verifying other dependencies...")
try:
    import cv2
    print(f" OpenCV version: {cv2.__version__}")
except ImportError:
    print(" OpenCV not installed")
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python"], check=False)

try:
    import numpy as np
    print(f" NumPy version: {np.__version__}")
except ImportError:
    print(" NumPy not installed")

try:
    from ultralytics import YOLO
    print(" Ultralytics (YOLO) installed")
except ImportError:
    print(" Ultralytics not installed")
    subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=False)

# 4. Test module imports
print("\n4. Testing module imports...")
try:
    from eye_gaze_tracker import EyeGazeTracker
    tracker = EyeGazeTracker()
    print(f" Eye Gaze Tracker: threshold={tracker.look_away_threshold}")
    del tracker
except Exception as e:
    print(f" Eye Gaze Tracker error: {e}")

try:
    from phone_detector import PhoneDetector
    detector = PhoneDetector()
    print(f" Phone Detector: confidence={detector.confidence_threshold}")
    del detector
except Exception as e:
    print(f" Phone Detector error: {e}")

try:
    from speech_analyzer import SpeechAnalyzer
    analyzer = SpeechAnalyzer()
    print(f" Speech Analyzer: penalty={analyzer.SPEECH_PENALTY_PER_BURST}")
    del analyzer
except Exception as e:
    print(f" Speech Analyzer error: {e}")

try:
    from face_detector import FaceDetector
    face_det = FaceDetector()
    print(" Face Detector imported")
    del face_det
except Exception as e:
    print(f" Face Detector error: {e}")

try:
    from lip_movement_detector import LipMovementDetector
    lip_det = LipMovementDetector()
    print(" Lip Movement Detector imported")
    del lip_det
except Exception as e:
    print(f" Lip Movement Detector error: {e}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nIf all modules show , you can now run: python app.py")
print("="*80 + "\n")
