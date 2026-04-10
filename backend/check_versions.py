"""
Diagnostic Script - Check Which Module Versions Are Loaded
Run this to verify the fixed files are being used
"""

import sys
import os

print("\n" + "="*80)
print("DIAGNOSTIC: CHECKING MODULE VERSIONS")
print("="*80)

# Check which files exist
print("\n CHECKING FILE LOCATIONS:")
files_to_check = [
    'eye_gaze_tracker.py',
    'phone_detector.py',
    'speech_analyzer.py',
    'face_detector.py',
    'lip_movement_detector.py'
]

for filename in files_to_check:
    if os.path.exists(filename):
        # Read first 20 lines to check version
        with open(filename, 'r') as f:
            lines = f.readlines()[:20]
            content = ''.join(lines)
            
            # Check for version markers
            if 'PRODUCTION OPTIMIZED' in content or 'PRODUCTION READY' in content:
                print(f" {filename}: FIXED VERSION")
            elif 'DIAGNOSTIC' in content:
                print(f"  {filename}: DIAGNOSTIC VERSION (old)")
            elif 'ADJUSTED' in content or 'FINAL CALIBRATED' in content:
                print(f"  {filename}: OLD VERSION")
            else:
                print(f" {filename}: UNKNOWN VERSION")
    else:
        print(f" {filename}: NOT FOUND")

# Try importing and checking thresholds
print("\n CHECKING LOADED MODULE SETTINGS:")

try:
    from eye_gaze_tracker import EyeGazeTracker
    tracker = EyeGazeTracker()
    threshold = tracker.look_away_threshold
    if threshold == 0.15:
        print(f" Eye Gaze: threshold={threshold} (OPTIMAL)")
    elif threshold == 0.4:
        print(f" Eye Gaze: threshold={threshold} (TOO HIGH - OLD VERSION!)")
    else:
        print(f"  Eye Gaze: threshold={threshold} (UNEXPECTED)")
except Exception as e:
    print(f" Eye Gaze import error: {e}")

try:
    from phone_detector import PhoneDetector
    detector = PhoneDetector()
    conf = detector.confidence_threshold
    min_frames = detector.min_detections
    if conf == 0.4 and min_frames == 5:
        print(f" Phone: confidence={conf}, min_frames={min_frames} (OPTIMAL)")
    elif conf == 0.5 and min_frames == 3:
        print(f" Phone: confidence={conf}, min_frames={min_frames} (DIAGNOSTIC VERSION!)")
    elif conf == 0.75 and min_frames == 10:
        print(f" Phone: confidence={conf}, min_frames={min_frames} (STRICT VERSION - OLD!)")
    else:
        print(f"  Phone: confidence={conf}, min_frames={min_frames} (UNEXPECTED)")
except Exception as e:
    print(f" Phone import error: {e}")

try:
    from speech_analyzer import SpeechAnalyzer
    analyzer = SpeechAnalyzer()
    speech_penalty = analyzer.SPEECH_PENALTY_PER_BURST
    if speech_penalty == 5.0:
        print(f" Speech: penalty={speech_penalty} (OPTIMAL)")
    elif speech_penalty == 8.0:
        print(f" Speech: penalty={speech_penalty} (TOO SENSITIVE - OLD VERSION!)")
    else:
        print(f"  Speech: penalty={speech_penalty} (UNEXPECTED)")
except Exception as e:
    print(f" Speech import error: {e}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("""
If you see ANY "OLD VERSION" or "DIAGNOSTIC VERSION" messages above:
1. Your files were NOT replaced correctly
2. Python may be caching the old modules
3. Follow the fix below
""")

print("\n" + "="*80)
print("FIX INSTRUCTIONS:")
print("="*80)
print("""
1. STOP the Flask server (Ctrl+C)
2. Delete Python cache:
   rm -rf __pycache__
   find . -name "*.pyc" -delete
3. Replace the files again
4. Restart: python app.py
5. Run this diagnostic again
""")
print("="*80 + "\n")
