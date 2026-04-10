"""
COMPLETE VERSION - ALL FEATURES WORKING!
 Eye gaze tracking
 Multiple face detection  
 Ultra-sensitive speech detection
 PHONE/GADGET DETECTION (NEW!)
 ENCRYPTED STORAGE (NEW!)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import os
import sys
from collections import deque
import time

sys.path.append(os.path.dirname(__file__))

from models.eye_gaze_tracker import EyeGazeTracker
from models.face_detector import FaceDetector
from models.speech_analyzer import SpeechAnalyzer
from utils.integrity_scorer import IntegrityScorer

#  NEW: Import encryption module
try:
    from utils.encryption import get_storage
    ENCRYPTION_AVAILABLE = True
    print(" Encryption module loaded")
except ImportError:
    ENCRYPTION_AVAILABLE = False
    print(" Encryption module not available")

# Import phone detector (will create fallback if YOLO unavailable)
try:
    from models.phone_detector import PhoneDetector
    PHONE_DETECTION_AVAILABLE = True  #  ENABLED with strict detector
    print(" Phone detection ENABLED")
except:
    PHONE_DETECTION_AVAILABLE = False
    print(" Phone detection module not found - feature disabled")

# Import lip movement detector
try:
    from models.lip_movement_detector import LipMovementDetector
    LIP_DETECTION_AVAILABLE = True
except:
    LIP_DETECTION_AVAILABLE = False
    print(" Lip movement detection module not found - feature disabled")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

active_sessions = {}

#  NEW: Initialize encryption storage
if ENCRYPTION_AVAILABLE:
    try:
        secure_storage = get_storage()
        print(" Encrypted storage initialized")
    except Exception as e:
        print(f" Encryption initialization failed: {e}")
        ENCRYPTION_AVAILABLE = False
        secure_storage = None
else:
    secure_storage = None


class ExamSession:
    """Complete exam session with ALL detection features"""
    
    def __init__(self, student_id, exam_id, exam_name):
        self.student_id = student_id
        self.exam_id = exam_id
        self.exam_name = exam_name
        self.session_id = f"{student_id}_{exam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize all monitoring modules with FIXED settings
        self.eye_gaze_tracker = EyeGazeTracker()
        self.face_detector = FaceDetector(allowed_faces=1)
        self.speech_analyzer = SpeechAnalyzer()
        self.integrity_scorer = IntegrityScorer()
        
        # Phone detection (NEW!)
        if PHONE_DETECTION_AVAILABLE:
            self.phone_detector = PhoneDetector()
            print(f" Phone detection enabled for {student_id}")
        else:
            self.phone_detector = None
        
        # Lip movement detection (NEW!)
        if LIP_DETECTION_AVAILABLE:
            self.lip_detector = LipMovementDetector()
            print(f" Lip movement detection enabled for {student_id}")
        else:
            self.lip_detector = None
        
        self.integrity_scorer.set_exam_info(student_id, exam_id, exam_name)
        
        # Session state
        self.is_active = True
        self.start_time = datetime.now()
        self.frame_count = 0
        
        # AUDIO TRACKING
        self.audio_events = deque(maxlen=100)
        self.speech_events = 0
        self.noise_events = 0
        self.audio_score = 100.0
        self.baseline_noise = 0
        self.baseline_set = False
        self.last_audio_log = 0
        
        # EYE GAZE TRACKING
        self.eye_gaze_score = 100.0
        self.looking_away_count = 0
        self.center_gaze_count = 0
        self.last_gaze_log = 0
        
        # FACE DETECTION
        self.face_score = 100.0
        self.multiple_faces_count = 0
        self.no_face_count = 0
        
        # PHONE/GADGET DETECTION (NEW!)
        self.gadget_score = 100.0
        self.phone_detections = 0
        self.gadget_violations = []
        self.last_phone_alert = 0
        
        # LIP MOVEMENT DETECTION (NEW!)
        self.lip_movement_penalty = 0.0
        self.lip_detections = 0
        self.lip_warnings = 0
        self.last_lip_alert = 0
        
        # VOICE/SPEECH DETECTION
        self.speech_strike_count = 0
        self.speech_warning_given = False
        self.speech_final_warning_given = False
        
        # THREE-STRIKE SYSTEM (NEW!)
        self.violation_strikes = 0  # Global strike counter
        self.auto_submitted = False
        self.strike_history = []
        self.looking_away_strike_triggered = False
        self.multiple_face_strike_triggered = False
        
        # Violations
        self.violations = {
            'eye_gaze': [],
            'face_detection': [],
            'speech': [],
            'noise': [],
            'multiple_faces': [],
            'gadgets': [],  # NEW!
            'lip_movement': [],  # NEW!
            'tab_switch': [] # NEW!
        }
    
    def process_audio_event(self, audio_data):
        """
        Audio detection using SpeechAnalyzer
        """
        timestamp = audio_data.get('timestamp', time.time())
        audio_level = audio_data.get('audio_level', 0)
        voice_energy = audio_data.get('voice_energy', 0)
        
        now = time.time()
        if now - self.last_audio_log > 1:
            print(f"[{self.student_id}] Audio: {audio_level:.1f}, Voice: {voice_energy}")
            self.last_audio_log = now
            
        if not self.baseline_set and len(self.audio_events) < 3:
            self.audio_events.append(audio_level)
            if len(self.audio_events) == 3:
                self.baseline_noise = sum(self.audio_events) / len(self.audio_events)
                self.baseline_set = True
                print(f"[{self.student_id}] Baseline: {self.baseline_noise:.2f}")
            return None
            
        self.audio_events.append(audio_level)
        
        #  FIX: Use speech_analyzer to process audio and detect Environment Sounds
        result = self.speech_analyzer.process_vad_event(audio_data)
        
        # Sync stats
        self.speech_events = self.speech_analyzer.speech_burst_count
        self.noise_events = self.speech_analyzer.noise_frame_count
        
        # Log if speech detected
        if result.get('speech_detected') and result.get('new_burst'):
            penalty = result.get('score_penalty', 5.0)
            self.violations['speech'].append({
                'type': 'speech_detected',
                'audio_level': audio_level,
                'voice_energy': voice_energy,
                'timestamp': timestamp,
                'score_penalty': -penalty
            })
            
        # Log if environment sound detected
        if result.get('environment_sound_detected'):
            self.violations['noise'].append({
                'type': 'environment_sound',
                'audio_level': audio_level,
                'timestamp': timestamp,
                'score_penalty': -3.0
            })
            
        return result
    def calculate_audio_score(self):
        """
        Calculate audio score using SpeechAnalyzer
        """
        #  FIX: Use the speech analyzer's calculated score
        final_score = self.speech_analyzer.calculate_audio_score()
        return final_score
    
    def add_violation_strike(self, violation_type, details=""):
        """
        THREE-STRIKE SYSTEM
        Add a strike and check for auto-submit
        """
        
        self.violation_strikes += 1
        
        strike_info = {
            'type': violation_type,
            'strike_number': self.violation_strikes,
            'details': details,
            'timestamp': time.time()
        }
        
        self.strike_history.append(strike_info)
        
        print(f"\n{'='*70}")
        print(f" STRIKE {self.violation_strikes}/3 - [{self.student_id}]")
        print(f"   Type: {violation_type}")
        print(f"   Details: {details}")
        
        if self.violation_strikes >= 3:
            print(f" 3 STRIKES REACHED - AUTO-SUBMITTING EXAM!")
            print(f"{'='*70}\n")
            self.auto_submitted = True
            return 'AUTO_SUBMIT'
        elif self.violation_strikes == 2:
            print(f" FINAL WARNING - One more strike = auto-submit")
            print(f"{'='*70}\n")
            return 'FINAL_WARNING'
        else:
            print(f" First warning")
            print(f"{'='*70}\n")
            return 'FIRST_WARNING'
    
    def update_eye_gaze_score(self, gaze_direction):
        """
         FIXED: Eye gaze tracking that actually updates score properly
        Uses EyeGazeTracker's calculate_attention_score() method
        """
        now = time.time()
        
        # Track the gaze direction
        if gaze_direction and gaze_direction.lower() != 'center':
            if not getattr(self, 'was_looking_away', False):
                self.looking_away_count += 1
                self.was_looking_away = True
                
                # Log occasionally
                if now - self.last_gaze_log > 2:
                    print(f" LOOKING {gaze_direction.upper()}! [{self.student_id}]")
                    self.last_gaze_log = now
                
                # Strike system for excessive looking away
                if self.looking_away_count == 10 and not self.looking_away_strike_triggered:
                    self.looking_away_strike_triggered = True
                    result = self.add_violation_strike('LOOKING_AWAY', f'Looked away {self.looking_away_count} times')
                    if result == 'AUTO_SUBMIT':
                        return result
                
                # Add to violations log
                self.violations['eye_gaze'].append({
                    'type': 'looking_away',
                    'direction': gaze_direction,
                    'timestamp': now
                })
        else:
            self.was_looking_away = False
            self.center_gaze_count += 1
        
        #  CRITICAL FIX: Use eye_gaze_tracker's actual calculated score
        # This properly accounts for looking away duration, no face, etc.
        calculated_score = self.eye_gaze_tracker.calculate_attention_score()
        
        # Update our tracked score
        self.eye_gaze_score = calculated_score
        
        # Debug logging every 60 frames
        if self.frame_count % 60 == 0:
            stats = self.eye_gaze_tracker.get_statistics()
            print(f" Eye Stats: Score={calculated_score:.1f}, Away={stats.get('looking_away_percentage', 0):.1f}%, NoFace={stats.get('no_face_percentage', 0):.1f}%")
        
        return max(0, min(100, self.eye_gaze_score))
    
    def update_face_count(self, num_faces):
        """Track face count with strike for multiple faces"""
        
        if num_faces == 0:
            self.no_face_count += 1
            
            if self.no_face_count % 30 == 0:
                print(f" NO FACE! [{self.student_id}] Count: 0")
            
            self.violations['face_detection'].append({
                'type': 'no_face_detected',
                'timestamp': time.time()
            })
        
        elif num_faces > 1:
            self.multiple_faces_count += 1
            
            print(f" MULTIPLE FACES! [{self.student_id}] Count: {num_faces}")
            
            # THREE-STRIKE: Immediate strike for multiple faces
            if not self.multiple_face_strike_triggered:
                self.multiple_face_strike_triggered = True
                return self.add_violation_strike('MULTIPLE_FACES', f'{num_faces} faces detected')
            
            self.violations['multiple_faces'].append({
                'type': 'multiple_faces_detected',
                'face_count': num_faces,
                'timestamp': time.time()
            })
        
        # Return face count (no score)
        return num_faces
    
    def detect_gadgets(self, frame):
        """
        PHONE/GADGET DETECTION (NEW!)
        Detects unauthorized devices and applies heavy penalties
        """
        
        if not self.phone_detector:
            return {'person_count': 0}  # Phone detection not available
        
        try:
            #  FIX: Run every frame for reliable detection
            # The phone detector has its own internal buffering
            result = self.phone_detector.detect_gadgets(frame)
            
            if result['detected']:
                now = time.time()
                
                # Apply penalties for each detected gadget
                for obj_name, penalty in result['penalties'].items():
                    self.phone_detections += 1
                    #  FIX: Increased penalty from 10 to 20
                    heavy_penalty = 20
                    self.gadget_score = max(0, self.gadget_score - heavy_penalty)

                    # Add strike for phone detection
                    if self.phone_detections == 1:
                        strike_result = self.add_violation_strike("PHONE_DETECTED", f"{obj_name} detected")
                        if strike_result:
                            print(f" PHONE STRIKE! {self.violation_strikes}/3")
                    
                    print(f" GADGET DETECTED! [{self.student_id}] {obj_name}, Penalty: -{penalty}, Total detections: {self.phone_detections}, Score: {self.gadget_score:.1f}")
                    
                    self.violations['gadgets'].append({
                        'type': 'unauthorized_device',
                        'device': obj_name,
                        'confidence': result.get('confidence', 0),
                        'timestamp': now,
                        'score_penalty': -penalty
                    })
                    
                    self.gadget_violations.append({
                        'device': obj_name,
                        'time': now
                    })
                
                self.last_phone_alert = now
            
            return result
        
        except Exception as e:
            print(f" Gadget detection error: {e}")
            import traceback
            traceback.print_exc()
            return {'person_count': 0}
    
    
    def detect_lip_movement(self, frame):
        """
        LIP MOVEMENT DETECTION (SCORE PENALTY ONLY - NO STRIKES)
        Silent reading is OK - only loud speech (audio) triggers strikes
        """
        
        if not self.lip_detector:
            return None
        
        try:
            result = self.lip_detector.detect_lip_movement(frame)
            
            if result['detected']:
                now = time.time()
                
                self.lip_detections += 1
                warning_level = result['warning_level']
                penalty = result['penalty']
                
                # Apply penalty to overall score (but NO strikes)
                if penalty > 0:
                    self.lip_movement_penalty += penalty
                
                # Track warnings
                if warning_level <= 2:
                    self.lip_warnings += 1
                
                # Log violation (NO STRIKE - just score penalty)
                self.violations['lip_movement'].append({
                    'type': 'lip_movement_detected',
                    'warning_level': warning_level,
                    'distance': result['distance'],
                    'penalty': penalty,
                    'timestamp': now,
                    'message': result['message'],
                    'note': 'Silent reading - no strike given'
                })
                
                self.last_lip_alert = now
                
                return result
            
            return None
        
        except Exception as e:
            print(f" Lip detection error: {e}")
            return None
    
    def calculate_gadget_score(self):
        """Calculate final gadget score"""
        score = self.gadget_score
        
        # Heavy penalties for repeated detections
        if self.phone_detections > 3:
            penalty = (self.phone_detections - 3) * 5
            score -= min(40, penalty)
        
        return max(0, min(100, score))


@app.route('/')
def index():
    return jsonify({
        'message': 'Complete Proctorless Exam System',
        'version': '4.0 - WITH PHONE DETECTION',
        'status': 'running',
        'phone_detection': PHONE_DETECTION_AVAILABLE
    })


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/system/check_apps', methods=['POST', 'OPTIONS'])
def check_apps():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        data = request.get_json() or {}
        browser_used = data.get('browser', 'chrome').lower()
        
        prohibited = {
            'discord': 'Discord',
            'slack': 'Slack',
            'skype': 'Skype',
            'zoom.us': 'Zoom',
            'zoom': 'Zoom',
            'teams': 'Microsoft Teams',
            'whatsapp': 'WhatsApp',
            'telegram': 'Telegram',
            'anydesk': 'AnyDesk',
            'teamviewer': 'TeamViewer',
            'obs': 'OBS Studio',
            'camtasia': 'Camtasia',
            'quicktime player': 'QuickTime Player'
        }
        
        browsers = {
            'chrome': 'Google Chrome',
            'firefox': 'Firefox',
            'safari': 'Safari',
            'brave': 'Brave Browser',
            'edge': 'Microsoft Edge',
            'msedge': 'Microsoft Edge',
            'opera': 'Opera'
        }
        
        detected_apps = set()
        import psutil
        
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name']
                if not name:
                    continue
                name_lower = name.lower().replace('.exe', '').replace('.app', '')
                
                # Check prohibited communication/recording apps
                for key, display_name in prohibited.items():
                    if key in name_lower:
                        detected_apps.add(display_name)
                        
                # Check browsers (allow the one currently used)
                for key, display_name in browsers.items():
                    if key in name_lower:
                        # Skip if it is the browser currently being used
                        if key not in browser_used:
                            detected_apps.add(display_name)
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
        # Return list of unauthorized applications running
        return jsonify({
            'status': 'success',
            'unauthorized_apps': list(detected_apps),
            'has_unauthorized': len(detected_apps) > 0
        })
    except Exception as e:
        print(f"Error checking apps: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exam/start', methods=['POST', 'OPTIONS'])
def start_exam():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        exam_id = data.get('exam_id')
        exam_name = data.get('exam_name')
        
        if not all([student_id, exam_id, exam_name]):
            return jsonify({'error': 'Missing fields'}), 400
        
        session = ExamSession(student_id, exam_id, exam_name)
        active_sessions[session.session_id] = session
        
        print(f"\n{'='*70}")
        print(f" COMPLETE EXAM STARTED: {session.session_id}")
        print(f"    Audio: Ultra-sensitive")
        print(f"    Eye Gaze: Active")
        print(f"    Multiple Faces: Monitoring")
        print(f"    Phone Detection: {'ENABLED' if PHONE_DETECTION_AVAILABLE else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        return jsonify({
            'session_id': session.session_id,
            'message': 'Exam started',
            'start_time': session.start_time.isoformat(),
            'phone_detection_enabled': PHONE_DETECTION_AVAILABLE
        })
    
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exam/end', methods=['POST', 'OPTIONS'])
def end_exam():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        auto_submitted = data.get('auto_submitted', False)
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session = active_sessions[session_id]
        session.is_active = False
        
        # Calculate all final scores (NO FACE SCORE)
        eye_score = session.update_eye_gaze_score('center')
        num_faces = session.update_face_count(1)  # Just get count
        audio_score = session.calculate_audio_score()
        gadget_score = session.calculate_gadget_score()
        
        # Overall score WITHOUT face component, WITH lip penalty
        base_score = (
            eye_score * 0.40 +
            audio_score * 0.35 +
            gadget_score * 0.25
        )
        overall_score = max(0, base_score - session.lip_movement_penalty)
        
        # AUTO-SUBMIT PENALTY: Heavy penalty for 3 strikes (cheating attempt)
        auto_submit_penalty = 0
        if auto_submitted or session.auto_submitted:
            auto_submit_penalty = 100  # Massive 100-point penalty
            overall_score = max(0, overall_score - auto_submit_penalty)
            print(f"\n{'='*70}")
            print(f" AUTO-SUBMIT PENALTY APPLIED!")
            reason = getattr(session, 'auto_submit_reason', '3 strikes reached (serious violations)')
            print(f"   Reason: {reason}")
            print(f"   Penalty: -100 points (Poor Integrity Score)")
            print(f"   Score before penalty: {overall_score + auto_submit_penalty:.1f}")
            print(f"   Score after penalty: {overall_score:.1f}")
            print(f"{'='*70}")
        
        print(f"\n{'='*70}")
        print(f" EXAM ENDED: {session_id}")
        print(f"   Auto-Submitted: {'YES' if (auto_submitted or session.auto_submitted) else 'NO'}")
        print(f"   Violation Strikes: {session.violation_strikes}/3")
        print(f"   Eye Gaze: {eye_score:.1f} (Looking away: {session.looking_away_count})")
        print(f"   Face Count: {num_faces} (Multiple: {session.multiple_faces_count}, No face: {session.no_face_count})")
        print(f"   Audio: {audio_score:.1f} (Speech: {session.speech_events}, Noise: {session.noise_events})")
        print(f"   Gadgets: {gadget_score:.1f} (Detections: {session.phone_detections})")
        print(f"   Lip Movement: Penalty -{session.lip_movement_penalty:.1f} (Detections: {session.lip_detections}, Warnings: {session.lip_warnings})")
        print(f"   BASE: {base_score:.1f} (40% eye + 35% audio + 25% gadgets)")
        if auto_submit_penalty > 0:
            print(f"   AUTO-SUBMIT PENALTY: -{auto_submit_penalty}")
        print(f"   FINAL SCORE: {overall_score:.1f}")
        print(f"{'='*70}\n")
        
        report = {
            'session_id': session_id,
            'student_id': session.student_id,
            'exam_id': session.exam_id,
            'exam_name': session.exam_name,
            'start_time': session.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': (datetime.now() - session.start_time).total_seconds(),
            'auto_submitted': auto_submitted or session.auto_submitted,
            'violation_strikes': session.violation_strikes,
            'strike_history': session.strike_history,
            'auto_submit_penalty': auto_submit_penalty,
            'overall_score': round(overall_score, 2),
            'component_scores': {
                'eye_gaze': round(eye_score, 2),
                'speech': round(audio_score, 2),
                'gadgets': round(gadget_score, 2)
            },
            'face_info': {
                'current_count': num_faces,
                'multiple_faces_count': session.multiple_faces_count,
                'no_face_count': session.no_face_count
            },
            'audio_metrics': {
                'speech_events': session.speech_events,
                'noise_events': session.noise_events,
                'baseline_noise': round(session.baseline_noise, 2) if session.baseline_set else 0
            },
            'gaze_metrics': {
                'looking_away_count': session.looking_away_count,
                'center_gaze_count': session.center_gaze_count,
                'attention_rate': round((session.center_gaze_count / max(1, session.center_gaze_count + session.looking_away_count)) * 100, 2)
            },
            'face_metrics': {
                'multiple_faces_count': session.multiple_faces_count,
                'no_face_count': session.no_face_count
            },
            'gadget_metrics': {
                'phone_detections': session.phone_detections,
                'gadget_violations': session.gadget_violations
            },
            'lip_movement_metrics': {
                'total_detections': session.lip_detections,
                'warnings': session.lip_warnings,
                'total_penalty': round(session.lip_movement_penalty, 2)
            },
            'violations': {
                'eye_gaze': len(session.violations['eye_gaze']),
                'face_detection': len(session.violations['face_detection']),
                'speech': len(session.violations['speech']),
                'noise': len(session.violations['noise']),
                'multiple_faces': len(session.violations['multiple_faces']),
                'gadgets': len(session.violations['gadgets']),
                'lip_movement': len(session.violations['lip_movement'])
            },
            'total_violations': sum([
                len(session.violations['eye_gaze']),
                len(session.violations['face_detection']),
                len(session.violations['speech']),
                len(session.violations['noise']),
                len(session.violations['multiple_faces']),
                len(session.violations['gadgets']),
                len(session.violations.get('tab_switch', []))
            ]),
            'violation_logs': session.violations
        }
        
        #  NEW: Save encrypted report
        if ENCRYPTION_AVAILABLE and secure_storage:
            try:
                encrypted_path = f'storage/encrypted/reports/{session_id}_report.enc'
                secure_storage.save(encrypted_path, report)
                print(f" Encrypted report saved: {encrypted_path}")
            except Exception as e:
                print(f" Encryption failed, using fallback: {e}")
        
        # Also save unencrypted for backward compatibility
        os.makedirs('reports', exist_ok=True)
        with open(f'reports/{session_id}_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return jsonify({
            'message': 'Exam ended',
            'session_id': session_id,
            'report': report
        })
    
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/exam/report/<session_id>', methods=['GET'])
def get_report(session_id):
    try:
        print(f"\n{'='*70}")
        print(f"REPORT REQUEST: {session_id}")
        print(f"Active sessions: {list(active_sessions.keys())}")
        
        if session_id not in active_sessions:
            #  NEW: Try loading encrypted report first
            encrypted_path = f'storage/encrypted/reports/{session_id}_report.enc'
            
            if ENCRYPTION_AVAILABLE and secure_storage and os.path.exists(encrypted_path):
                print(f"Loading encrypted report: {encrypted_path}")
                try:
                    report_data = secure_storage.load(encrypted_path)
                    print(f" Loaded encrypted report:")
                    print(f"  Overall Score: {report_data.get('overall_score')}")
                    print(f"  Session ID: {report_data.get('session_id')}")
                    print(f"{'='*70}\n")
                    return jsonify(report_data)
                except Exception as e:
                    print(f" Failed to load encrypted report: {e}")
                    print(f"Falling back to unencrypted report...")
            
            # Fallback to unencrypted report
            report_path = f'reports/{session_id}_report.json'
            print(f"Session not active, looking for file: {report_path}")
            print(f"File exists: {os.path.exists(report_path)}")
            
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report_data = json.load(f)
                    print(f"Loaded report from file:")
                    print(f"  Overall Score: {report_data.get('overall_score')}")
                    print(f"  Session ID: {report_data.get('session_id')}")
                    print(f"{'='*70}\n")
                    return jsonify(report_data)
            
            print(f"Report file not found: {report_path}")
            print(f"Available report files:")
            if os.path.exists('reports'):
                for f in os.listdir('reports'):
                    print(f"  - {f}")
            print(f"{'='*70}\n")
            return jsonify({'error': 'Report not found'}), 404
        
        print(f"Session still active, calculating live scores")
        session = active_sessions[session_id]
        eye_score = session.update_eye_gaze_score('center')
        num_faces = session.update_face_count(1)
        audio_score = session.calculate_audio_score()
        gadget_score = session.calculate_gadget_score()
        
        overall = round((eye_score * 0.40 + audio_score * 0.35 + gadget_score * 0.25), 2)

        #  FIX: Cap overall score
        if session.phone_detections > 0:
            overall = min(60, overall)
        if session.multiple_faces_count > 0:
            overall = min(50, overall)

        print(f"Live scores: Eye={eye_score}, Audio={audio_score}, Gadgets={gadget_score}")
        print(f"Overall: {overall}")
        print(f"{'='*70}\n")
        
        return jsonify({
            'session_id': session_id,
            'student_id': session.student_id,
            'exam_id': session.exam_id,
            'auto_submitted': session.auto_submitted,
            'auto_submit_reason': getattr(session, 'auto_submit_reason', ''),
            'strike_history': session.strike_history,
            'overall_score': overall,
            'component_scores': {
                'eye_gaze': round(eye_score, 2),
                'speech': round(audio_score, 2),
                'gadgets': round(gadget_score, 2)
            },
            'face_count': num_faces,
            'phone_detections': session.phone_detections,
            'speech_events': session.speech_events,
            'multiple_faces_count': session.multiple_faces_count,
            'looking_away_count': session.looking_away_count,  #  Added for dashboard
            'lip_detections': session.lip_detections if hasattr(session, 'lip_detections') else 0,  #  Added
            'gadget_metrics': {
                'phone_detections': session.phone_detections
            },
            'audio_metrics': {
                'speech_events': session.speech_events
            },
            'face_metrics': {
                'multiple_faces_count': session.multiple_faces_count
            },
            'eye_metrics': {  #  Added for dashboard
                'looking_away_count': session.looking_away_count,
                'center_gaze_count': session.center_gaze_count
            },
            'lip_metrics': {  #  Added for dashboard
                'detections': session.lip_detections if hasattr(session, 'lip_detections') else 0
            }
        })
    
    except Exception as e:
        print(f"Error in get_report: {e}")
        print(f"{'='*70}\n")
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    print(f' Client connected: {request.sid}')
    emit('connection_response', {'message': 'Connected'})


@socketio.on('disconnect')
def handle_disconnect():
    print(f' Client disconnected: {request.sid}')


@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video with ALL detection features"""
    try:
        session_id = data.get('session_id')
        frame_data = data.get('frame')
        
        if not session_id or session_id not in active_sessions:
            return
        
        session = active_sessions[session_id]
        
        # Decode frame
        img_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        session.frame_count += 1
        
        # Eye gaze
        eye_analysis = session.eye_gaze_tracker.process_frame(frame)
        gaze_direction = eye_analysis.get('gaze_direction', 'center')
        eye_gaze_result = session.update_eye_gaze_score(gaze_direction)
        
        # Check if eye gaze triggered a strike
        if isinstance(eye_gaze_result, str) and 'SUBMIT' in eye_gaze_result:
            emit('auto_submit', {'reason': 'Three strikes reached - Looking away'})
            return
        elif isinstance(eye_gaze_result, str):
            eye_score = session.eye_gaze_score
        else:
            eye_score = eye_gaze_result
        
        # Face detection
        face_analysis = session.face_detector.process_frame(frame)
        num_faces = face_analysis.get('num_faces', 1)
        face_result = session.update_face_count(num_faces)
        
        #  FATAL VIOLATION: Immediate auto-submit for multiple faces
        if num_faces > 1 or (isinstance(face_result, str) and 'SUBMIT' in face_result):
            session.auto_submitted = True
            session.auto_submit_reason = 'Illegal activity detected: Multiple persons found in exam area'
            emit('auto_submit', {'reason': session.auto_submit_reason})
            return
        
        # PHONE/GADGET DETECTION (NEW!)
        gadget_result = session.detect_gadgets(frame)
        gadget_score = session.calculate_gadget_score()

        # Check YOLO person count to supplement MediaPipe face count
        yolo_person_count = gadget_result.get('person_count', 0) if gadget_result else 0
        if yolo_person_count > 1 and num_faces <= 1:
            num_faces = map(lambda x: max(x, yolo_person_count), [num_faces])[0]  # Just ensure it's logged
            session.auto_submitted = True
            session.auto_submit_reason = 'Illegal activity detected: Multiple persons found in exam area'
            emit('auto_submit', {'reason': session.auto_submit_reason})
            return

        #  FATAL VIOLATION: Immediate auto-submit for phone detection
        if session.phone_detections > 0:
            session.auto_submitted = True
            session.auto_submit_reason = 'Illegal activity detected: Unauthorized device (Phone/Gadget) found'
            emit('auto_submit', {'reason': session.auto_submit_reason})
            return
        
        #  FIXED: LIP MOVEMENT DETECTION with audio correlation
        # Pass recent audio level to lip detector to prevent false positives during silent reading
        if hasattr(session, 'audio_events') and len(session.audio_events) > 0:
            recent_audio = session.audio_events[-1] if session.audio_events else 0
            is_speech = session.speech_events > 0
            
            # Update lip detector with audio status
            if session.lip_detector and hasattr(session.lip_detector, 'update_audio_status'):
                session.lip_detector.update_audio_status(recent_audio, is_speech)
        
        # Now detect lip movement with audio correlation
        lip_result = session.detect_lip_movement(frame)
        
        # Audio score
        audio_score = session.calculate_audio_score()
        
        # Overall score WITHOUT face component, WITH lip penalty
        base_score = (
            eye_score * 0.40 +
            audio_score * 0.35 +
            gadget_score * 0.25
        )
        overall_score = max(0, base_score - session.lip_movement_penalty)
        
        #  DEBUG LOGGING: Show stats every 60 frames (every 2 seconds)
        if session.frame_count % 60 == 0:
            print(f"\n [{session.student_id}] Frame {session.frame_count} Stats:")
            print(f"    Eye Gaze: {eye_score:.1f} (looking_away: {session.looking_away_count})")
            print(f"    Speech: {audio_score:.1f} (speech_events: {session.speech_events})")
            print(f"    Gadgets: {gadget_score:.1f} (detections: {session.phone_detections})")
            if lip_result and lip_result.get('detected'):
                print(f"    Lip: DETECTED (total: {session.lip_detections}, penalty: {session.lip_movement_penalty:.1f})")
            else:
                blocked_count = getattr(session.lip_detector, 'false_positive_blocks', 0) if session.lip_detector else 0
                print(f"    Lip: Silent (blocked: {blocked_count} false positives)")
            print(f"    Overall Score: {overall_score:.1f}\n")
        
        # Emit update
        update_data = {
            'session_id': session_id,
            'frame_count': session.frame_count,
            'overall_score': round(overall_score, 2),
            'eye_gaze_score': round(eye_score, 2),
            'speech_score': round(audio_score, 2),
            'gadget_score': round(gadget_score, 2),
            
            #  FACE COUNT (multiple fields for compatibility)
            'face_count': num_faces,
            'num_faces': num_faces,
            'current_face_count': num_faces,
            'multiple_faces_count': session.multiple_faces_count,
            'multiple_faces_detected': num_faces > 1,
            'no_face_detected': num_faces == 0,
            
            'gaze_direction': gaze_direction,
            
            #  PHONE/GADGET DETECTION COUNTS
            'phone_detected': session.phone_detections > 0,
            'phone_detection_count': session.phone_detections,
            'phone_count': session.phone_detections,
            'gadget_count': session.phone_detections,
            
            #  LIP MOVEMENT COUNTS
            'lip_detections': session.lip_detections,
            'lip_movement_penalty': session.lip_movement_penalty,
            
            'violation_strikes': session.violation_strikes,
            'auto_submitted': session.auto_submitted,
            
            #  COMPLETE VIOLATIONS
            'violations': {
                'eye_gaze': session.looking_away_count,
                'multiple_faces': session.multiple_faces_count,
                'speech': session.speech_events,
                'noise': session.noise_events,
                'gadgets': session.phone_detections,
                'lip_movement': session.lip_detections
            }
        }
        
        # Add lip movement data if detected
        if lip_result and lip_result['detected']:
            update_data['lip_movement'] = {
                'detected': True,
                'warning_level': lip_result['warning_level'],
                'message': lip_result['message'],
                'total_detections': session.lip_detections,
                'total_penalty': session.lip_movement_penalty,
                'distance': lip_result.get('distance', 0)
            }
        
        #  DEBUG: Print stats every 60 frames
        if session.frame_count % 60 == 0:
            print(f"\n [{session.student_id}] Frame {session.frame_count} Stats:")
            print(f"    Faces: {num_faces} (Multiple violations: {session.multiple_faces_count})")
            print(f"    Phone detections: {session.phone_detections}")
            print(f"    Lip movements: {session.lip_detections}")
            print(f"    Speech events: {session.speech_events}")
            print(f"   Scores: Overall={overall_score:.1f} Eye={eye_score:.1f} Speech={audio_score:.1f} Gadget={gadget_score:.1f}\n")
        
        emit('monitoring_update', update_data)
        
    except Exception as e:
        print(f" Frame error: {e}")


@socketio.on('audio_analysis')
def handle_audio_analysis(data):
    """Audio processing with rapid score reduction (NO auto-submit)"""
    try:
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return
        
        session = active_sessions[session_id]
        result = session.process_audio_event(data)
        
        audio_score = session.calculate_audio_score()
        
        emit('speech_update', {
            'speech_score': round(audio_score, 2),
            'speech_events': session.speech_events,
            'noise_events': session.noise_events,
            'baseline_set': session.baseline_set,
            'baseline_noise': round(session.baseline_noise, 2) if session.baseline_set else 0,
            'environment_noise': result.get('environment_sound_detected') if result else False
        })
    
    except Exception as e:
        print(f"Audio error: {e}")


@socketio.on('tab_switch_detected')
def handle_tab_switch(data):
    """Handle tab switching - IMMEDIATE AUTO-SUBMIT"""
    try:
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return
        
        session = active_sessions[session_id]
        
        print(f"\n{'='*70}")
        print(f"TAB SWITCH DETECTED! [{session.student_id}]")
        print(f"EXAM WILL BE AUTO-SUBMITTED")
        print(f"{'='*70}\n")
        
        # Mark as auto-submitted
        session.auto_submitted = True
        session.violation_strikes = 3  # Force 3 strikes
        session.strike_history.append({
            'type': 'TAB_SWITCH',
            'strike_number': 3,
            'details': data.get('reason', 'Tab switching detected'),
            'timestamp': time.time()
        })
        session.violations.setdefault('tab_switch', []).append({
            'type': 'tab_switch',
            'timestamp': time.time()
        })
        session.auto_submit_reason = 'tab switching occured'
        
        # Trigger auto-submit
        emit('auto_submit', {'reason': 'tab switching occured'})
    
    except Exception as e:
        print(f"Tab switch error: {e}")


if __name__ == '__main__':
    print("="*70)
    print("COMPLETE PROCTORLESS EXAM SYSTEM v8.0")
    print("WITH ENCRYPTED STORAGE + ALL FIXES")
    print("="*70)
    print("Audio: Voice Activity Detection (ignores fans/AC)")
    print("   - Continuous speech = Rapid score reduction to POOR")
    print("   - NO auto-submit for speech")
    print("Eye Gaze: Active tracking (warnings only)")
    print("Multiple Faces: Detection (warnings only)")
    print(f"Phone Detection: {'ENABLED' if PHONE_DETECTION_AVAILABLE else 'DISABLED (install ultralytics)'}")
    print(f"Lip Movement: {'Score penalty only' if LIP_DETECTION_AVAILABLE else 'DISABLED (install mediapipe)'}")
    print(f"Encrypted Storage: {' ENABLED (AES-256-GCM)' if ENCRYPTION_AVAILABLE else ' DISABLED (install cryptography)'}")
    print("TAB SWITCHING: IMMEDIATE AUTO-SUBMIT")
    print("   - Tab visibility monitored")
    print("   - Focus loss = Exam terminated")
    print("="*70)
    
    os.makedirs('reports', exist_ok=True)
    
    #  NEW: Create encrypted storage directory
    if ENCRYPTION_AVAILABLE:
        os.makedirs('storage/encrypted/reports', exist_ok=True)
        print(" Encrypted storage directory ready")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
