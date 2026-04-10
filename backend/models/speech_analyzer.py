"""
Speech Analyzer - DETECTS ENVIRONMENT SOUNDS
Now detects TV, music, background conversations, etc.
"""

import numpy as np
from collections import deque
import time


class SpeechAnalyzer:
    """Detects both speech AND environment sounds (TV, music, conversations)"""

    def __init__(self,
                 suspicious_keywords=None,
                 max_history=200,
                 audio_threshold=500):

        # Voice activity tracking
        self.audio_events = deque(maxlen=max_history)
        self.speech_segments = []
        self.noise_events = deque(maxlen=max_history)
        self.environment_sound_events = deque(maxlen=max_history)  #  NEW

        # Session stats
        self.total_frames = 0
        self.speech_frame_count = 0
        self.noise_frame_count = 0
        self.speech_burst_count = 0
        self.environment_sound_count = 0  #  NEW
        self.last_speech_time = 0
        self.last_env_sound_time = 0  #  NEW
        self.burst_gap_threshold = 2.0
        self._in_burst = False
        self._in_env_sound = False  #  NEW

        #  OPTIMIZED SCORING with environment sound detection
        self.base_score = 100.0
        self.score_penalty = 0.0
        self.SPEECH_PENALTY_PER_BURST = 5.0
        self.NOISE_PENALTY_PER_EVENT = 0.2
        self.ENV_SOUND_PENALTY_PER_EVENT = 3.0  #  NEW: TV/music penalty
        
        #  LOWER THRESHOLDS to detect environment sounds
        self.MIN_VOICE_ENERGY = 2000  # Reduced from 3000 (more sensitive)
        self.MIN_AUDIO_LEVEL = 20     # Reduced from 25 (catches TV/music)
        self.ENV_SOUND_THRESHOLD = 30  #  NEW: Continuous sound threshold

        self.start_time = time.time()
        self.suspicious_keywords = suspicious_keywords or []

        print(" Speech Analyzer (Environment Sound Detection)")
        print(f"   - Speech penalty: {self.SPEECH_PENALTY_PER_BURST} pts/burst")
        print(f"   - Environment sound penalty: {self.ENV_SOUND_PENALTY_PER_EVENT} pts/event")
        print(f"   - Detects: TV, music, conversations")

    def process_vad_event(self, audio_data: dict) -> dict:
        """Process VAD event with environment sound detection"""
        self.total_frames += 1
        audio_level  = float(audio_data.get('audio_level', 0))
        voice_energy = float(audio_data.get('voice_energy', 0))
        is_speech    = bool(audio_data.get('is_speech', False))
        ts = float(audio_data.get('timestamp', time.time() * 1000)) / 1000.0

        result = {
            'speech_detected': False,
            'noise_detected': False,
            'environment_sound_detected': False,  #  NEW
            'new_burst': False,
            'score_penalty': 0.0,
        }

        now = time.time()

        #  1. DETECT DIRECT SPEECH (voice patterns with high energy)
        if is_speech and voice_energy > self.MIN_VOICE_ENERGY:
            self.speech_frame_count += 1

            # Detect burst start
            if not self._in_burst or (now - self.last_speech_time > self.burst_gap_threshold):
                self._in_burst = True
                self.speech_burst_count += 1
                result['new_burst'] = True

                # Apply penalty
                self.score_penalty += self.SPEECH_PENALTY_PER_BURST

                self.speech_segments.append({
                    'burst_number': self.speech_burst_count,
                    'timestamp': now - self.start_time,
                    'audio_level': audio_level,
                    'voice_energy': voice_energy,
                    'type': 'direct_speech'
                })

            self.last_speech_time = now
            result['speech_detected'] = True
            result['score_penalty'] = self.SPEECH_PENALTY_PER_BURST if result['new_burst'] else 0.0

            self.audio_events.append({
                'type': 'speech', 'level': audio_level, 'ts': now - self.start_time
            })

        #  2. DETECT ENVIRONMENT SOUNDS (TV, music, background conversations)
        elif audio_level > self.ENV_SOUND_THRESHOLD:
            # Reset speech burst
            self._in_burst = False
            
            # Detect environment sound event
            if not self._in_env_sound or (now - self.last_env_sound_time > self.burst_gap_threshold):
                self._in_env_sound = True
                self.environment_sound_count += 1

                # Apply penalty for environment sounds
                self.score_penalty += self.ENV_SOUND_PENALTY_PER_EVENT

                self.speech_segments.append({
                    'burst_number': self.environment_sound_count,
                    'timestamp': now - self.start_time,
                    'audio_level': audio_level,
                    'voice_energy': voice_energy,
                    'type': 'environment_sound'  #  TV, music, etc.
                })

                print(f" Environment sound #{self.environment_sound_count} | level={audio_level:.1f}")

            self.last_env_sound_time = now
            result['environment_sound_detected'] = True
            result['score_penalty'] = self.ENV_SOUND_PENALTY_PER_EVENT

            self.environment_sound_events.append({
                'type': 'environment', 'level': audio_level, 'ts': now - self.start_time
            })

        #  3. DETECT AMBIENT NOISE (lower levels)
        elif audio_level > self.MIN_AUDIO_LEVEL:
            self._in_burst = False
            self._in_env_sound = False
            self.noise_frame_count += 1
            self.score_penalty += self.NOISE_PENALTY_PER_EVENT
            result['noise_detected'] = True
            result['score_penalty'] = self.NOISE_PENALTY_PER_EVENT

            self.noise_events.append({
                'type': 'noise', 'level': audio_level, 'ts': now - self.start_time
            })

        # 4. SILENCE
        else:
            if self._in_burst and (now - self.last_speech_time > self.burst_gap_threshold):
                self._in_burst = False
            if self._in_env_sound and (now - self.last_env_sound_time > self.burst_gap_threshold):
                self._in_env_sound = False

        return result

    def calculate_speech_integrity_score(self) -> float:
        """Calculate speech integrity score"""
        score = self.base_score - self.score_penalty
        return round(max(0.0, min(100.0, score)), 2)

    def calculate_audio_score(self) -> float:
        """Alias for compatibility"""
        return self.calculate_speech_integrity_score()

    def get_statistics(self) -> dict:
        """Get statistics including environment sounds"""
        elapsed = max(1.0, time.time() - self.start_time)
        return {
            'total_frames': self.total_frames,
            'speech_frame_count': self.speech_frame_count,
            'noise_frame_count': self.noise_frame_count,
            'speech_burst_count': self.speech_burst_count,
            'environment_sound_count': self.environment_sound_count,  #  NEW
            'session_duration_seconds': round(elapsed, 2),
            'bursts_per_minute': round(self.speech_burst_count / (elapsed / 60), 2),
            'speech_rate_pct': round((self.speech_frame_count / max(1, self.total_frames)) * 100, 2),
            'score_penalty': round(self.score_penalty, 2),
            'integrity_score': self.calculate_speech_integrity_score(),
        }

    @property
    def speech_events(self) -> int:
        return self.speech_burst_count + self.environment_sound_count  #  Include env sounds

    @property
    def noise_events_count(self) -> int:
        return self.noise_frame_count

    def get_transcript_summary(self) -> list:
        return self.speech_segments

    def get_violation_timeline(self) -> list:
        return [s['timestamp'] for s in self.speech_segments]

    def process_audio_segment(self, audio_data) -> dict:
        """Legacy method for compatibility"""
        if isinstance(audio_data, dict):
            return self.process_vad_event(audio_data)
        return {'speech_detected': False, 'transcript': '', 'is_suspicious': False, 'violations': []}

    def reset(self):
        """Reset analyzer"""
        self.audio_events.clear()
        self.noise_events.clear()
        self.environment_sound_events.clear()
        self.speech_segments.clear()
        self.total_frames = 0
        self.speech_frame_count = 0
        self.noise_frame_count = 0
        self.speech_burst_count = 0
        self.environment_sound_count = 0
        self.last_speech_time = 0
        self.last_env_sound_time = 0
        self._in_burst = False
        self._in_env_sound = False
        self.score_penalty = 0.0
        self.start_time = time.time()
