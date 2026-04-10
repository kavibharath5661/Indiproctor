"""
Integrity Scoring Module
Combines eye-gaze, face detection, and speech analysis
to calculate comprehensive exam integrity score
"""

import numpy as np
from datetime import datetime
import json


class IntegrityScorer:
    """
    Comprehensive integrity scoring system that combines multiple
    monitoring modules to produce a final exam integrity score
    """
    
    def __init__(self, 
                 eye_gaze_weight=0.35,
                 face_detection_weight=0.40,
                 speech_analysis_weight=0.25):
        """
        Initialize the integrity scorer with weights for each component
        
        Args:
            eye_gaze_weight: Weight for eye gaze tracking score (0-1)
            face_detection_weight: Weight for face detection score (0-1)
            speech_analysis_weight: Weight for speech analysis score (0-1)
        """
        # Validate weights sum to 1.0
        total_weight = eye_gaze_weight + face_detection_weight + speech_analysis_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.weights = {
            'eye_gaze': eye_gaze_weight,
            'face_detection': face_detection_weight,
            'speech_analysis': speech_analysis_weight
        }
        
        # Score history
        self.score_history = []
        self.violation_log = []
        
        # Component scores
        self.component_scores = {
            'eye_gaze': 100.0,
            'face_detection': 100.0,
            'speech_analysis': 100.0
        }
        
        # Overall score
        self.overall_score = 100.0
        
        # Exam metadata
        self.exam_info = {}
        self.start_time = datetime.now()
    
    def update_scores(self, eye_gaze_score=None, face_score=None, speech_score=None):
        """
        Update component scores and calculate overall score
        
        Args:
            eye_gaze_score: Eye gaze tracking score (0-100)
            face_score: Face detection integrity score (0-100)
            speech_score: Speech analysis integrity score (0-100)
        """
        # Update component scores
        if eye_gaze_score is not None:
            self.component_scores['eye_gaze'] = eye_gaze_score
        
        if face_score is not None:
            self.component_scores['face_detection'] = face_score
        
        if speech_score is not None:
            self.component_scores['speech_analysis'] = speech_score
        
        # Calculate weighted overall score
        overall = 0
        for component, score in self.component_scores.items():
            overall += score * self.weights[component]
        
        self.overall_score = round(overall, 2)
        
        # Store in history
        self.score_history.append({
            'timestamp': datetime.now().isoformat(),
            'overall_score': self.overall_score,
            'component_scores': self.component_scores.copy()
        })
        
        return self.overall_score
    
    def log_violation(self, violation_type, component, severity='medium', details=None):
        """
        Log a violation event
        
        Args:
            violation_type: Type of violation
            component: Component that detected the violation
            severity: Severity level ('low', 'medium', 'high', 'critical')
            details: Additional details about the violation
        """
        violation = {
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'component': component,
            'severity': severity,
            'details': details or {}
        }
        
        self.violation_log.append(violation)
    
    def get_integrity_level(self):
        """
        Get integrity level based on overall score
        
        Returns:
            str: Integrity level category
        """
        score = self.overall_score
        
        if score >= 90:
            return 'EXCELLENT'
        elif score >= 75:
            return 'GOOD'
        elif score >= 60:
            return 'ACCEPTABLE'
        elif score >= 40:
            return 'QUESTIONABLE'
        else:
            return 'POOR'
    
    def get_risk_assessment(self):
        """
        Assess the risk level of cheating based on overall score
        
        Returns:
            dict: Risk assessment details
        """
        score = self.overall_score
        
        if score >= 85:
            risk_level = 'LOW'
            recommendation = 'No action required. Exam conducted with high integrity.'
        elif score >= 70:
            risk_level = 'MEDIUM-LOW'
            recommendation = 'Minor concerns. Review flagged incidents.'
        elif score >= 50:
            risk_level = 'MEDIUM'
            recommendation = 'Moderate concerns. Manual review recommended.'
        elif score >= 30:
            risk_level = 'MEDIUM-HIGH'
            recommendation = 'Significant concerns. Detailed investigation required.'
        else:
            risk_level = 'HIGH'
            recommendation = 'Severe integrity concerns. Re-examination may be necessary.'
        
        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'overall_score': self.overall_score,
            'integrity_level': self.get_integrity_level()
        }
    
    def get_violation_summary(self):
        """
        Get summary of all violations
        
        Returns:
            dict: Violation summary by type and severity
        """
        if not self.violation_log:
            return {
                'total_violations': 0,
                'by_severity': {},
                'by_component': {},
                'by_type': {}
            }
        
        summary = {
            'total_violations': len(self.violation_log),
            'by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'by_component': {},
            'by_type': {}
        }
        
        for violation in self.violation_log:
            # Count by severity
            severity = violation.get('severity', 'medium')
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by component
            component = violation.get('component', 'unknown')
            summary['by_component'][component] = summary['by_component'].get(component, 0) + 1
            
            # Count by type
            v_type = violation.get('type', 'unknown')
            summary['by_type'][v_type] = summary['by_type'].get(v_type, 0) + 1
        
        return summary
    
    def get_detailed_report(self):
        """
        Generate comprehensive integrity report
        
        Returns:
            dict: Detailed integrity report
        """
        duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            'exam_info': self.exam_info,
            'session_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'duration_formatted': self._format_duration(duration)
            },
            'overall_score': self.overall_score,
            'integrity_level': self.get_integrity_level(),
            'risk_assessment': self.get_risk_assessment(),
            'component_scores': self.component_scores.copy(),
            'component_weights': self.weights.copy(),
            'violation_summary': self.get_violation_summary(),
            'detailed_violations': self.violation_log.copy(),
            'score_timeline': self.score_history.copy()
        }
        
        return report
    
    def get_score_trend(self):
        """
        Analyze score trend over time
        
        Returns:
            dict: Trend analysis
        """
        if len(self.score_history) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        
        # Get first and last scores
        first_score = self.score_history[0]['overall_score']
        last_score = self.score_history[-1]['overall_score']
        
        change = last_score - first_score
        
        if change > 5:
            trend = 'improving'
        elif change < -5:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Calculate average score
        avg_score = np.mean([s['overall_score'] for s in self.score_history])
        
        # Calculate standard deviation
        std_score = np.std([s['overall_score'] for s in self.score_history])
        
        return {
            'trend': trend,
            'change': round(change, 2),
            'initial_score': round(first_score, 2),
            'final_score': round(last_score, 2),
            'average_score': round(avg_score, 2),
            'score_volatility': round(std_score, 2)
        }
    
    def set_exam_info(self, student_id, exam_id, exam_name, duration=None):
        """
        Set exam metadata
        
        Args:
            student_id: Student identifier
            exam_id: Exam identifier
            exam_name: Name of the exam
            duration: Expected exam duration in minutes
        """
        self.exam_info = {
            'student_id': student_id,
            'exam_id': exam_id,
            'exam_name': exam_name,
            'expected_duration_minutes': duration
        }
    
    def generate_recommendations(self):
        """
        Generate recommendations based on scores and violations
        
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Check each component
        if self.component_scores['eye_gaze'] < 70:
            recommendations.append({
                'component': 'eye_gaze',
                'severity': 'medium',
                'recommendation': 'Student frequently looked away from screen. Review exam recording.'
            })
        
        if self.component_scores['face_detection'] < 60:
            recommendations.append({
                'component': 'face_detection',
                'severity': 'high',
                'recommendation': 'Multiple face detection or face absence issues. Immediate review required.'
            })
        
        if self.component_scores['speech_analysis'] < 80:
            recommendations.append({
                'component': 'speech_analysis',
                'severity': 'medium',
                'recommendation': 'Unauthorized speech detected. Review audio logs for potential collaboration.'
            })
        
        # Check violation patterns
        violation_summary = self.get_violation_summary()
        if violation_summary['total_violations'] > 20:
            recommendations.append({
                'component': 'overall',
                'severity': 'high',
                'recommendation': f"High number of violations ({violation_summary['total_violations']}). Consider re-examination."
            })
        
        # Check overall score
        if self.overall_score < 50:
            recommendations.append({
                'component': 'overall',
                'severity': 'critical',
                'recommendation': 'Overall integrity score is critically low. Manual review and possible re-examination strongly recommended.'
            })
        
        return recommendations
    
    def export_to_json(self, filepath=None):
        """
        Export report to JSON file
        
        Args:
            filepath: Path to save JSON file
            
        Returns:
            str: JSON string if filepath is None, otherwise None
        """
        report = self.get_detailed_report()
        report['recommendations'] = self.generate_recommendations()
        report['score_trend'] = self.get_score_trend()
        
        json_str = json.dumps(report, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        else:
            return json_str
    
    def _format_duration(self, seconds):
        """
        Format duration in human-readable format
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            str: Formatted duration
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def reset(self):
        """Reset all scores and history"""
        self.score_history.clear()
        self.violation_log.clear()
        self.component_scores = {
            'eye_gaze': 100.0,
            'face_detection': 100.0,
            'speech_analysis': 100.0
        }
        self.overall_score = 100.0
        self.start_time = datetime.now()


# Example usage
if __name__ == "__main__":
    print("Integrity Scorer Test")
    print("=" * 60)
    
    # Initialize scorer
    scorer = IntegrityScorer()
    
    # Set exam info
    scorer.set_exam_info(
        student_id="21MID0009",
        exam_id="CAP2024_AI",
        exam_name="AI Capstone Examination",
        duration=120
    )
    
    # Simulate score updates
    print("\nSimulating exam monitoring...")
    
    # Initial scores (perfect)
    scorer.update_scores(
        eye_gaze_score=100,
        face_score=100,
        speech_score=100
    )
    print(f"Initial Score: {scorer.overall_score}")
    
    # Some violations occur
    scorer.log_violation('looking_away', 'eye_gaze', 'low', {'direction': 'left'})
    scorer.update_scores(eye_gaze_score=85, face_score=100, speech_score=100)
    print(f"After eye gaze violation: {scorer.overall_score}")
    
    scorer.log_violation('multiple_faces', 'face_detection', 'high', {'faces_detected': 2})
    scorer.update_scores(eye_gaze_score=85, face_score=60, speech_score=100)
    print(f"After face detection violation: {scorer.overall_score}")
    
    scorer.log_violation('suspicious_keywords', 'speech_analysis', 'medium', {'keywords': ['help', 'answer']})
    scorer.update_scores(eye_gaze_score=85, face_score=60, speech_score=75)
    print(f"After speech violation: {scorer.overall_score}")
    
    # Generate report
    print("\n" + "=" * 60)
    print("FINAL INTEGRITY REPORT")
    print("=" * 60)
    
    report = scorer.get_detailed_report()
    
    print(f"\nOverall Integrity Score: {report['overall_score']}")
    print(f"Integrity Level: {report['integrity_level']}")
    print(f"\nRisk Assessment: {report['risk_assessment']['risk_level']}")
    print(f"Recommendation: {report['risk_assessment']['recommendation']}")
    
    print("\nComponent Scores:")
    for component, score in report['component_scores'].items():
        print(f"  {component}: {score:.2f}")
    
    print(f"\nTotal Violations: {report['violation_summary']['total_violations']}")
    
    print("\nViolations by Severity:")
    for severity, count in report['violation_summary']['by_severity'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    
    print("\nRecommendations:")
    for rec in scorer.generate_recommendations():
        print(f"  [{rec['severity'].upper()}] {rec['recommendation']}")
    
    # Export to JSON
    json_report = scorer.export_to_json()
    print("\n" + "=" * 60)
    print("JSON Report (truncated):")
    print(json_report[:500] + "...")
