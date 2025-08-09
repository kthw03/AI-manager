# 자세 분류
# posture_classifier_v6.py

from utils import calculate_angle, distance
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])

class PostureClassifierV6:
    def __init__(self):
        # Kneeling thresholds
        self.KNEE_KNEEL_MIN = 80
        self.KNEE_KNEEL_MAX = 130

        # Sitting thresholds
        self.TORSO_SIT_MIN = 100
        self.TORSO_SIT_MAX = 150
        
        # Lying thresholds
        self.Y_RANGE_LYING     = 0.05   # 세로 평탄도 임계
        self.X_RANGE_LYING     = 0.30   # 가로 퍼짐 임계 (튜닝 요망)
        self.Z_PRONE_THRESHOLD = 0.0    # supine/prone 구분용
        
        self.TORSO_KNEEL_MIN = 120
        self.TORSO_KNEEL_MAX = 160

        

    def get_angles(self, landmarks):
        return {
            'leg': (calculate_angle(*[Point(*landmarks[i][:2]) for i in [23, 25, 27]]) +
                    calculate_angle(*[Point(*landmarks[i][:2]) for i in [24, 26, 28]])) / 2,
            'torso': (calculate_angle(*[Point(*landmarks[i][:2]) for i in [11, 23, 25]]) +
                      calculate_angle(*[Point(*landmarks[i][:2]) for i in [12, 24, 26]])) / 2
        }

    def get_y_values(self, landmarks):
        return {
            'nose': landmarks[0][1],
            'shoulder_avg': (landmarks[11][1] + landmarks[12][1]) / 2,
            'hip_avg': (landmarks[23][1] + landmarks[24][1]) / 2,
            'knee_avg': (landmarks[25][1] + landmarks[26][1]) / 2,
            'ankle_avg': (landmarks[27][1] + landmarks[28][1]) / 2
        }

    def get_segment_lengths(self, landmarks):
        sh = Point(*landmarks[11][:2])
        hip = Point(*landmarks[23][:2])
        knee = Point(*landmarks[25][:2])
        return {
            'shoulder_hip': distance(sh, hip),
            'hip_knee': distance(hip, knee)
        }

    def is_lying(self, landmarks):
        # --- 1) X축 퍼짐 검사 (가로 누움) ---
        x_idxs = [11,12,23,24,25,26,27,28]
        xs = [landmarks[i][0] for i in x_idxs if landmarks[i][3] > 0.5]
        x_range = max(xs) - min(xs) if xs else 0.0
        if x_range > self.X_RANGE_LYING:
            # z로 prone/supine 구분
            nose_z = landmarks[0][2]
            hip_z  = (landmarks[23][2] + landmarks[24][2]) / 2
            return "lying_prone" if (nose_z - hip_z) < self.Z_PRONE_THRESHOLD else "lying_supine"
        
        # --- 2) 기존 Y축 평탄도 검사 (세로 누움) ---
        y_vals = [lm[1] for lm in landmarks[11:29] if lm[3] > 0.5]
        if len(y_vals) < 5 or (max(y_vals) - min(y_vals)) > self.Y_RANGE_LYING:
            return None

        nose_z = landmarks[0][2]
        hip_z = (landmarks[23][2] + landmarks[24][2]) / 2
        nose_y = landmarks[0][1]
        ankle_y = (landmarks[27][1] + landmarks[28][1]) / 2

        if abs(nose_y - ankle_y) > 0.15:
            return None

        return "lying_prone" if (nose_z - hip_z) < self.Z_PRONE_THRESHOLD else "lying_supine"

    def is_kneeling(self, landmarks, angles, y_vals):
        if not (self.KNEE_KNEEL_MIN <= angles['leg'] <= self.KNEE_KNEEL_MAX):
            return False
        if not (self.TORSO_KNEEL_MIN <= angles['torso'] <= self.TORSO_KNEEL_MAX):
            return False
        if y_vals['hip_avg'] <= y_vals['knee_avg']:
            return False
        if abs(y_vals['ankle_avg'] - y_vals['knee_avg']) > 0.1:
            return False
        return True

    def is_sitting(self, landmarks, angles, y_vals, segs):
        def is_front_view(landmarks):
            vis_l = landmarks[11][3]
            vis_r = landmarks[12][3]
            return vis_l > 0.6 and vis_r > 0.6
        front_view = is_front_view(landmarks)

        torso_alt_l = calculate_angle(
            Point(*landmarks[11][:2]), Point(*landmarks[23][:2]), Point(*landmarks[27][:2])
        )
        torso_alt_r = calculate_angle(
            Point(*landmarks[12][:2]), Point(*landmarks[24][:2]), Point(*landmarks[28][:2])
        )
        torso_alt_avg = (torso_alt_l + torso_alt_r) / 2

        dy_sh_hip = y_vals['hip_avg'] - y_vals['shoulder_avg']
        dy_hip_knee = y_vals['knee_avg'] - y_vals['hip_avg']
        dy_ratio = dy_sh_hip / (dy_hip_knee + 1e-6)

        if dy_ratio < 1.6:
            return False

        y_cond = (
            y_vals['shoulder_avg'] < y_vals['hip_avg'] < y_vals['knee_avg']
            or abs(y_vals['hip_avg'] - y_vals['knee_avg']) < 0.05
        )
        if not y_cond:
            return False

        return True

    def is_standing(self, landmarks, angles, y_vals, segs):
        # ✅ 개선된 완화 조건 반영
        if angles['leg'] < 155:
            return False
        if angles['torso'] < 150:
            return False
        if y_vals['hip_avg'] >= y_vals['knee_avg'] + 0.02:
            return False
        if y_vals['nose'] >= y_vals['hip_avg']:
            return False
        return True

    def classify(self, landmarks):
        angles = self.get_angles(landmarks)
        y_vals = self.get_y_values(landmarks)
        segs = self.get_segment_lengths(landmarks)

        if self.is_sitting(landmarks, angles, y_vals, segs):
            return "sitting"
        if (lying := self.is_lying(landmarks)):
            return lying
        if self.is_kneeling(landmarks, angles, y_vals):
            return "kneeling"
        if self.is_standing(landmarks, angles, y_vals, segs):
            return "standing"
        return "irregular"
