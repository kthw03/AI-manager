# 자세 분류
# posture_classifier_v6.py

# 자세 분류
# posture_classifier_v6_min.py

from utils import calculate_angle  # distance 제거
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])

class PostureClassifierV6:
    def __init__(self):
        # Kneeling thresholds
        self.KNEE_KNEEL_MIN = 80
        self.KNEE_KNEEL_MAX = 130

        # Lying thresholds
        self.Y_RANGE_LYING     = 0.05   # 세로 평탄도 임계
        self.X_RANGE_LYING     = 0.30   # 가로 퍼짐 임계 (튜닝 요망)
        self.Z_PRONE_THRESHOLD = 0.0    # supine/prone 구분용

        # Kneeling torso range
        self.TORSO_KNEEL_MIN = 120
        self.TORSO_KNEEL_MAX = 160

    def get_angles(self, landmarks):
        leg_l = calculate_angle(
            *[Point(*landmarks[i][:2]) for i in [23, 25, 27]]
        )
        leg_r = calculate_angle(
            *[Point(*landmarks[i][:2]) for i in [24, 26, 28]]
        )
        torso_l = calculate_angle(
            *[Point(*landmarks[i][:2]) for i in [11, 23, 25]]
        )
        torso_r = calculate_angle(
            *[Point(*landmarks[i][:2]) for i in [12, 24, 26]]
        )
        return {
            "leg":   (leg_l + leg_r) / 2.0,
            "torso": (torso_l + torso_r) / 2.0,
        }

    def get_y_values(self, landmarks):
        return {
            "nose": landmarks[0][1],
            "shoulder_avg": (landmarks[11][1] + landmarks[12][1]) / 2.0,
            "hip_avg":      (landmarks[23][1] + landmarks[24][1]) / 2.0,
            "knee_avg":     (landmarks[25][1] + landmarks[26][1]) / 2.0,
            "ankle_avg":    (landmarks[27][1] + landmarks[28][1]) / 2.0,
        }

    def is_lying(self, landmarks):
        # Precompute z once (used in both branches)
        nose_z = landmarks[0][2]
        hip_z  = (landmarks[23][2] + landmarks[24][2]) / 2.0

        # --- 1) X축 퍼짐 검사 (가로 누움) ---
        x_idxs = [11, 12, 23, 24, 25, 26, 27, 28]
        xs = [landmarks[i][0] for i in x_idxs if landmarks[i][3] > 0.5]
        if xs and (max(xs) - min(xs) > self.X_RANGE_LYING):
            # z로 prone/supine 구분
            return "lying_prone" if (nose_z - hip_z) < self.Z_PRONE_THRESHOLD else "lying_supine"

        # --- 2) 기존 Y축 평탄도 검사 (세로 누움) ---
        y_all = [lm[1] for lm in landmarks[11:29] if lm[3] > 0.5]
        if len(y_all) < 5 or (max(y_all) - min(y_all)) > self.Y_RANGE_LYING:
            return None

        nose_y  = landmarks[0][1]
        ankle_y = (landmarks[27][1] + landmarks[28][1]) / 2.0
        if abs(nose_y - ankle_y) > 0.15:
            return None

        return "lying_prone" if (nose_z - hip_z) < self.Z_PRONE_THRESHOLD else "lying_supine"


    def is_sitting(self, landmarks, angles, y_vals):
        # Core sitting logic only
        dy_sh_hip  = y_vals["hip_avg"]  - y_vals["shoulder_avg"]
        dy_hip_knee= y_vals["knee_avg"] - y_vals["hip_avg"]
        dy_ratio   = dy_sh_hip / (dy_hip_knee + 1e-6)
        if dy_ratio < 1.6:
            return False

        y_cond = (
            y_vals["shoulder_avg"] < y_vals["hip_avg"] < y_vals["knee_avg"]
            or abs(y_vals["hip_avg"] - y_vals["knee_avg"]) < 0.05
        )
        if not y_cond:
            return False
        return True

    def is_standing(self, landmarks, angles, y_vals):
        if angles["leg"] < 155:
            return False
        if angles["torso"] < 150:
            return False
        if y_vals["hip_avg"] >= y_vals["knee_avg"] + 0.02:
            return False
        if y_vals["nose"] >= y_vals["hip_avg"]:
            return False
        return True

    def classify(self, landmarks):
        angles = self.get_angles(landmarks)
        y_vals = self.get_y_values(landmarks)
        '''if self.is_kneeling(landmarks, angles, y_vals):
            return "kneeling"'''
        if self.is_sitting(landmarks, angles, y_vals):
            return "sitting"
        lying = self.is_lying(landmarks)
        if lying:
            return lying
        if self.is_standing(landmarks, angles, y_vals):
            return "standing"
        return "irregular"

