# pose_extractor.py

import cv2
import numpy as np
import mediapipe as mp
from config import MP_DETECT_CONFIDENCE, MP_TRACK_CONFIDENCE


def pad_to_square(img, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    side = max(h, w)
    dh, dw = side - h, side - w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=pad_color)


def normalize_z_roi(landmarks, bbox):
    """
    ROI 높이 기준으로 z값 정규화
    """
    x1, y1, x2, y2 = bbox
    roi_height = y2 - y1 if (y2 - y1) != 0 else 1
    return [(x, y, z / roi_height, v) for (x, y, z, v) in landmarks]


def normalize_z_relative(landmarks):
    """
    어깨 평균 z를 기준으로 상대값 변환
    """
    z_ref = (landmarks[11][2] + landmarks[12][2]) / 2
    return [(x, y, z - z_ref, v) for (x, y, z, v) in landmarks]


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=MP_DETECT_CONFIDENCE,
            min_tracking_confidence=MP_TRACK_CONFIDENCE
        )

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # 1) 크롭된 ROI를 정사각형으로 패딩
        square = pad_to_square(roi)

        # 2) BGR→RGB 변환 후 MediaPipe에 입력
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return None

        # 3) 랜드마크 리스트로 변환 (원본 z값 포함)
        landmarks = [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in results.pose_landmarks.landmark
        ]

        # 4) z값 정규화 병행 적용
        landmarks = normalize_z_roi(landmarks, bbox)
        landmarks = normalize_z_relative(landmarks)

        # 5) 결과 반환
        return {
            "bbox": bbox,
            "landmarks": landmarks,
            "pose_landmarks": results.pose_landmarks  # 시각화용
        }

    def close(self):
        self.pose.close()
