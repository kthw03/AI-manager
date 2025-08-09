# utils.py
# 좌표계산 함수 등 각종 함수 설정된 파일

import time
import math
from config import FRAME_WIDTH, FRAME_HEIGHT


def get_timestamp():
    """
    현재 시스템 시간에 기반한 타임스탬프를 문자열로 반환합니다.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def calculate_euclidean_distance(point1, point2):
    """
    두 랜드마크(point1, point2)의 유클리디안 거리를 픽셀 단위로 계산합니다.
    point.x, point.y는 0~1 사이의 정규화된 좌표라고 가정합니다.
    """
    dx = (point1.x - point2.x) * FRAME_WIDTH
    dy = (point1.y - point2.y) * FRAME_HEIGHT
    return math.hypot(dx, dy)


def calculate_angle(a, b, c):
    """
    세 점 a, b, c가 주어졌을 때, 각 ABC의 각도를 계산해 반환합니다.
    a, b, c에는 x, y 속성이 있어야 하며, 반환값은 도(degree) 단위입니다.
    """
    # 벡터 BA, BC 계산
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    # 내적과 크기 계산
    dot_product = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)
    if mag_ba * mag_bc == 0:
        return 0.0

    # 각도 계산
    cos_angle = max(min(dot_product / (mag_ba * mag_bc), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def is_point_in_roi(landmark, roi):
    """
    랜드마크의 좌표가 주어진 ROI(픽셀 기준)의 내부에 있는지 여부를 반환합니다.
    roi는 (x1, y1, x2, y2) 형태의 박스입니다.
    landmark.x, landmark.y는 0~1 정규화 좌표.
    """
    x1, y1, x2, y2 = roi
    px = landmark.x * FRAME_WIDTH
    py = landmark.y * FRAME_HEIGHT
    return x1 <= px <= x2 and y1 <= py <= y2


def has_significant_movement(prev_landmarks, curr_landmarks, threshold):
    """
    이전 프레임(prev_landmarks)과 현재 프레임(curr_landmarks)의 랜드마크 변화량 평균이
    threshold(픽셀 단위) 이상인지를 판단합니다.
    """
    if not prev_landmarks or not curr_landmarks:
        return False

    total_dist = 0.0
    count = min(len(prev_landmarks), len(curr_landmarks))
    for i in range(count):
        total_dist += calculate_euclidean_distance(prev_landmarks[i], curr_landmarks[i])

    avg_dist = total_dist / count
    return avg_dist > threshold


def distance(p1, p2):
    """
    두 Point(x, y) 사이의 2D 거리 계산.
    posture_classifier_v6에서 segment 길이 비교에 사용됨.
    """
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
