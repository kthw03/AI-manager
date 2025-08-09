# config.py
# 각종 설정 값 존재하는 파일

# 카메라 설정
FRAME_WIDTH = 640  # 캡처할 프레임 너비
FRAME_HEIGHT = 480  # 캡처할 프레임 높이

# YOLO 설정
YOLO_MODEL_PATH = "yolov8n"  # YOLO 가중치 파일 경로
YOLO_FURNITURE_CLASSES = [56, 59]  # COCO 기준 chair=56, bed=59
YOLO_CONF_THRESHOLD = 0.5  # 탐지 신뢰도 임계값
YOLO_IOU_THRESHOLD = 0.4  # NMS(IOU) 임계값

# MediaPipe Pose 설정
MP_DETECT_CONFIDENCE = 0.5  # Pose 탐지 최소 신뢰도
MP_TRACK_CONFIDENCE = 0.5   # 랜드마크 추적 최소 신뢰도

# 자세 분류 임계값 (랜드마크 상대 위치 기준)
SHOULDER_HIP_DIFF_THRESHOLD = 0.1  # 어깨-엉덩이 높이 차이 (누움 판단)
HIP_KNEE_DIFF_THRESHOLD = 0.15     # 엉덩이-무릎 높이 차이 (앉음 판단)
SHOULDER_LEVEL_DIFF_THRESHOLD = 0.05  # 좌우 어깨 높이 차이 (기울어짐 판단)
# posture_classifier.py 개선을 위한 추가
HIP_KNEE_ANGLE_THRESHOLD = 160  # standing: 엉덩이-무릎-발 각도가 거의 180도
SHOULDER_HIP_ANGLE_THRESHOLD = 150  # 상체 수직 기준
POSE_Z_PRONE_THRESHOLD = -0.3  # prone: z값 평균이 낮음


# 행동/이벤트 임계값
NO_MOVEMENT_TIME_THRESHOLD = 30.0  # 움직임 없음 경고 시간(초)
FALL_TRANSITION_TIME = 2.0         # 낙상 전이 최대 허용 시간(초)
TILT_DURATION = 10.0               # 기울어진 자세 유지시간(초)

# 관심 영역(ROI) 기본값 (x1, y1, x2, y2)
DEFAULT_ROI = (100, 200, 500, 600)

# 로깅 설정
LOG_CSV_PATH = "logs/posture_log.csv"  # 자세/이벤트 로그 CSV 파일 경로
LOG_CONSOLE = True                     # 콘솔 출력 여부

# 알림 관리자 설정
ALERT_METHODS = ["console", "firebase", "email"]

# Firebase 연동 설정 (예시)
FIREBASE_CONFIG = {
    "apiKey": "<YOUR_API_KEY>",
    "authDomain": "<YOUR_AUTH_DOMAIN>",
    "databaseURL": "<YOUR_DATABASE_URL>",
    "storageBucket": "<YOUR_STORAGE_BUCKET>"
}

# 이메일 알림 설정
EMAIL_RECIPIENT = "caregiver@example.com"

# --- 파일 맨 아래에 추가 ---
# PostureAnalyzerV3 전용 임계값
ANALYZER_TILT_THRESHOLD       = 0.1   # 어깨 y 변동 임계값 (px 단위)
ANALYZER_MOTION_THRESHOLD     = 5.0   # 무동작 임계값 (평균 이동량, px 단위)
ANALYZER_IRREGULAR_THRESHOLD  = 3     # 윈도우 내 posture 전이 횟수 임계값
# POSE_Z_PRONE_THRESHOLD       = -0.3  # 이미 정의돼 있음
# FALL_TRANSITION_TIME         = 2.0   # 이미 정의돼 있음
# NO_MOVEMENT_TIME_THRESHOLD   = 30.0  # 이미 정의돼 있음
# TILT_DURATION                = 10.0  # 이미 정의돼 있음
