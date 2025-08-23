import time
import cv2
from ultralytics import YOLO
from typing import List, Tuple
from config import YOLO_MODEL_PATH, YOLO_FURNITURE_CLASSES, YOLO_CONF_THRESHOLD

# ROI와 사람 바운딩 박스를 동일 형식(좌상단/우하단 좌표)으로 다루기 위한 타입 힌트
BBox = Tuple[int, int, int, int]

class ROIManager:
    """
    침대(bed), 의자(chair) 등 '관심 영역(ROI)'을 자동 검출·관리하는 클래스.
    - 일정 주기(update_interval)마다 프레임에서 가구 클래스만 YOLO로 검출해 self.rois 갱신
    - 검출 실패 시 self.rois는 빈 리스트로 유지 → 그리기/포함 판정 시 스킵
    """

    def __init__(self, update_interval: float = 10.0):
        """
        :param update_interval: ROI 자동 갱신 주기(초)
        """
        # 현재 유효한 ROI 목록(각 항목은 (x1, y1, x2, y2) 튜플)
        self.rois: List[BBox] = []
        self.update_interval = update_interval
        self._last_update = 0.0

        # COCO 사전학습 YOLO 가중치 로드
        self.model = YOLO(YOLO_MODEL_PATH)

    def get_rois(self) -> List[BBox]:
        """
        외부에서 현재 ROI 목록을 조회할 때 사용.
        """
        return self.rois

    def update_roi(self, roi: BBox) -> None:
        """
        수동으로 ROI 하나를 강제 설정.
        :param roi: (x1, y1, x2, y2)
        """
        self.rois = [roi]

    def auto_update(self, frame) -> None:
        """
        update_interval 주기마다 프레임에서 '가구 클래스'만 검출해 ROI를 갱신.
        :param frame: BGR 이미지(ndarray)
        """
        now = time.time()
        if now - self._last_update < self.update_interval:
            # 아직 갱신 주기가 안 됐으면 패스
            return
        self._last_update = now

        # classes 옵션으로 침대/의자 등 필요한 가구 클래스만 필터링
        results = self.model(frame, classes=YOLO_FURNITURE_CLASSES, verbose=False)[0]
        detected: List[BBox] = []

        # 결과에서 신뢰도(conf)가 임계치 이상인 박스만 ROI로 채택
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if float(conf) < YOLO_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detected.append((x1, y1, x2, y2))

        # 검출된 가구가 있으면 갱신, 없으면 빈 리스트로 유지
        self.rois = detected

    def is_bbox_in_roi(self, bbox: BBox) -> bool:
        """
        사람 바운딩 박스가 현재 ROI들 중 하나에 '포함'되는지 판단.
        포함 기준: 사람 박스의 중심점이 ROI 사각형 내부에 있으면 True.
        :param bbox: (x1, y1, x2, y2)
        :return: bool
        """
        if not self.rois:
            return False

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for rx1, ry1, rx2, ry2 in self.rois:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    def draw(self, frame, color=(0, 0, 255), thickness=2):
        """
        현재 보유한 모든 ROI를 프레임 위에 시각화.
        :param frame: BGR 이미지
        :param color: 사각형 색(B,G,R)
        :param thickness: 선 두께
        :return: 그려진 프레임
        """
        if not self.rois:
            return frame

        for idx, (x1, y1, x2, y2) in enumerate(self.rois):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                f"ROI_{idx}",
                (x1, max(0, y1 - 10)),  # 텍스트가 프레임 바깥으로 나가지 않도록 보호
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        return frame
