import time
import cv2
from ultralytics import YOLO
from typing import List, Tuple, Optional
from config import YOLO_MODEL_PATH, YOLO_FURNITURE_CLASSES, YOLO_CONF_THRESHOLD

# (x1, y1, x2, y2) 형태의 바운딩 박스 타입
BBox = Tuple[int, int, int, int]

class ROIManager:
    def __init__(self, update_interval: float = 10.0):
        # YOLO로 자동 검출된 ROI 목록
        self.auto_rois: List[BBox] = []
        # 사용자가 드래그로 추가한 수동 ROI 목록
        self.manual_rois: List[BBox] = []
        # 자동 갱신 주기(초)
        self.update_interval = update_interval
        self._last_update = 0.0
        # YOLO 모델
        self.model = YOLO(YOLO_MODEL_PATH)
        # 수동 편집 모드 on/off
        self.edit_enabled = False
        # 드래그 시작점/현재점(수동 ROI 그릴 때 사용)
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_current: Optional[Tuple[int, int]] = None

    def get_rois(self) -> List[BBox]:
        # 자동 ROI + 수동 ROI를 합쳐서 반환
        return [*self.auto_rois, *self.manual_rois]

    def update_roi(self, roi: BBox) -> None:
        # 수동 ROI를 하나만 강제로 설정(기존 수동 ROI는 덮어씀)
        x1, y1, x2, y2 = roi
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        self.manual_rois = [(x1, y1, x2, y2)]

    def add_manual_roi(self, roi: BBox) -> None:
        # 수동 ROI를 추가(기존 목록 유지)
        x1, y1, x2, y2 = roi
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        self.manual_rois.append((x1, y1, x2, y2))

    def clear_manual_rois(self) -> None:
        # 수동 ROI 전부 삭제
        self.manual_rois = []

    def set_edit_enabled(self, enabled: bool) -> None:
        # 편집 모드 on/off
        self.edit_enabled = bool(enabled)
        self._drag_start = None
        self._drag_current = None

    def toggle_edit_mode(self) -> None:
        # 편집 모드 토글
        self.set_edit_enabled(not self.edit_enabled)

    def handle_mouse(self, event, x, y, flags, param) -> None:
        # 마우스 드래그로 수동 ROI 추가
        if not self.edit_enabled:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # 드래그 시작
            self._drag_start = (x, y)
            self._drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            # 드래그 중 위치 갱신
            self._drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            # 드래그 종료 → 박스 확정
            x1, y1 = self._drag_start
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 > x1 and y2 > y1:
                self.manual_rois.append((x1, y1, x2, y2))
            self._drag_start = None
            self._drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 우클릭: 드래그 취소 또는 마지막 수동 ROI 제거(되돌리기)
            if self._drag_start is not None:
                self._drag_start = None
                self._drag_current = None
            elif self.manual_rois:
                self.manual_rois.pop()

    def auto_update(self, frame) -> None:
        # 일정 주기마다 YOLO로 침대/의자 클래스만 검출해 자동 ROI 갱신
        now = time.time()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now

        # classes로 원하는 가구 클래스만 추론
        results = self.model(frame, classes=YOLO_FURNITURE_CLASSES, verbose=False)[0]
        detected: List[BBox] = []
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if float(conf) < YOLO_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detected.append((x1, y1, x2, y2))
        self.auto_rois = detected

    def is_bbox_in_roi(self, bbox: BBox) -> bool:
        # 사람 박스의 중심점이 자동/수동 ROI 중 하나라도 안에 있으면 True
        rois = self.get_rois()
        if not rois:
            return False
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        for rx1, ry1, rx2, ry2 in rois:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    def draw(self, frame, color_auto=(0, 0, 255), color_manual=(0, 255, 0), thickness=2):
        # 자동 ROI는 빨간색, 수동 ROI는 초록색으로 그림
        for (x1, y1, x2, y2) in self.auto_rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_auto, thickness)
            cv2.putText(frame, "AUTO", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_auto, 2)
        for (x1, y1, x2, y2) in self.manual_rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_manual, thickness)
            cv2.putText(frame, "MANUAL", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_manual, 2)

        # 편집 모드 중 드래그 프리뷰(노란색 가이드 박스)
        if self.edit_enabled and self._drag_start is not None and self._drag_current is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, "DRAWING", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        return frame
