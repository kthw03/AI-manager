import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Union
from config import YOLO_MODEL_PATH, YOLO_FURNITURE_CLASSES, YOLO_CONF_THRESHOLD

# (x1, y1, x2, y2) 형태의 바운딩 박스 타입
BBox = Tuple[int, int, int, int]

class ROIManager:
    def __init__(self, update_interval: float = 10.0):
        # YOLO 자동 ROI(사각형 bbox)
        self.auto_rois: List[BBox] = []
        # 수동 ROI(사각형 bbox) + 수동 폴리곤(4점) — 폴리곤은 시각화/정확 판정용
        self.manual_rois: List[BBox] = []
        self.manual_polygons: List[np.ndarray] = []

        # 자동 갱신 주기
        self.update_interval = update_interval
        self._last_update = 0.0

        # YOLO 모델
        self.model = YOLO(YOLO_MODEL_PATH)

        # 편집 모드(ON일 때만 클릭 반응)
        self.edit_enabled = False
        # 진행 중 점(좌클릭으로 누적, 4점 시 확정). 순서 무관
        self._click_points: List[Tuple[int, int]] = []

    # ---------------- 공용 API ---------------- #
    def get_rois(self) -> List[BBox]:
        # 자동 ROI + 수동 ROI를 합쳐 반환
        return [*self.auto_rois, *self.manual_rois]

    def update_roi(self, roi: Union[BBox, List[Tuple[int, int]]]) -> None:
        """
        수동 ROI를 외부에서 강제 설정.
        - (x1,y1,x2,y2) 튜플 또는 임의 4점 리스트를 허용
        """
        if isinstance(roi, tuple) and len(roi) == 4:
            x1, y1, x2, y2 = roi
            x1, x2 = sorted((int(x1), int(x2)))
            y1, y2 = sorted((int(y1), int(y2)))
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        elif isinstance(roi, list) and len(roi) >= 4:
            pts = np.array(roi[:4], dtype=np.float32)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            order = np.argsort(ang)
            poly = pts[order].astype(np.int32)
        else:
            raise ValueError("roi must be (x1,y1,x2,y2) or list of 4 points")

        x, y, w, h = cv2.boundingRect(poly)
        self.manual_polygons = [poly]
        self.manual_rois = [(x, y, x + w, y + h)]
        self._click_points = []

    def add_manual_roi(self, roi: BBox) -> None:
        # 수동 ROI 추가(사각형 튜플)
        x1, y1, x2, y2 = roi
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        self.manual_rois.append((x1, y1, x2, y2))
        self.manual_polygons.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))

    def clear_manual_rois(self) -> None:
        # 수동 ROI/폴리곤 전체 삭제
        self.manual_rois = []
        self.manual_polygons = []
        self._click_points = []

    def set_edit_enabled(self, enabled: bool) -> None:
        # 편집 모드 on/off
        self.edit_enabled = bool(enabled)
        self._click_points = []

    def toggle_edit_mode(self) -> None:
        # 편집 모드 토글  ← 여기서 'not' 사용 (파이썬은 !가 아님)
        self.edit_enabled = not self.edit_enabled
        self._click_points = []

    # ---------------- 마우스 콜백(좌클릭 4점 → 임의 사각형 확정) ---------------- #
    def handle_mouse(self, event, x, y, flags, param) -> None:
        """
        편집 모드에서만 동작:
        - 좌클릭: 점 추가(순서 무관, 4점 도달 시 중심각 정렬로 사다리꼴/임의 사각형 폴리곤 확정)
        - 우클릭: 진행 중이면 마지막 점 삭제, 아니면 마지막 확정 ROI 삭제
        """
        if not self.edit_enabled:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_points.append((int(x), int(y)))
            if len(self._click_points) == 4:
                pts = np.array(self._click_points, dtype=np.float32)
                cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
                order = np.argsort(ang)
                poly = pts[order].astype(np.int32)               # 시계/반시계 정렬된 4점 폴리곤
                self.manual_polygons.append(poly)
                x, y, w, h = cv2.boundingRect(poly)
                self.manual_rois.append((x, y, x + w, y + h))    # bbox는 빠른 필터/표시용
                self._click_points = []

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._click_points:
                self._click_points.pop()
            elif self.manual_rois:
                self.manual_rois.pop()
                if self.manual_polygons:
                    self.manual_polygons.pop()

    # ---------------- 자동 ROI(YOLO) ---------------- #
    def auto_update(self, frame) -> None:
        # 일정 주기마다 YOLO로 가구 클래스만 검출해 자동 ROI 갱신
        now = time.time()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now

        results = self.model(frame, classes=YOLO_FURNITURE_CLASSES, verbose=False)[0]
        detected: List[BBox] = []
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if float(conf) < YOLO_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detected.append((x1, y1, x2, y2))
        self.auto_rois = detected

    # ---------------- 포함 판정 ---------------- #
    def is_bbox_in_roi(self, bbox: BBox) -> bool:
        # 사람 박스 중심점이 수동 폴리곤 내부 또는 자동/수동 사각형 내부에 있으면 True
        rois = self.get_rois()
        if not rois and not self.manual_polygons:
            return False

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for poly in self.manual_polygons:
            if poly is None or len(poly) < 3:
                continue
            res = cv2.pointPolygonTest(poly.reshape((-1, 1, 2)), (float(cx), float(cy)), False)
            if res >= 0:
                return True

        for rx1, ry1, rx2, ry2 in rois:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    # ---------------- 시각화 ---------------- #
    def draw(self, frame, color_auto=(0, 0, 255), color_manual=(0, 255, 0), thickness=2):
        # 자동 ROI: 빨간 사각형
        for (x1, y1, x2, y2) in self.auto_rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_auto, thickness)
            cv2.putText(frame, "AUTO", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_auto, 2)

        # 수동 폴리곤: 초록 폴리곤(사다리꼴/임의 사각형) + 라벨
        for idx, poly in enumerate(self.manual_polygons):
            if poly is None or len(poly) < 3:
                continue
            cv2.polylines(frame, [poly], isClosed=True, color=color_manual, thickness=thickness)
            x0, y0, w, h = cv2.boundingRect(poly)
            cv2.putText(frame, f"MANUAL_{idx}", (x0, max(0, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_manual, 2)

        # 편집 모드 프리뷰(노란 점/선 + 임시 폴리곤)
        if self.edit_enabled and self._click_points:
            for p in self._click_points:
                cv2.circle(frame, p, 5, (255, 255, 0), -1)
            for i in range(1, len(self._click_points)):
                cv2.line(frame, self._click_points[i-1], self._click_points[i], (255, 255, 0), 1)
            if len(self._click_points) >= 3:
                pts = np.array(self._click_points, dtype=np.float32)
                cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
                ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
                order = np.argsort(ang)
                poly_prev = pts[order].astype(np.int32)
                cv2.polylines(frame, [poly_prev], isClosed=False, color=(255, 255, 0), thickness=1)
            cv2.putText(frame, f"POINTS: {len(self._click_points)}/4",
                        (self._click_points[-1][0] + 6, self._click_points[-1][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return frame
