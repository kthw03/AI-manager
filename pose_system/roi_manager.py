# roi_manager.py

import time
import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_FURNITURE_CLASSES, YOLO_CONF_THRESHOLD  # 설정 값 불러오기 :contentReference[oaicite:0]{index=0}

class ROIManager:
    """
    침대(bed), 의자(chair) 등 관심 영역(ROI)을 자동 검출·관리하는 모듈입니다.
    - update_interval 초마다 YOLO로 가구만 검출하여 self.rois 갱신
    - 가구 미검출 시 빈 리스트로 유지 → draw()/is_bbox_in_roi() 모두 스킵
    """

    def __init__(self, update_interval: float = 10.0):
        """
        :param update_interval: ROI 자동 갱신 주기(초)
        """
        # 자동 검출 전까지는 ROI가 없는 상태
        self.rois = []  
        self.update_interval = update_interval
        self._last_update = 0.0

        # COCO 사전학습된 YOLO 모델 로드
        self.model = YOLO(YOLO_MODEL_PATH)

    def get_rois(self):
        """
        현재 검출된 ROI 리스트를 반환합니다.
        :return: List[Tuple[x1, y1, x2, y2]]
        """
        return self.rois

    def update_roi(self, roi):
        """
        수동으로 ROI 하나를 설정할 때 사용합니다.
        :param roi: (x1, y1, x2, y2)
        """
        self.rois = [roi]

    def auto_update(self, frame):
        """
        update_interval 주기마다 frame에서 'bed'와 'chair' 클래스만 검출해 ROI를 갱신합니다.
        :param frame: BGR 이미지 (np.ndarray)
        """
        now = time.time()
        if now - self._last_update < self.update_interval:
            return

        self._last_update = now

        # verbose=False로 로그 억제, classes 옵션으로 침대·의자만 필터링
        results = self.model(frame, classes=YOLO_FURNITURE_CLASSES, verbose=False)[0]

        detected = []
        # results.boxes.xyxy: [N,4], results.boxes.conf: [N]
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if float(conf) < YOLO_CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detected.append((x1, y1, x2, y2))

        # 검출된 가구가 있으면 갱신, 없으면 빈 리스트 유지
        self.rois = detected

    def is_bbox_in_roi(self, bbox):
        """
        주어진 사람 바운딩 박스가 어떤 ROI(가구 영역) 내부에 있는지 판정합니다.
        :param bbox: (x1, y1, x2, y2)
        :return: bool (하나라도 포함되면 True)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        for rx1, ry1, rx2, ry2 in self.rois:
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                return True
        return False

    def draw(self, frame, color=(0, 0, 255), thickness=2):
        """
        frame 위에 현재 self.rois에 저장된 모든 ROI를 그려줍니다.
        :param frame: BGR 이미지 (np.ndarray)
        :param color: 사각형 색상 (B, G, R)
        :param thickness: 선 두께
        :return: 그려진 이미지 (np.ndarray)
        """
        if not self.rois:
            return frame

        for idx, (x1, y1, x2, y2) in enumerate(self.rois):
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame,
                        f"ROI_{idx}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2)
        return frame
