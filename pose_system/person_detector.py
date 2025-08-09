# person_detector.py
# 사람 인식해서 yolo로 바운딩 박스 만들어주는 파일

from ultralytics import YOLO
import cv2
from config import YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD

class PersonDetector:
    """
    YOLOv8 모델을 로드하고, 사람 탐지를 위한 설정을 초기화합니다.
    """
    def __init__(self,
                 model_path: str = YOLO_MODEL_PATH,
                 conf_threshold: float = YOLO_CONF_THRESHOLD):
        """
        :param model_path: YOLOv8 가중치 파일 경로
        :param conf_threshold: 탐지 신뢰도 임계값
        """
        # YOLOv8 모델 로드
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        프레임에서 사람 바운딩 박스를 검출합니다.
        :param frame: BGR 이미지 (numpy.ndarray)
        :return: 사람 클래스의 바운딩 박스 리스트 [(x1, y1, x2, y2), ...]
        """
        # ultralytics YOLOv8은 BGR/RGB 자동 처리
        # 성능 로그 출력
        #results = self.model(frame)
        # 성능 로그 출력 X
        results = self.model(frame, verbose=False)

        boxes = []
        # results 객체는 리스트 형태로 반환됨
        for r in results:
            # r.boxes.xyxy: tensor of shape [N,4]
            # r.boxes.conf: tensor of shape [N]
            # r.boxes.cls: tensor of shape [N]
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(xyxy, confs, clss):
                # COCO person 클래스는 0번
                if int(cls) == 0 and conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))
        return boxes
