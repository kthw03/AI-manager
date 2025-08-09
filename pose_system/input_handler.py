# input_handler.py
# -*- coding: utf-8 -*-
# 각종 영상 정보 소스를 통일된 인터페이스로 뽑아주는 모듈

import cv2
from picamera2 import Picamera2
from config import FRAME_WIDTH, FRAME_HEIGHT

class InputHandler:
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        영상 소스 초기화: Raspberry Pi CSI 카메라 (Picamera2)
        :param width: 캡처 해상도 너비
        :param height: 캡처 해상도 높이
        """
        self.picam2 = Picamera2()
        preview_config = self.picam2.create_preview_configuration({
            "size": (width, height),
            "format": "RGB888"
        })
        self.picam2.configure(preview_config)
        self.picam2.start()

    def is_opened(self):
        """
        Picamera2 연결 여부 확인 (항상 True 반환)
        """
        return True

    def get_frame(self):
        """
        한 프레임을 읽어서 BGR 이미지로 반환합니다.
        :return: BGR 이미지(np.ndarray)
        """
        rgb = self.picam2.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def release(self):
        """
        캡처 리소스를 해제합니다.
        """
        self.picam2.stop()


'''
# input_handler.py
# 각종 영상 정보 소스를 통일된 인터페이스로 뽑아주는 모듈

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

class InputHandler:
    def __init__(self, source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        영상 소스 초기화
        :param source: int(웹캠 인덱스) 또는 str(동영상 파일 경로)
        """
        self.cap = cv2.VideoCapture(source)
        # 카메라가 안 열리면 경고
        if not self.cap.isOpened():
            print(f"⚠️ 카메라 열기 실패 (source={source})")
        # 캡처 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    def is_opened(self):
        return self.cap.isOpened()

    def get_frame(self):
        """
        한 프레임을 읽어서 반환합니다.
        읽기 실패 시 None을 반환합니다.
        :return: BGR 이미지(np.ndarray) 또는 None
        """
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self):
        """
        캡처 리소스를 해제합니다.
        """
        if self.cap.isOpened():
            self.cap.release()
'''