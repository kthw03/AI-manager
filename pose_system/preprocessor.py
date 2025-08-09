# preprocessor.py

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

class Preprocessor:
    """
    공통 전처리 모듈: 전체 프레임 리사이즈, BGR->RGB 변환, 노이즈 제거 등을 수행합니다.
    ROI 크롭은 이 모듈이 아닌 ROIManager에서 처리합니다.
    """
    def __init__(self, blur_kernel=(5,5), interp=cv2.INTER_AREA):
        self.blur_kernel = blur_kernel
        self.interp      = interp

    def preprocess(self, frame):
        """
        프레임 전처리 수행:
          1. 리사이즈 (FRAME_WIDTH x FRAME_HEIGHT)
          2. BGR -> RGB 변환
          3. 노이즈 제거 (Gaussian Blur)

        :param frame: 원본 BGR 이미지 (np.ndarray)
        :return: 전처리된 전체 프레임 (RGB, np.ndarray)
        """
        # 1. 리사이즈
        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=self.interp)

        # 2. 컬러 변환 (BGR -> RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. 노이즈 제거 (Gaussian Blur)
        denoised = cv2.GaussianBlur(rgb, self.blur_kernel, 0)

        return denoised
