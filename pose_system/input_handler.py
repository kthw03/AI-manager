# input_handler.py
# 횇챘횉횛 ?횚쨌횂 횉횣쨉챕쨌짱 (Picamera2 쩔챙쩌짹, 횈횆?횕 ?횚쨌횂?쨘 OpenCV쨌횓 횄쨀쨍짰)

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except Exception:
    _HAS_PICAMERA2 = False


class InputHandler:
    def __init__(self, source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        쩔쨉쨩처 쩌횘쩍쨘 횄횎짹창횊짯
        :param source: int/None -> Raspberry Pi 횆짬쨍횧쨋처(Picamera2),
                       str -> 쨉쩔쩔쨉쨩처 횈횆?횕 째챈쨌횓(OpenCV)
        """
        self.width = int(width)
        self.height = int(height)

        self._mode = None           # "picam2" or "cv2"
        self.cap = None             # for cv2
        self.picam2 = None          # for picamera2
        self._opened = False

        # 횈횆?횕 째챈쨌횓(str)?횑쨍챕 OpenCV쨌횓 횄쨀쨍짰
        if isinstance(source, str):
            self._init_cv2(source)
            return

        # 짹창쨘쨩: 쨋처횁챤쨘짙쨍짰횈횆?횑 횆짬쨍횧쨋처 쩍횄쨉쨉
        if _HAS_PICAMERA2:
            self._init_picam2()
        else:
            # Picamera2째징 쩐첩?쨍쨍챕 ?짜횆쨌(OpenCV)쨌횓 쩍횄쨉쨉
            self._init_cv2(source)

    def _init_picam2(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self._mode = "picam2"
            self._opened = True
        except Exception as e:
            print(f"?? Picamera2 횄횎짹창횊짯 쩍횉횈횖: {e}")
            self._opened = False

    def _init_cv2(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"?? 횆짬쨍횧쨋처/횈횆?횕 쩔짯짹창 쩍횉횈횖 (source={source})")
            self._opened = False
            self._mode = "cv2"
            return
        # 횉횠쨩처쨉쨉 쩌쨀횁짚 (째징쨈횋횉횗 째챈쩔챙쩔징쨍쨍 쨔횦쩔쨉)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._mode = "cv2"
        self._opened = True

    def is_opened(self):
        return self._opened

    def get_frame(self):
        """
        횉횗 횉횁쨌쨔?횙?쨩 ?횖쩐챤쩌짯 쨔횦횊짱(BGR).
        쩍횉횈횖 쩍횄 None.
        """
        if not self._opened:
            return None

        if self._mode == "picam2":
            try:
                # Picamera2쨈횂 RGB 쨔챔쩔짯?쨩 쨔횦횊짱 -> BGR쨌횓 쨘짱횊짱횉횠 OpenCV 횊짙횊짱
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None:
                    return None
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                #return frame_bgr
                return frame_rgb
            except Exception as e:
                print(f"?? Picamera2 횉횁쨌쨔?횙 횆쨍횄쨀 쩍횉횈횖: {e}")
                return None

        elif self._mode == "cv2":
            success, frame = self.cap.read()
            if not success:
                return None
            return frame

        return None

    def release(self):
        """
        횆쨍횄쨀 쨍짰쩌횘쩍쨘 횉횠횁짝
        """
        if self._mode == "picam2" and self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
            self._opened = False

        if self._mode == "cv2" and self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
            self._opened = False
