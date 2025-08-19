# test6_posture.py
import cv2
import mediapipe as mp
import time

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

import firebase_handler as fb

DEVICE_ID = "rpi5-01"
INTERVAL_DB_SEC = 1.0
INTERVAL_IMG_SEC = 5.0
JPEG_QUALITY = 85

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("Camera open failed")
        return

    detector = PersonDetector()
    pose_extractor = PoseExtractor()
    classifier = PostureClassifierWrapper()

    cv2.namedWindow("Posture V3 Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture V3 Test", 640, 480)

    last_db_send = 0.0
    last_img_send = 0.0
    primary_bbox = None
    current_label = None
    current_view = None

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue

            boxes = detector.detect(frame)
            display = frame.copy()

            primary_bbox = None
            current_label = None
            current_view = None

            for bbox in boxes:
                res = pose_extractor.extract(frame, bbox)
                if not res:
                    continue

                landmarks = res["landmarks"]
                label = classifier.classify(landmarks)
                view = classifier.determine_view_side(landmarks)

                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(display, f"{label} ({view})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                roi = display[y1:y2, x1:x2]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mp_drawing.draw_landmarks(
                    rgb_roi,
                    res["pose_landmarks"],
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                display[y1:y2, x1:x2] = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2BGR)

                if primary_bbox is None:
                    primary_bbox = (x1, y1, x2, y2)
                    current_label = label
                    current_view = view

            now = time.monotonic()

            if primary_bbox is not None and current_label is not None:
                if (now - last_db_send) >= INTERVAL_DB_SEC:
                    try:
                        fb.upload_posture(DEVICE_ID, current_label, current_view, bbox=primary_bbox)
                    except Exception as e:
                        print(f"[FB] posture upload error: {e}")
                    last_db_send = now
            if (now - last_img_send) >= INTERVAL_IMG_SEC:
                try:
                    fb.upload_frame(DEVICE_ID, frame, jpeg_quality=JPEG_QUALITY)
                except Exception as e:
                    print(f"[FB] image upload error: {e}")
                last_img_send = now

            cv2.imshow("Posture V3 Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try:
            handler.release()
        except Exception:
            pass
        try:
            pose_extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
