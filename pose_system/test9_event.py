# test9.py
import cv2
import time
import mediapipe as mp

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper
from posture_analyzer import PostureAnalyzerV4
import firebase_handler as fb

DEVICE_ID = "rpi5-01"
INTERVAL_DB_SEC = 1.0

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
    analyzer = PostureAnalyzerV4()

    cv2.namedWindow("Posture V3 Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture V3 Test", 640, 480)

    last_db_send = 0.0

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue

            display = frame.copy()
            boxes = detector.detect(frame)

            primary_bbox = None
            current_label = None
            current_view = None
            landmarks = None
            pose_lm = None

            if boxes:
                boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                x1, y1, x2, y2 = boxes[0]
                primary_bbox = (x1, y1, x2, y2)

                res = pose_extractor.extract(frame, primary_bbox)
                if res:
                    landmarks = res["landmarks"]
                    pose_lm = res["pose_landmarks"]
                    current_label = classifier.classify(landmarks)
                    current_view = classifier.determine_view_side(landmarks)

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"{current_label} ({current_view})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    roi = display[y1:y2, x1:x2]
                    if roi.size > 0 and pose_lm is not None:
                        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        mp_drawing.draw_landmarks(
                            rgb_roi, pose_lm, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )
                        display[y1:y2, x1:x2] = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2BGR)

            if current_label and primary_bbox:
                analyzer.update(current_label, landmarks, primary_bbox)
            else:
                analyzer.update("no_person", None, None)

            state = analyzer.get_state()

            fw, _ = analyzer.is_falling_warning()
            fd, _ = analyzer.is_falling_detect()
            pe, _ = analyzer.is_patient_escape()
            sf, _ = analyzer.is_standing_freeze()

            anomalies = {
                "falling_warning": bool(fw),
                "falling_detect": bool(fd),
                "patient_escape": bool(pe),
                "standing_freeze": bool(sf),
            }

            now = time.time()
            if (now - last_db_send) >= INTERVAL_DB_SEC:
                up_label = current_label if current_label else "no_person"
                up_view = current_view if current_view else "unknown"
                up_bbox = primary_bbox if primary_bbox else None
                ok = fb.upload_posture(
                    device_id=DEVICE_ID,
                    label=up_label,
                    view=up_view,
                    bbox=up_bbox,
                    anomalies=anomalies,
                    state=state
                )
                if not ok:
                    print("[APP][WARN] posture upload failed")
                last_db_send = now

            cv2.imshow("Posture V3 Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try: handler.release()
        except: pass
        try: pose_extractor.close()
        except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
