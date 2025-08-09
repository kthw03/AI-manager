import cv2
import mediapipe as mp
from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

def main():
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("❌ 카메라 열기 실패")
        return

    detector = PersonDetector()
    pose_extractor = PoseExtractor()
    classifier = PostureClassifierWrapper()

    cv2.namedWindow("Posture V3 Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Posture V3 Test", 640, 480)

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue

            boxes = detector.detect(frame)
            display = frame.copy()

            for bbox in boxes:
                res = pose_extractor.extract(frame, bbox)
                if not res:
                    continue

                landmarks = res["landmarks"]
                label = classifier.classify(landmarks)
                view = classifier.determine_view_side(landmarks)

                # 바운딩 박스 + 라벨 표시
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"{label} ({view})"
                cv2.putText(display, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # 랜드마크 시각화
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

            cv2.imshow("Posture V3 Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        handler.release()
        pose_extractor.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
