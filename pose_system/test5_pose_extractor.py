# main.py 수정 예시

import cv2
import mediapipe as mp
from input_handler import InputHandler
from person_detector import PersonDetector
from roi_manager import ROIManager
from pose_extractor import PoseExtractor

mp_drawing = mp.solutions.drawing_utils
mp_pose    = mp.solutions.pose

def main():
    handler        = InputHandler(source=0)
    if not handler.is_opened():
        print("카메라 열기 실패")
        return
    detector       = PersonDetector()
    roi_manager    = ROIManager(update_interval=10.0)
    pose_extractor = PoseExtractor()

    cv2.namedWindow("AI Caregiver System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Caregiver System", 640, 480)

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue

            roi_manager.auto_update(frame)
            boxes = detector.detect(frame)

            display = frame.copy()
            display = roi_manager.draw(display)

            for bbox in boxes:
                x1, y1, x2, y2 = bbox
                # 1) Pose 추출
                res = pose_extractor.extract(frame, bbox)
                if not res:
                    continue

                # 2) ROI 그대로 가져오기
                roi = display[y1:y2, x1:x2]
                # 3) RGB로 변환 (mp_drawing이 RGB 입력을 기대)
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # 4) 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    rgb_roi,
                    res["pose_landmarks"],
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

                # 5) 다시 BGR로 바꿔서 원본 위치에 덮어쓰기
                display[y1:y2, x1:x2] = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2BGR)

                # 6) 바운딩 박스 + inside/outside 표시
                inside = roi_manager.is_bbox_in_roi(bbox)
                color  = (0,255,0) if inside else (0,0,255)
                label  = "Inside" if inside else "Outside"
                cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("AI Caregiver System", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        handler.release()
        pose_extractor.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
