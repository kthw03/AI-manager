# test_roi_manager.py

import cv2
from input_handler import InputHandler
from person_detector import PersonDetector
from roi_manager import ROIManager

def main():
    handler = InputHandler(source=0)
    detector = PersonDetector()
    roi_manager = ROIManager(update_interval=10.0)

    print("ROI Manager 테스트: 침대/의자 영역이 10초마다 자동 갱신되고, 사람 바운딩 박스의 안/밖을 표시합니다. 'q'로 종료하세요.")

    while True:
        frame = handler.get_frame()
        if frame is None:
            print("프레임 읽기 실패 - 종료합니다.")
            break

        # 1) ROI 자동 갱신
        roi_manager.auto_update(frame)

        # 2) 사람 검출
        boxes = detector.detect(frame)

        # 3) ROI와 Person 박스 시각화
        display = frame.copy()
        display = roi_manager.draw(display)  # ROI 그리기

        for bbox in boxes:
            x1, y1, x2, y2 = bbox

            # 4) 각 ROI coords에 대해 inside 여부 판정
            inside = any(
                roi_manager.is_bbox_in_roi(bbox, roi["coords"])
                for roi in roi_manager.rois
            )

            color = (0, 255, 0) if inside else (0, 0, 255)
            label = "Inside" if inside else "Outside"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("ROI & Person Detector Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("테스트 종료")
            break

    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
