import cv2
from input_handler import InputHandler
from person_detector import PersonDetector
from roi_manager import ROIManager

def main():
    # 카메라/동영상 입력 핸들러 열기
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("카메라 열기에 실패했습니다.")
        return

    # 사람 검출기와 ROI 관리자 준비
    detector = PersonDetector()
    roi_manager = ROIManager(update_interval=10.0)

    print("ROI 테스트 시작: 'q'를 누르면 종료합니다.")

    while True:
        # 프레임 획득
        frame = handler.get_frame()
        if frame is None:
            print("프레임이 비어 있습니다. 종료합니다.")
            break

        # 주기적으로 ROI 자동 갱신(침대/의자만 YOLO로 탐지)
        roi_manager.auto_update(frame)

        # 사람 바운딩 박스 검출
        boxes = detector.detect(frame)

        # 시각화용 프레임 복제 후 ROI 그리기
        display = frame.copy()
        display = roi_manager.draw(display)

        # 각 사람 박스에 대해 ROI 내부/외부 여부 표시
        for bbox in boxes:
            inside = roi_manager.is_bbox_in_roi(bbox)  # 내부에서 모든 ROI에 대해 판정 수행
            x1, y1, x2, y2 = bbox

            color = (0, 255, 0) if inside else (0, 0, 255)
            label = "Inside" if inside else "Outside"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        # 창에 출력
        cv2.imshow("ROI & Person Detector Test", display)

        # 종료 키 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 정리
    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
