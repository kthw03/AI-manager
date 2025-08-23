import cv2
from input_handler import InputHandler
from person_detector import PersonDetector
from roi_manager import ROIManager

def put_text_lines(img, lines, org=(10, 30), color=(255, 255, 255)):
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(img, str(txt), (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    # 카메라/동영상 입력 핸들러 열기
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("카메라 열기에 실패했습니다.")
        return

    # 사람 검출기와 ROI 관리자 준비
    detector = PersonDetector()
    roi_manager = ROIManager(update_interval=10.0)

    # 창을 먼저 만들고 마우스 콜백을 연결(수동 ROI 드래그용)
    win_name = "ROI & Person Detector Test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 720)
    if hasattr(roi_manager, "handle_mouse"):
        cv2.setMouseCallback(win_name, roi_manager.handle_mouse)

    print("ROI 테스트 시작: 'q' 종료, 'e' 편집모드 토글, 'c' 수동 ROI 삭제, 마우스 드래그로 수동 ROI 추가")

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
            cv2.putText(display, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 상단 안내 텍스트
        auto_cnt = len(getattr(roi_manager, "auto_rois", [])) if hasattr(roi_manager, "auto_rois") else len(roi_manager.get_rois())
        manual_cnt = len(getattr(roi_manager, "manual_rois", [])) if hasattr(roi_manager, "manual_rois") else 0
        edit_mode = getattr(roi_manager, "edit_enabled", False)
        lines = [
            f"Auto ROI: {auto_cnt} | Manual ROI: {manual_cnt} | EditMode: {'ON' if edit_mode else 'OFF'}",
            "키: E=편집모드 토글, C=수동 ROI 전체 삭제, Q=종료",
            "편집모드가 ON이면 마우스 드래그로 수동 ROI를 추가할 수 있습니다.",
        ]
        put_text_lines(display, lines, org=(10, 30), color=(255, 255, 255))

        # 창에 출력
        cv2.imshow(win_name, display)

        # 키 처리: q 종료, e 편집 토글, c 수동 ROI 전체 삭제
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e') and hasattr(roi_manager, "toggle_edit_mode"):
            roi_manager.toggle_edit_mode()
        if key == ord('c') and hasattr(roi_manager, "clear_manual_rois"):
            roi_manager.clear_manual_rois()

    # 리소스 정리
    handler.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
