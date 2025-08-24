# test10_analyzer.py

import time
import cv2
import mediapipe as mp

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

# 분석기 import (심플한 폴백)
try:
    from posture_analyzer import PostureAnalyzerV4
except Exception:
    import posture_analyzer as pa
    PostureAnalyzerV4 = pa.PostureAnalyzerV4

# ROI 매니저 import
try:
    from roi_manager import ROIManager
    _HAS_ROI = True
except Exception:
    ROIManager = None
    _HAS_ROI = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def pick_main_box(boxes):
    """사람 박스가 여러 개면 면적이 가장 큰 박스를 선택."""
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    areas = [max(0, (b[2]-b[0])) * max(0, (b[3]-b[1])) for b in boxes]
    return boxes[int(max(range(len(areas)), key=lambda i: areas[i]))]


def put_text_lines(img, lines, org=(10, 30), color=(255, 255, 255)):
    """여러 줄 텍스트를 일정 간격으로 그리기."""
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(img, str(txt), (x, y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def maybe_draw_pose(display, bbox, pose_landmarks):
    """선택된 사람 박스 내부에 포즈 랜드마크 시각화."""
    if bbox is None or pose_landmarks is None:
        return
    x1, y1, x2, y2 = bbox
    roi = display[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_drawing.draw_landmarks(
        rgb_roi,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(thickness=2),
    )
    display[y1:y2, x1:x2] = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2BGR)


def main():
    # 입력 소스 열기
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("Camera open failed")
        return

    # 모델 준비
    detector = PersonDetector()
    pose_extractor = PoseExtractor()
    classifier = PostureClassifierWrapper()

    # ROI 매니저(자동 + 수동 4점 폴리곤)
    roi_manager = ROIManager(update_interval=8.0) if _HAS_ROI else None
    analyzer = PostureAnalyzerV4(roi_manager=roi_manager)

    # 창 준비 + 마우스 콜백 연결(수동 ROI: 좌클릭 4점, 우클릭 되돌리기)
    win = "Analyzer Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    if roi_manager and hasattr(roi_manager, "handle_mouse"):
        cv2.setMouseCallback(win, roi_manager.handle_mouse)

    last_event_print_ts = 0.0
    active_alerts = {}

    # ---- FIX: 경보 on/off를 확실히 반영하는 헬퍼 (해제 시 키 제거) ----
    def set_alert(key: str, is_on: bool, msg: str):
        if is_on:
            active_alerts[key] = msg
        else:
            active_alerts.pop(key, None)
    # -------------------------------------------------------------------

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue
            display = frame.copy()

            # ROI 자동 갱신 및 시각화
            if roi_manager:
                try:
                    roi_manager.auto_update(frame)      # 자동(YOLO) ROI
                except Exception:
                    pass
                display = roi_manager.draw(display)     # 자동+수동 모두 그림

            # 사람 검출 및 대표 박스 선택
            boxes = detector.detect(frame)
            main_box = pick_main_box(boxes)

            if main_box is not None:
                x1, y1, x2, y2 = main_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 2)

                res = pose_extractor.extract(frame, main_box)
                if res:
                    landmarks = res["landmarks"]
                    pose_lm = res["pose_landmarks"]

                    label = classifier.classify(landmarks)
                    view = classifier.determine_view_side(landmarks)

                    analyzer.update(label=label, landmarks=landmarks, bbox=main_box)

                    label_text = f"{label} ({view})"
                    cv2.putText(display, label_text, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)

                    maybe_draw_pose(display, main_box, pose_lm)
                else:
                    analyzer.update(label="unknown", landmarks=None, bbox=main_box)
            else:
                analyzer.update(label="no_person", landmarks=None, bbox=None)

            # 상태/이벤트
            state = analyzer.get_state()
            events = analyzer.get_events()

            # 활성 경보 유지/해제 (우선순위 표시는 아래 draw 구간에서 처리)
            ok_fw, _ = analyzer.is_falling_warning()
            set_alert("falling_warning", ok_fw, "낙상 경고: standing_tilt 1초 이상")

            ok_fd, _ = analyzer.is_falling_detect()
            set_alert("falling_detect", ok_fd, "낙상 감지: ROI 유무 규칙 기반 sitting/lying")

            ok_pe, _ = analyzer.is_patient_escape()
            set_alert("patient_escape", ok_pe, "환자 이탈: 1초 이상 사람 미검출")

            ok_sf, _ = analyzer.is_standing_freeze()
            set_alert("standing_freeze", ok_sf, "서 있는 상태에서 10초 이상 움직임 없음")

            # 상단 정보 라인 + 사용법(4점/사다리꼴, 진행 점 카운트 표시)
            auto_cnt = len(getattr(roi_manager, "auto_rois", [])) if roi_manager else 0
            manual_cnt = len(getattr(roi_manager, "manual_rois", [])) if roi_manager else 0
            edit_mode = getattr(roi_manager, "edit_enabled", False) if roi_manager else False
            pending_pts = len(getattr(roi_manager, "_click_points", [])) if roi_manager else 0
            lines = [
                f"State: {state}",
                f"Last label: {analyzer.last_label}",
                f"Events (this frame): {len(events)}",
                f"Active alerts: {len(active_alerts)}",
                f"ROI: {'ON' if roi_manager else 'OFF'} (auto:{auto_cnt}, manual:{manual_cnt}, edit:{'ON' if edit_mode else 'OFF'}, pending:{pending_pts}/4)",
                "Keys: E=edit mode toggle  C=clear manual ROI  Q=quit  (우클릭=되돌리기)",
                "Edit ON 상태에서 화면에 좌클릭 4번(순서 무관)으로 ROI(사다리꼴 포함)를 확정합니다.",
            ]
            put_text_lines(display, lines, org=(10, 30), color=(255, 255, 255))

            # 활성 경보 표시(우선순: detect > escape > freeze > warning)
            y_offset = 170
            for key in ["falling_detect", "patient_escape", "standing_freeze", "falling_warning"]:
                if key in active_alerts and active_alerts[key]:
                    msg = f"[{key}] {active_alerts[key]}"
                    cv2.putText(display, msg, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 24

            # 콘솔 로그(새 이벤트만)
            now = time.monotonic()
            if events and (now - last_event_print_ts) > 0.2:
                for ev in events:
                    print(ev)
                last_event_print_ts = now

            # 화면 갱신 + 키 처리
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e') and roi_manager and hasattr(roi_manager, "toggle_edit_mode"):
                roi_manager.toggle_edit_mode()
            if key == ord('c') and roi_manager and hasattr(roi_manager, "clear_manual_rois"):
                roi_manager.clear_manual_rois()

    finally:
        handler.release()
        try:
            pose_extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
