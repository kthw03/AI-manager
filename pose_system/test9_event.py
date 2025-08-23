# test10.py
import time
import cv2
import mediapipe as mp

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

try:
    from posture_analyzer import PostureAnalyzerV4
except Exception:
    import posture_analyzer as pa
    PostureAnalyzerV4 = pa.PostureAnalyzerV4

try:
    from roi_manager import ROIManager
    _HAS_ROI = True
except Exception:
    ROIManager = None
    _HAS_ROI = False

import firebase_handler as fb

DEVICE_ID = "rpi5-01"
INTERVAL_DB_SEC = 1.0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def pick_main_box(boxes):
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    areas = [max(0, (b[2]-b[0])) * max(0, (b[3]-b[1])) for b in boxes]
    return boxes[int(max(range(len(areas)), key=lambda i: areas[i]))]


def put_text_lines(img, lines, org=(10,30), color=(255,255,255)):
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(img, str(txt), (x, y + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def maybe_draw_pose(display, bbox, pose_landmarks):
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
    handler = InputHandler(source=0)
    if not handler.is_opened():
        print("Camera open failed")
        return

    detector = PersonDetector()
    pose_extractor = PoseExtractor()
    classifier = PostureClassifierWrapper()
    roi_manager = ROIManager(update_interval=8.0) if _HAS_ROI else None
    analyzer = PostureAnalyzerV4(roi_manager=roi_manager)

    cv2.namedWindow("Analyzer Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analyzer Test", 960, 720)
    if roi_manager and hasattr(roi_manager, "handle_mouse"):
        cv2.setMouseCallback("Analyzer Test", roi_manager.handle_mouse)
        # ← 4점 클릭이 바로 먹히도록 기본 ON
        if hasattr(roi_manager, "set_edit_enabled"):
            roi_manager.set_edit_enabled(True)

    last_event_print_ts = 0.0
    last_db_send = 0.0
    active_alerts = {}

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue
            display = frame.copy()

            # (선택) YOLO 자동 ROI도 쓰고 싶으면 다음 줄 주석 해제
            # if roi_manager: roi_manager.auto_update(frame)

            if roi_manager:
                display = roi_manager.draw(display)

            boxes = detector.detect(frame)
            main_box = pick_main_box(boxes)

            current_label = None
            current_view = None
            landmarks = None
            pose_lm = None

            if main_box is not None:
                x1, y1, x2, y2 = main_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 2)
                res = pose_extractor.extract(frame, main_box)
                if res:
                    landmarks = res["landmarks"]
                    pose_lm = res["pose_landmarks"]
                    current_label = classifier.classify(landmarks)
                    current_view = classifier.determine_view_side(landmarks)
                    analyzer.update(label=current_label, landmarks=landmarks, bbox=main_box)
                    cv2.putText(display, f"{current_label} ({current_view})",
                                (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,160,255), 2)
                    maybe_draw_pose(display, main_box, pose_lm)
                else:
                    analyzer.update(label="unknown", landmarks=None, bbox=main_box)
            else:
                analyzer.update(label="no_person", landmarks=None, bbox=None)

            state = analyzer.get_state()
            events = analyzer.get_events()

            ok_fw, _ = analyzer.is_falling_warning()
            if ok_fw: active_alerts["falling_warning"] = "낙상 경고: standing_tilt 1초 이상"
            else: active_alerts.pop("falling_warning", None)

            ok_fd, _ = analyzer.is_falling_detect()
            if ok_fd: active_alerts["falling_detect"] = "낙상 감지: ROI 유무 규칙 기반 sitting/lying"
            else: active_alerts.pop("falling_detect", None)

            ok_pe, _ = analyzer.is_patient_escape()
            if ok_pe: active_alerts["patient_escape"] = "환자 이탈: 1초 이상 사람 미검출"
            else: active_alerts.pop("patient_escape", None)

            ok_sf, _ = analyzer.is_standing_freeze()
            if ok_sf: active_alerts["standing_freeze"] = "서 있는 상태에서 10초 이상 움직임 없음"
            else: active_alerts.pop("standing_freeze", None)

            roi_status = "OFF"
            pending = 0
            if roi_manager:
                try:
                    roi_count = len(roi_manager.get_rois())
                    pending = len(getattr(roi_manager, "_click_points", []))
                    edit_mode = getattr(roi_manager, "edit_enabled", False)
                    roi_status = f"ON ({roi_count}) | edit:{'ON' if edit_mode else 'OFF'} | pending:{pending}/4"
                except Exception:
                    roi_status = "ON"

            lines = [
                f"State: {state}",
                f"Last label: {analyzer.last_label}",
                f"Events (this frame): {len(events)}",
                f"Active alerts: {len(active_alerts)}",
                f"ROI: {roi_status}",
                "Left-click 4 points (order-free) to set bed ROI; Right-click to undo last.",
                "Keys: E=toggle edit  C=clear manual ROI  Q=quit",
            ]
            put_text_lines(display, lines, org=(10,30), color=(255,255,255))

            y_offset = 170
            for key in ["falling_detect","patient_escape","standing_freeze","falling_warning"]:
                if key in active_alerts:
                    msg = f"[{key}] {active_alerts[key]}"
                    cv2.putText(display, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    y_offset += 24

            now = time.time()
            if (now - last_db_send) >= INTERVAL_DB_SEC:
                up_label = current_label if current_label else "no_person"
                up_view = current_view if current_view else "unknown"
                up_bbox = main_box if main_box else None
                anomalies = {
                    "falling_warning": bool(ok_fw),
                    "falling_detect": bool(ok_fd),
                    "patient_escape": bool(ok_pe),
                    "standing_freeze": bool(ok_sf),
                }
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

            cv2.imshow("Analyzer Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e') and roi_manager and hasattr(roi_manager, "toggle_edit_mode"):
                roi_manager.toggle_edit_mode()
            if key == ord('c') and roi_manager and hasattr(roi_manager, "clear_manual_rois"):
                roi_manager.clear_manual_rois()

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
