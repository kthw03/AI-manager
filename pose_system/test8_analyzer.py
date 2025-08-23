# test_analyzer_realtime.py
import time
import cv2
import mediapipe as mp

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

# analyzer import (v4)
import posture_analyzer as pa  # file should define PostureAnalyzerV4

# optional ROI manager (safe import)
try:
    from roi_manager import ROIManager
    _HAS_ROI = True
except Exception:
    ROIManager = None
    _HAS_ROI = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def pick_main_box(boxes):
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]
    # pick by area
    areas = [max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) for b in boxes]
    return boxes[int(max(range(len(areas)), key=lambda i: areas[i]))]


def put_text_lines(img, lines, org=(10, 30), color=(255, 255, 255)):
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(img, str(txt), (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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
    analyzer = pa.PostureAnalyzerV4(roi_manager=roi_manager)

    cv2.namedWindow("Analyzer Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analyzer Test", 960, 720)

    last_event_print_ts = 0.0

    # 활성 경보를 유지하는 딕셔너리: 조건이 해제되기 전까지 화면에 계속 표시
    active_alerts = {}  # key -> message

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue

            display = frame.copy()

            # ROI update/draw (if available)
            if roi_manager and hasattr(roi_manager, "update"):
                try:
                    roi_manager.update(frame)
                except Exception:
                    pass
            if roi_manager and hasattr(roi_manager, "draw"):
                try:
                    display = roi_manager.draw(display)
                except Exception:
                    pass

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
                    cv2.putText(display, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)

                    maybe_draw_pose(display, main_box, pose_lm)
                else:
                    analyzer.update(label="unknown", landmarks=None, bbox=main_box)
            else:
                analyzer.update(label="no_person", landmarks=None, bbox=None)

            state = analyzer.get_state()
            events = analyzer.get_events()  # 쿨다운 적용된 "새로 발생한" 이벤트들

            # ---- 활성 경보 갱신: 조건 참/해제에 따라 켜고 끄기 ----
            ok_fw, _ = analyzer.is_falling_warning()
            if ok_fw:
                active_alerts["falling_warning"] = "낙상 경고: standing_tilt 1초 이상"
            else:
                active_alerts.pop("falling_warning", None)

            ok_fd, _ = analyzer.is_falling_detect()
            if ok_fd:
                active_alerts["falling_detect"] = "낙상 감지: ROI 외부에서 1초 이상 sitting/lying"
            else:
                active_alerts.pop("falling_detect", None)

            ok_pe, _ = analyzer.is_patient_escape()
            if ok_pe:
                active_alerts["patient_escape"] = "환자 이탈: 1초 이상 사람 미검출"
            else:
                active_alerts.pop("patient_escape", None)
            # ------------------------------------------------------

            # 상단 정보 라인
            lines = [
                f"State: {state}",
                f"Last label: {analyzer.last_label}",
                f"Events (this frame): {len(events)}",   # 새로 발생한 이벤트 개수
                f"Active alerts: {len(active_alerts)}",  # 현재 유지 중인 경보 수
                f"ROI: {'ON' if roi_manager else 'OFF'}",
                "Press 'q' to quit",
            ]
            put_text_lines(display, lines, org=(10, 30), color=(255, 255, 255))

            # 🔴 활성 경보를 지속 표기 (중요도 순서: detect > escape > warning)
            y_offset = 150
            order = ["falling_detect", "patient_escape", "falling_warning"]
            for key in order:
                if key in active_alerts:
                    msg = f"[{key}] {active_alerts[key]}"
                    cv2.putText(display, msg, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 24

            # 콘솔 로그: 새 이벤트가 있을 때만 출력(여전히 쿨다운 영향)
            now = time.monotonic()
            if events and (now - last_event_print_ts) > 0.2:
                for ev in events:
                    print(ev)
                last_event_print_ts = now

            cv2.imshow("Analyzer Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        handler.release()
        pose_extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
