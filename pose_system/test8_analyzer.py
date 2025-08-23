# test_analyzer_realtime.py
import time
import cv2
import mediapipe as mp

from input_handler import InputHandler
from person_detector import PersonDetector
from pose_extractor import PoseExtractor
from posture_wrapper import PostureClassifierWrapper

# 분석기 import (우선 posture_analyzer_v4, 실패 시 기존 이름 백업)
try:
    from posture_analyzer import PostureAnalyzerV4
except Exception:
    import posture_analyzer as pa
    PostureAnalyzerV4 = pa.PostureAnalyzerV4  # 파일명이 다른 프로젝트 환경 호환

# ROI 매니저 import (실패해도 동작 가능)
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
    areas = [max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) for b in boxes]
    return boxes[int(max(range(len(areas)), key=lambda i: areas[i]))]


def put_text_lines(img, lines, org=(10, 30), color=(255, 255, 255)):
    """여러 줄 텍스트를 일정 간격으로 그리기."""
    x, y = org
    for i, txt in enumerate(lines):
        cv2.putText(img, str(txt), (x, y + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


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

    # ROI 매니저: 있으면 사용, 없어도 분석기는 동작
    roi_manager = ROIManager(update_interval=8.0) if _HAS_ROI else None
    analyzer = PostureAnalyzerV4(roi_manager=roi_manager)

    # 창 준비
    cv2.namedWindow("Analyzer Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Analyzer Test", 960, 720)

    last_event_print_ts = 0.0

    # 화면에 지속적으로 띄울 활성 경보(조건 해제 시 제거)
    active_alerts = {}  # key -> message

    try:
        while True:
            frame = handler.get_frame()
            if frame is None:
                continue
            display = frame.copy()

            # ROI 자동 갱신 및 시각화
            if roi_manager:
                try:
                    roi_manager.auto_update(frame)   # ← 핵심: update()가 아닌 auto_update()
                    display = roi_manager.draw(display)
                except Exception:
                    pass

            # 사람 검출 및 대표 박스 선택
            boxes = detector.detect(frame)
            main_box = pick_main_box(boxes)

            if main_box is not None:
                # 대표 박스 표시
                x1, y1, x2, y2 = main_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 0), 2)

                # 포즈 추출 → 라벨 분류 → 분석기에 업데이트
                res = pose_extractor.extract(frame, main_box)
                if res:
                    landmarks = res["landmarks"]
                    pose_lm = res["pose_landmarks"]

                    label = classifier.classify(landmarks)
                    view = classifier.determine_view_side(landmarks)

                    analyzer.update(label=label, landmarks=landmarks, bbox=main_box)

                    # 현재 라벨 표시
                    label_text = f"{label} ({view})"
                    cv2.putText(display, label_text, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)

                    # 포즈 랜드마크 그리기
                    maybe_draw_pose(display, main_box, pose_lm)
                else:
                    # 포즈 실패 시 라벨만 unknown으로 업데이트
                    analyzer.update(label="unknown", landmarks=None, bbox=main_box)
            else:
                # 아무도 없으면 no_person
                analyzer.update(label="no_person", landmarks=None, bbox=None)

            # 상태/이벤트 가져오기(이벤트는 쿨다운 반영된 신규만)
            state = analyzer.get_state()
            events = analyzer.get_events()

            # ---- 활성 경보 갱신: 조건 참/해제에 따라 화면 유지/제거 ----
            ok_fw, _ = analyzer.is_falling_warning()
            if ok_fw:
                active_alerts["falling_warning"] = "낙상 경고: standing_tilt 1초 이상"
            else:
                active_alerts.pop("falling_warning", None)

            ok_fd, _ = analyzer.is_falling_detect()
            if ok_fd:
                # 새 규칙 반영 문구(ROI 없음이거나 ROI 밖에서 sitting/lying 지속)
                active_alerts["falling_detect"] = "낙상 감지: ROI 유무 규칙 기반 sitting/lying"
            else:
                active_alerts.pop("falling_detect", None)

            ok_pe, _ = analyzer.is_patient_escape()
            if ok_pe:
                active_alerts["patient_escape"] = "환자 이탈: 1초 이상 사람 미검출"
            else:
                active_alerts.pop("patient_escape", None)

            ok_sf, _ = analyzer.is_standing_freeze()
            if ok_sf:
                active_alerts["standing_freeze"] = "서 있는 상태에서 10초 이상 움직임 없음"
            else:
                active_alerts.pop("standing_freeze", None)
            # -------------------------------------------------------------

            # 상단 정보 라인
            roi_status = "OFF"
            if roi_manager:
                try:
                    roi_count = len(roi_manager.get_rois())
                    roi_status = f"ON ({roi_count})"
                except Exception:
                    roi_status = "ON"
            lines = [
                f"State: {state}",
                f"Last label: {analyzer.last_label}",
                f"Events (this frame): {len(events)}",
                f"Active alerts: {len(active_alerts)}",
                f"ROI: {roi_status}",
                "Press 'q' to quit",
            ]
            put_text_lines(display, lines, org=(10, 30), color=(255, 255, 255))

            # 🔴 활성 경보를 지속 표기 (중요도 순: detect > escape > freeze > warning)
            y_offset = 150
            order = ["falling_detect", "patient_escape", "standing_freeze", "falling_warning"]
            for key in order:
                if key in active_alerts:
                    msg = f"[{key}] {active_alerts[key]}"
                    cv2.putText(display, msg, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_offset += 24

            # 콘솔 로그: 새 이벤트가 있을 때만 출력(쿨다운 영향)
            now = time.monotonic()
            if events and (now - last_event_print_ts) > 0.2:
                for ev in events:
                    print(ev)
                last_event_print_ts = now

            # 화면 갱신 및 종료 키
            cv2.imshow("Analyzer Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        handler.release()
        try:
            pose_extractor.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
