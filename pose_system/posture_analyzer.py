# posture_analyzer_v4.py
import time
from collections import deque, namedtuple
from typing import Deque, List, Dict, Optional, Tuple, Any

from utils import calculate_euclidean_distance, get_timestamp
from config import (
    FALL_TRANSITION_TIME,
    NO_MOVEMENT_TIME_THRESHOLD,
    TILT_DURATION,
    POSE_Z_PRONE_THRESHOLD,
    ANALYZER_TILT_THRESHOLD,
    ANALYZER_MOTION_THRESHOLD,
    ANALYZER_IRREGULAR_THRESHOLD,
)

AnalyzedFrame = namedtuple(
    "AnalyzedFrame",
    ["monotonic_ts", "wall_ts", "label", "shoulder_y", "landmarks", "in_roi"]
)

COOL_DOWN = 5.0

# 메타 키(출력 형식 유지)
META_KEYS = ("in_roi", "duration_sec", "changes", "threshold", "nose_minus_hip_z")

# === 신규 규칙 임계값 (요구사항) ===
STANDING_TILT_DURATION = 1.0          # 1) standing_tilt 1초 이상 → falling_warning
OFFROI_LOWPOSTURE_DURATION = 1.0      # 2) ROI 밖에서 sitting/lying 1초 이상 → falling_detect
NO_PERSON_DURATION = 1.0              # 3) 1초 이상 사람 미검출 → patient_escape

def _is_lying(label: str) -> bool:
    return label.startswith("lying")

def _is_low_posture(label: str) -> bool:
    # sitting 또는 lying 전체 포함
    return (label == "sitting") or _is_lying(label)


class PostureAnalyzerV4:
    """
    Time-based posture/event analyzer with a sliding window.
    - Internal timing: time.monotonic()
    - Event timestamps: get_timestamp() based on wall clock
    """

    def __init__(self, roi_manager: Any = None):
        self.roi_manager = roi_manager
        self.buffer: Deque[AnalyzedFrame] = deque()
        self.last_label: Optional[str] = None

        self._last_event_ts: Dict[str, float] = {}
        self._motionless_start: Optional[float] = None
        self._tilt_start: Optional[float] = None

    def update(
        self,
        label: str,
        landmarks: Optional[List[Tuple[float, float, float, float]]],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> None:
        now_mon = time.monotonic()
        now_wall = time.time()

        sh_y = None
        if landmarks and len(landmarks) > 12:
            try:
                sh_y = (landmarks[11][1] + landmarks[12][1]) / 2.0
            except Exception:
                sh_y = None

        in_roi = True
        if self.roi_manager and bbox is not None:
            try:
                in_roi = self.roi_manager.is_bbox_in_roi(bbox)
            except Exception:
                in_roi = True

        # 버퍼 윈도우 유지: 기존 임계 + 신규 1초 윈도우까지 보장
        cutoff = now_mon - max(
            TILT_DURATION,
            NO_MOVEMENT_TIME_THRESHOLD,
            STANDING_TILT_DURATION,
            OFFROI_LOWPOSTURE_DURATION,
            NO_PERSON_DURATION,
        )
        while self.buffer and self.buffer[0].monotonic_ts < cutoff:
            self.buffer.popleft()

        self.buffer.append(
            AnalyzedFrame(now_mon, now_wall, label, sh_y, landmarks, in_roi)
        )
        self.last_label = label

    # ----------------- 기존 상태 조회(유지) ----------------- #
    def get_state(self) -> str:
        now = time.monotonic()
        if self._check_tilt(now):
            return "tilting"
        if self._check_motionless(now):
            return "motionless"
        return self.last_label or "unknown"

    # ----------------- internal sustained checks (기존 유지) ----------------- #
    def _check_tilt(self, now: float) -> bool:
        ys: List[float] = []
        for f in reversed(self.buffer):
            if now - f.monotonic_ts > TILT_DURATION:
                break
            if f.shoulder_y is not None:
                ys.append(f.shoulder_y)

        if not ys:
            self._tilt_start = None
            return False

        if (max(ys) - min(ys)) > ANALYZER_TILT_THRESHOLD:
            if self._tilt_start is None:
                self._tilt_start = now
            return (now - self._tilt_start) >= TILT_DURATION
        else:
            self._tilt_start = None
            return False

    def _check_motionless(self, now: float) -> bool:
        frames = [f for f in self.buffer if now - f.monotonic_ts <= NO_MOVEMENT_TIME_THRESHOLD]
        if len(frames) < 2:
            self._motionless_start = None
            return False

        class P:
            __slots__ = ("x", "y")
            def __init__(self, x: float, y: float):
                self.x = x
                self.y = y

        total, pair_count = 0.0, 0
        for prev, curr in zip(frames, frames[1:]):
            if not prev.landmarks or not curr.landmarks:
                continue
            L = min(len(prev.landmarks), len(curr.landmarks))
            if L == 0:
                continue
            move_sum = 0.0
            for i in range(L):
                try:
                    ax, ay = prev.landmarks[i][0], prev.landmarks[i][1]
                    bx, by = curr.landmarks[i][0], curr.landmarks[i][1]
                except Exception:
                    continue
                move_sum += calculate_euclidean_distance(P(ax, ay), P(bx, by))
            total += (move_sum / max(L, 1))
            pair_count += 1

        if pair_count == 0:
            self._motionless_start = None
            return False

        avg_motion = total / pair_count
        if avg_motion >= ANALYZER_MOTION_THRESHOLD:
            self._motionless_start = None
            return False

        if self._motionless_start is None:
            self._motionless_start = now
        return (now - self._motionless_start) >= NO_MOVEMENT_TIME_THRESHOLD

    def has_transition(self, from_label: str, to_label: str) -> bool:
        if len(self.buffer) < 2:
            return False
        return self.buffer[-2].label == from_label and self.buffer[-1].label == to_label

    # ----------------- 새 규칙: 이벤트 판정 ----------------- #
    def _window(self, duration: float) -> List[AnalyzedFrame]:
        """최근 duration 초의 프레임 집합 반환."""
        now = time.monotonic()
        return [f for f in self.buffer if (now - f.monotonic_ts) <= duration]

    def is_falling_warning(self) -> Tuple[bool, Dict[str, Any]]:
        """
        1) falling_warning: classifier 라벨 'standing_tilt'가 1초 이상 지속.
        """
        frames = self._window(STANDING_TILT_DURATION)
        ok = bool(frames) and all(f.label == "standing_tilt" for f in frames)
        meta = self._meta_template(duration_sec=float(STANDING_TILT_DURATION))
        return ok, meta

    def is_falling_detect(self) -> Tuple[bool, Dict[str, Any]]:
        """
        2) falling_detect: ROI(침대/의자) 외부에서 1초 이상 sitting/lying 유지.
        """
        frames = self._window(OFFROI_LOWPOSTURE_DURATION)
        if not frames:
            return False, self._meta_template(duration_sec=float(OFFROI_LOWPOSTURE_DURATION))
        ok = all((not f.in_roi) and _is_low_posture(f.label) for f in frames)
        meta = self._meta_template(
            in_roi=(frames[-1].in_roi if frames else None),
            duration_sec=float(OFFROI_LOWPOSTURE_DURATION),
        )
        return ok, meta

    def is_patient_escape(self) -> Tuple[bool, Dict[str, Any]]:
        """
        3) patient_escape: 1초 이상 '사람 미검출' 상태.
           - 라벨이 'no_person'으로 들어온다고 가정.
        """
        frames = self._window(NO_PERSON_DURATION)
        if not frames:
            return False, self._meta_template(duration_sec=float(NO_PERSON_DURATION))
        ok = all((f.label == "no_person") for f in frames)
        meta = self._meta_template(duration_sec=float(NO_PERSON_DURATION))
        return ok, meta

    # ----------------- 기존 이벤트(미사용; 남겨둠) ----------------- #
    def is_fall_detected(self) -> bool:
        # 유지: 이전 로직(standing->lying 전이) — 현재 get_events에서는 사용하지 않음
        if not self.buffer:
            return False
        last = self.buffer[-1]
        if not _is_lying(last.label) or last.in_roi:
            return False

        stand_idx = None
        for i in range(len(self.buffer) - 2, -1, -1):
            if self.buffer[i].label == "standing":
                stand_idx = i
                break
        if stand_idx is None:
            return False

        if (last.monotonic_ts - self.buffer[stand_idx].monotonic_ts) > FALL_TRANSITION_TIME:
            return False

        for f in list(self.buffer)[stand_idx + 1:]:
            if f.label == "sitting":
                return False
        return True

    def is_prone_warning(self) -> bool:
        # 유지(미사용)
        if not self.buffer:
            return False
        last = self.buffer[-1]
        if not _is_lying(last.label):
            return False
        lm = last.landmarks
        if not lm or len(lm) <= 24:
            return False
        try:
            nose_z = lm[0][2]
            hip_z = (lm[23][2] + lm[24][2]) / 2.0
        except Exception:
            return False
        return (nose_z - hip_z) < POSE_Z_PRONE_THRESHOLD

    def is_irregular_movement(self) -> bool:
        # 유지(미사용)
        changes = 0
        prev = None
        for f in self.buffer:
            if prev is not None and f.label != prev:
                changes += 1
            prev = f.label
        return changes >= ANALYZER_IRREGULAR_THRESHOLD

    # ----------------- meta helpers (그대로) ----------------- #
    def _meta_template(self, base: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        out = {k: None for k in META_KEYS}
        if base:
            for k in META_KEYS:
                if k in base:
                    out[k] = base[k]
        for k, v in kwargs.items():
            if k in META_KEYS:
                out[k] = v
        return out

    def _normalize_meta(self, meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return self._meta_template(meta or {})

    # ----------------- 이벤트 수집 (출력 형식 유지) ----------------- #
    def _append_event(
        self,
        ev_type: str,
        message: str,
        events: List[Dict[str, Any]],
        severity: str = "warning",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.monotonic()
        last = self._last_event_ts.get(ev_type, 0.0)
        if (now - last) >= COOL_DOWN:
            events.append({
                "type": ev_type,
                "severity": severity,
                "timestamp": get_timestamp(),
                "message": message,
                "meta": self._normalize_meta(meta),
            })
            self._last_event_ts[ev_type] = now

    def get_events(self) -> List[Dict[str, Any]]:
        """
        요구사항 3가지만 이벤트로 내보냄.
        출력 형식(type/severity/timestamp/message/meta)은 동일.
        """
        events: List[Dict[str, Any]] = []

        # 1) falling_warning: standing_tilt 1초 이상
        ok, meta = self.is_falling_warning()
        if ok:
            self._append_event(
                "falling_warning",
                "낙상 경고: standing_tilt 1초 이상",
                events,
                severity="warning",
                meta=meta,
            )

        # 2) falling_detect: ROI 외부에서 1초 이상 sitting/lying
        ok, meta = self.is_falling_detect()
        if ok:
            self._append_event(
                "falling_detect",
                "낙상 감지: ROI 외부에서 1초 이상 sitting/lying",
                events,
                severity="critical",
                meta=meta,
            )

        # 3) patient_escape: 1초 이상 사람 미검출
        ok, meta = self.is_patient_escape()
        if ok:
            self._append_event(
                "patient_escape",
                "환자 이탈: 1초 이상 사람 미검출",
                events,
                severity="critical",
                meta=meta,
            )

        return events
