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


def _is_lying(label: str) -> bool:
    return label.startswith("lying")


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

        cutoff = now_mon - max(TILT_DURATION, NO_MOVEMENT_TIME_THRESHOLD)
        while self.buffer and self.buffer[0].monotonic_ts < cutoff:
            self.buffer.popleft()

        self.buffer.append(
            AnalyzedFrame(now_mon, now_wall, label, sh_y, landmarks, in_roi)
        )
        self.last_label = label

    def get_state(self) -> str:
        now = time.monotonic()
        if self._check_tilt(now):
            return "tilting"
        if self._check_motionless(now):
            return "motionless"
        return self.last_label or "unknown"

    # ----------------- internal sustained checks ----------------- #

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

        # distance helper for (x, y, z, v) tuples
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

    # ----------------- event primitives ----------------- #

    def is_fall_detected(self) -> bool:
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
        changes = 0
        prev = None
        for f in self.buffer:
            if prev is not None and f.label != prev:
                changes += 1
            prev = f.label
        return changes >= ANALYZER_IRREGULAR_THRESHOLD

    # ----------------- unified public event wrappers ----------------- #

    def is_motionless_sustained(self) -> Tuple[bool, Dict[str, Any]]:
        now = time.monotonic()
        ok = self._check_motionless(now)
        last = self.buffer[-1] if self.buffer else None
        meta = {
            "duration_sec": float(NO_MOVEMENT_TIME_THRESHOLD),
            "in_roi": (last.in_roi if last else None),
        }
        return ok, meta

    def is_tilt_sustained(self) -> Tuple[bool, Dict[str, Any]]:
        now = time.monotonic()
        ok = self._check_tilt(now)
        meta = {"duration_sec": float(TILT_DURATION)}
        return ok, meta

    def is_fall_event(self) -> Tuple[bool, Dict[str, Any]]:
        ok = self.is_fall_detected()
        last = self.buffer[-1] if self.buffer else None
        meta = {
            "in_roi": (last.in_roi if last else None),
        }
        return ok, meta

    def is_prone_event(self) -> Tuple[bool, Dict[str, Any]]:
        ok = self.is_prone_warning()
        meta: Dict[str, Any] = {}
        if ok and self.buffer and self.buffer[-1].landmarks and len(self.buffer[-1].landmarks) > 24:
            lm = self.buffer[-1].landmarks
            try:
                meta["nose_minus_hip_z"] = lm[0][2] - (lm[23][2] + lm[24][2]) / 2.0
                meta["threshold"] = float(POSE_Z_PRONE_THRESHOLD)
            except Exception:
                pass
        return ok, meta

    def is_irregular_event(self) -> Tuple[bool, Dict[str, Any]]:
        changes = 0
        prev = None
        for f in self.buffer:
            if prev is not None and f.label != prev:
                changes += 1
            prev = f.label
        ok = changes >= ANALYZER_IRREGULAR_THRESHOLD
        meta = {"changes": int(changes), "threshold": int(ANALYZER_IRREGULAR_THRESHOLD)}
        return ok, meta

    # ----------------- event collection with cooldown ----------------- #

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
                "timestamp": get_timestamp(),  # utils.get_timestamp() has no args
                "message": message,
                "meta": meta or {},
            })
            self._last_event_ts[ev_type] = now

    def get_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        ok, meta = self.is_fall_event()
        if ok:
            self._append_event(
                "fall_detected",
                "낙상 감지: ROI 외부에서 standing→lying 전이",
                events,
                severity="critical",
                meta=meta,
            )

        ok, meta = self.is_prone_event()
        if ok:
            self._append_event(
                "prone_warning",
                "엎드린 자세 감지: 호흡곤란 우려",
                events,
                severity="warning",
                meta=meta,
            )

        ok, meta = self.is_motionless_sustained()
        if ok:
            last = self.buffer[-1] if self.buffer else None
            in_roi_txt = ""
            if last and last.in_roi is False:
                in_roi_txt = "ROI 외부에서 "
            self._append_event(
                "danger_motionless",
                f"위험 무동작: {in_roi_txt}{NO_MOVEMENT_TIME_THRESHOLD:.0f}초 이상 움직임 없음",
                events,
                severity="critical",
                meta=meta,
            )

        ok, meta = self.is_irregular_event()
        if ok:
            self._append_event(
                "irregular_movement",
                "비정상적 자세 변화 빈번",
                events,
                severity="warning",
                meta=meta,
            )

        ok, meta = self.is_tilt_sustained()
        if ok:
            self._append_event(
                "tilt_sustained",
                f"기울임 상태 {TILT_DURATION:.0f}초 이상 지속",
                events,
                severity="info",
                meta=meta,
            )

        return events
