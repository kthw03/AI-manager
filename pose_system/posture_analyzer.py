# 시간 기반 자세 상태 분석 (tilting, motionless 등)

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

COOL_DOWN = 5.0  # 이벤트 쿨다운 시간 (초)

def _is_lying(label: str) -> bool:
    """lying 계열 레이블인지 검사."""
    return label.startswith("lying")


class PostureAnalyzerV4:
    """
    시간 기반 슬라이딩 윈도우로 posture/event 분석.
    - 내부 타이밍: time.monotonic()
    - 이벤트 타임스탬프: get_timestamp(time.time())
    """
    def __init__(self, roi_manager: Any = None):
        self.roi_manager = roi_manager
        self.buffer: Deque[AnalyzedFrame] = deque()
        self.last_label: Optional[str] = None

        # 이벤트 쿨다운 관리
        self._last_event_ts: Dict[str, float] = {}

        # 지속 이벤트 시작 시각 추적
        self._motionless_start: Optional[float] = None
        self._tilt_start: Optional[float] = None

    def update(
        self,
        label: str,
        landmarks: Optional[List[Tuple[float, float, float, float]]],
        bbox: Optional[Tuple[int,int,int,int]],
    ) -> None:
        """
        프레임 한 장의 분석 결과를 버퍼에 추가.
        - monotonic_ts: 내부 지속판정 시 사용
        - wall_ts: 이벤트 타임스탬프(log)에 사용
        """
        now_mon = time.monotonic()
        now_wall = time.time()

        # 어깨 y 평균
        sh_y = None
        if landmarks:
            sh_y = (landmarks[11][1] + landmarks[12][1]) / 2

        # ROI 안/밖 판정
        in_roi = True
        if self.roi_manager and bbox:
            in_roi = self.roi_manager.is_bbox_in_roi(bbox)

        # 슬라이딩 윈도우: 오래된 프레임 제거
        cutoff = now_mon - max(TILT_DURATION, NO_MOVEMENT_TIME_THRESHOLD)
        while self.buffer and self.buffer[0].monotonic_ts < cutoff:
            self.buffer.popleft()

        # 추가
        self.buffer.append(
            AnalyzedFrame(now_mon, now_wall, label, sh_y, landmarks, in_roi)
        )
        self.last_label = label

    def get_state(self) -> str:
        """
        통일된 기준으로 현재 상태 반환.
        - 연속 TILT_DURATION 이상 tilting → 'tilting'
        - 연속 NO_MOVEMENT_TIME_THRESHOLD 이상 motionless → 'motionless'
        - 그 외에는 마지막 posture 레이블
        """
        now = time.monotonic()

        # 1) tilt 지속 판정
        if self._check_tilt(now):
            return "tilting"

        # 2) motionless 지속 판정
        if self._check_motionless(now):
            return "motionless"

        return self.last_label or "unknown"

    # ────────────── 내부 지속판정 헬퍼 ────────────── #

    def _check_tilt(self, now: float) -> bool:
        """현재 윈도우 안에 연속으로 기울임이 TILT_DURATION 이상인지."""
        # 최신 프레임부터 역순으로 시간 차이 < TILT_DURATION인 부분만 검사
        ys = []
        for f in reversed(self.buffer):
            if now - f.monotonic_ts > TILT_DURATION:
                break
            if f.shoulder_y is not None:
                ys.append(f.shoulder_y)
        if not ys:
            self._tilt_start = None
            return False
        if (max(ys) - min(ys)) > ANALYZER_TILT_THRESHOLD:
            # 지속 시작 타임스탬프 설정
            if self._tilt_start is None:
                self._tilt_start = now
            return (now - self._tilt_start) >= TILT_DURATION
        else:
            self._tilt_start = None
            return False

    def _check_motionless(self, now: float) -> bool:
        """현재 윈도우 안에 연속으로 무동작이 NO_MOVEMENT_TIME_THRESHOLD 이상인지."""
        # 윈도우 안 프레임 추출
        frames = [f for f in self.buffer if now - f.monotonic_ts <= NO_MOVEMENT_TIME_THRESHOLD]
        if len(frames) < 2:
            self._motionless_start = None
            return False

        # 이동량 평균 계산
        total, count = 0.0, 0
        for prev, curr in zip(frames, frames[1:]):
            if not prev.landmarks or not curr.landmarks:
                continue
            for a, b in zip(prev.landmarks, curr.landmarks):
                total += calculate_euclidean_distance(a, b)
            count += 1
        if count == 0 or (total / count) >= ANALYZER_MOTION_THRESHOLD:
            self._motionless_start = None
            return False

        # 지속 시작 타임스탬프 설정
        if self._motionless_start is None:
            self._motionless_start = now
        return (now - self._motionless_start) >= NO_MOVEMENT_TIME_THRESHOLD

    def has_transition(self, from_label: str, to_label: str) -> bool:
        """
        직전 프레임부터 즉시 발생한 레이블 전이 판단 (1-frame).
        """
        if len(self.buffer) < 2:
            return False
        return (
            self.buffer[-2].label == from_label and
            self.buffer[-1].label == to_label
        )

    # ────────────── 이벤트 판단 메서드 ────────────── #

    def is_fall_detected(self) -> bool:
        """
        낙상 감지:
          • 마지막 프레임이 lying 계열 & ROI 밖
          • FALL_TRANSITION_TIME 이내 'standing'→'lying' 전이 (중간 'sitting' 無)
        """
        if not self.buffer:
            return False
        last = self.buffer[-1]
        if not _is_lying(last.label) or last.in_roi:
            return False

        # standing 인덱스 찾기
        stand_idx = None
        for i in range(len(self.buffer)-2, -1, -1):
            if self.buffer[i].label == "standing":
                stand_idx = i
                break
        if stand_idx is None:
            return False

        # 전이 시간
        if (last.monotonic_ts - self.buffer[stand_idx].monotonic_ts) > FALL_TRANSITION_TIME:
            return False

        # 중간 sitting 체크
        for f in list(self.buffer)[stand_idx+1:]:
            if f.label == "sitting":
                return False
        return True

    def is_prone_warning(self) -> bool:
        """
        엎드림(prone) 경고:
          • lying 계열
          • nose.z – hip.z < POSE_Z_PRONE_THRESHOLD
        """
        if not self.buffer:
            return False
        last = self.buffer[-1]
        if not _is_lying(last.label):
            return False
        lm = last.landmarks
        nose_z = lm[0][2]
        hip_z  = (lm[23][2] + lm[24][2]) / 2
        return (nose_z - hip_z) < POSE_Z_PRONE_THRESHOLD

    def is_irregular_movement(self) -> bool:
        """
        비정상적 움직임:
          • 윈도우 전체 posture 전이 횟수 ≥ ANALYZER_IRREGULAR_THRESHOLD
        """
        changes = 0
        prev = None
        for f in self.buffer:
            if prev is not None and f.label != prev:
                changes += 1
            prev = f.label
        return changes >= ANALYZER_IRREGULAR_THRESHOLD

    # ────────────── 이벤트 수집 & 쿨다운 ────────────── #

    def _append_event(
        self,
        ev_type: str,
        message: str,
        events: List[Dict[str, str]]
    ) -> None:
        """
        동일 이벤트 중복 방지를 위해 마지막 발생 시간과 쿨다운 검사 후 events에 추가.
        """
        now = time.monotonic()
        last = self._last_event_ts.get(ev_type, 0.0)
        if (now - last) >= COOL_DOWN:
            events.append({
                "type": ev_type,
                "timestamp": get_timestamp(time.time()),
                "message": message
            })
            self._last_event_ts[ev_type] = now

    def get_events(self) -> List[Dict[str, str]]:
        """
        감지된 이상 이벤트들을 쿨다운 적용해 리스트로 반환.
        """
        events: List[Dict[str, str]] = []

        if self.is_fall_detected():
            self._append_event(
                "fall_detected",
                "낙상 감지: ROI 외부에서 빠른 standing→lying 전이",
                events
            )
        if self.is_prone_warning():
            self._append_event(
                "prone_warning",
                "엎드린 자세 감지: 호흡곤란 우려",
                events
            )
        if self._check_motionless(time.monotonic()):
            self._append_event(
                "danger_motionless",
                f"위험 무동작: ROI 외부에서 {NO_MOVEMENT_TIME_THRESHOLD:.0f}초 이상 움직임 없음",
                events
            )
        if self.is_irregular_movement():
            self._append_event(
                "irregular_movement",
                "비정상적 자세 변화 빈번",
                events
            )
        # tilt 지속 이벤트
        if self._check_tilt(time.monotonic()):
            self._append_event(
                "tilt_sustained",
                f"기울임 상태 {TILT_DURATION:.0f}초 이상 지속",
                events
            )

        return events
