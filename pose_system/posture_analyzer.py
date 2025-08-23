# posture_analyzer.py
import time
from collections import deque, namedtuple
from typing import Deque, List, Dict, Optional, Tuple, Any, TYPE_CHECKING

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

# 타입 힌트 전용 임포트(실행 시 순환참조 방지)
if TYPE_CHECKING:
    from roi_manager import ROIManager

# 분석 버퍼에 저장할 프레임 구조
AnalyzedFrame = namedtuple(
    "AnalyzedFrame",
    ["monotonic_ts", "wall_ts", "label", "shoulder_y", "landmarks", "in_roi"]
)

# 동일 이벤트 재발행 쿨다운(초)
COOL_DOWN = 5.0

# 이벤트 메타 필드(출력 형식 통일)
META_KEYS = ("in_roi", "duration_sec", "changes", "threshold", "nose_minus_hip_z")

# === 신규 규칙 임계값 ===
STANDING_TILT_DURATION = 1.0          # standing_tilt 1초 이상 → falling_warning
OFFROI_LOWPOSTURE_DURATION = 1.0      # ROI 밖에서 sitting/lying 1초 이상 → falling_detect
NO_PERSON_DURATION = 1.0              # 1초 이상 사람 미검출 → patient_escape

# === 서있음 무동작(추가 제안) ===
STANDING_FREEZE_DURATION = 10.0       # 서 있음에서 무동작 지속 시간
STANDING_MAJORITY_RATIO = 0.70        # 윈도우 내 standing 비율 임계


def _is_lying(label: str) -> bool:
    """lying 계열 레이블인지 검사."""
    return label.startswith("lying")


def _is_low_posture(label: str) -> bool:
    """앉음 또는 누움 전체를 '낮은 자세'로 간주."""
    return (label == "sitting") or _is_lying(label)


class PostureAnalyzerV4:
    """
    시간 기반 슬라이딩 윈도우로 자세/이벤트를 분석하는 클래스.
    - 내부 타이밍: time.monotonic() (드리프트 없는 상대 시계)
    - 이벤트 타임스탬프: get_timestamp() (벽시계 문자열)
    - ROI는 외부에서 ROIManager 인스턴스를 '주입'(DI) 받아 사용
    """

    def __init__(self, roi_manager: "Optional[ROIManager]" = None):
        # ROI 포함 판정에 사용할 매니저(없어도 동작)
        self.roi_manager = roi_manager
        # 최근 프레임 슬라이딩 버퍼
        self.buffer: Deque[AnalyzedFrame] = deque()
        # 마지막 분류 라벨(편의상 저장)
        self.last_label: Optional[str] = None

        # 이벤트별 마지막 발행 시각(쿨다운 관리)
        self._last_event_ts: Dict[str, float] = {}
        # 무동작/기울기 상태 시작 시각
        self._motionless_start: Optional[float] = None
        self._tilt_start: Optional[float] = None

    def update(
        self,
        label: str,
        landmarks: Optional[List[Tuple[float, float, float, float]]],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """
        한 프레임 단위로 분석 버퍼를 갱신.
        - label: 분류기 결과(standing/sitting/lying*/standing_tilt/no_person 등)
        - landmarks: 포즈 키포인트 목록(없을 수 있음)
        - bbox: 사람 바운딩 박스(x1,y1,x2,y2)
        """
        now_mon = time.monotonic()
        now_wall = time.time()

        # 양쪽 어깨 y의 평균(기울기 탐지에 사용)
        sh_y = None
        if landmarks and len(landmarks) > 12:
            try:
                sh_y = (landmarks[11][1] + landmarks[12][1]) / 2.0
            except Exception:
                sh_y = None

        # ROI 포함 여부(in_roi) 계산: ROI가 실제 존재할 때만 검사(없으면 의미 없으므로 True)
        in_roi = True
        if self.roi_manager and bbox is not None:
            try:
                if self._has_roi():
                    in_roi = self.roi_manager.is_bbox_in_roi(bbox)
                else:
                    in_roi = True
            except Exception:
                in_roi = True

        # 윈도우 유지: 사용되는 모든 임계시간 중 최대치만큼 보존
        cutoff = now_mon - max(
            TILT_DURATION,
            NO_MOVEMENT_TIME_THRESHOLD,
            STANDING_TILT_DURATION,
            OFFROI_LOWPOSTURE_DURATION,
            NO_PERSON_DURATION,
            STANDING_FREEZE_DURATION,
        )
        while self.buffer and self.buffer[0].monotonic_ts < cutoff:
            self.buffer.popleft()

        # 버퍼에 프레임 추가
        self.buffer.append(
            AnalyzedFrame(now_mon, now_wall, label, sh_y, landmarks, in_roi)
        )
        self.last_label = label

    # ----------------- 상태 조회 ----------------- #
    def get_state(self) -> str:
        """현재 상태 요약(tilting/motionless/마지막 분류 라벨)."""
        now = time.monotonic()
        if self._check_tilt(now):
            return "tilting"
        if self._check_motionless(now):
            return "motionless"
        return self.last_label or "unknown"

    # ----------------- 지속 조건 체크 ----------------- #
    def _check_tilt(self, now: float) -> bool:
        """어깨 y 변화량 기반 기울기 지속 여부."""
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
        """키포인트 평균 이동량이 임계 미만 상태가 일정 시간 지속되는지 확인."""
        frames = [f for f in self.buffer if now - f.monotonic_ts <= NO_MOVEMENT_TIME_THRESHOLD]
        if len(frames) < 2:
            self._motionless_start = None
            return False

        # 간단한 point 래퍼(유클리디안 거리 함수 호환용)
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
        """직전 프레임에서 특정 라벨→현재 라벨로 전이가 있었는지 확인."""
        if len(self.buffer) < 2:
            return False
        return self.buffer[-2].label == from_label and self.buffer[-1].label == to_label

    # ----------------- 윈도우/ROI 헬퍼 ----------------- #
    def _window(self, duration: float) -> List[AnalyzedFrame]:
        """최근 duration초 구간의 프레임 집합을 반환."""
        now = time.monotonic()
        return [f for f in self.buffer if (now - f.monotonic_ts) <= duration]

    def _has_roi(self) -> bool:
        """
        ROI 보유 여부 확인:
        - bbox 기반(get_rois()/rois)에 1개 이상 있거나
        - 사다리꼴 등 수동 폴리곤(manual_polygons)이 1개 이상이면 True
        """
        if not self.roi_manager:
            return False
        try:
            # 1) bbox 기반(자동+수동 사각형)
            if hasattr(self.roi_manager, "get_rois"):
                rois = self.roi_manager.get_rois()
            else:
                rois = getattr(self.roi_manager, "rois", None)
            if rois and len(rois) > 0:
                return True

            # 2) 폴리곤 기반(수동 사다리꼴 ROI)
            polys = getattr(self.roi_manager, "manual_polygons", None)
            if polys and len(polys) > 0:
                return True

            return False
        except Exception:
            return False

    # ----------------- 이벤트 규칙 ----------------- #
    def is_falling_warning(self) -> Tuple[bool, Dict[str, Any]]:
        """
        낙상 경고:
        - standing_tilt가 STANDING_TILT_DURATION 이상 연속
        """
        frames = self._window(STANDING_TILT_DURATION)
        ok = bool(frames) and all(f.label == "standing_tilt" for f in frames)
        meta = self._meta_template(duration_sec=float(STANDING_TILT_DURATION))
        return ok, meta

    def is_falling_detect(self) -> Tuple[bool, Dict[str, Any]]:
        """
        낙상 감지(요구사항 반영):
        1) ROI가 없음(침대/의자 미검출) → OFFROI_LOWPOSTURE_DURATION 동안 sitting/lying이면 낙상
        2) ROI가 있음 → OFFROI_LOWPOSTURE_DURATION 동안 ROI '밖'에서 sitting/lying이면 낙상
        3) ROI '안'에서 sitting/lying이면 정상
        """
        frames = self._window(OFFROI_LOWPOSTURE_DURATION)
        meta = self._meta_template(duration_sec=float(OFFROI_LOWPOSTURE_DURATION))

        if not frames:
            return False, meta

        has_roi = self._has_roi()

        if not has_roi:
            # ROI 미검출: 자세만으로 판단
            ok = all(_is_low_posture(f.label) for f in frames)
            meta = self._meta_template(
                in_roi=None,  # ROI가 없으므로 판정 불가
                duration_sec=float(OFFROI_LOWPOSTURE_DURATION),
            )
            return ok, meta

        # ROI 보유: ROI 밖 + 낮은 자세가 지속될 때만 낙상
        ok = all((not f.in_roi) and _is_low_posture(f.label) for f in frames)
        meta = self._meta_template(
            in_roi=(frames[-1].in_roi if frames else None),
            duration_sec=float(OFFROI_LOWPOSTURE_DURATION),
        )
        return ok, meta

    def is_patient_escape(self) -> Tuple[bool, Dict[str, Any]]:
        """
        환자 이탈:
        - 라벨이 'no_person'으로 1초 이상 지속
        """
        frames = self._window(NO_PERSON_DURATION)
        if not frames:
            return False, self._meta_template(duration_sec=float(NO_PERSON_DURATION))
        ok = all((f.label == "no_person") for f in frames)
        meta = self._meta_template(duration_sec=float(NO_PERSON_DURATION))
        return ok, meta

    def is_standing_freeze(self) -> Tuple[bool, Dict[str, Any]]:
        """
        서있음 무동작 경고:
        - 윈도우(STANDING_FREEZE_DURATION) 내 standing 비율 ≥ STANDING_MAJORITY_RATIO
        - 평균 움직임 < ANALYZER_MOTION_THRESHOLD
        - standing_tilt 경고 중이면 중복 경고 방지로 제외
        """
        now = time.monotonic()
        frames = [f for f in self.buffer if now - f.monotonic_ts <= STANDING_FREEZE_DURATION]
        meta = self._meta_template(duration_sec=float(STANDING_FREEZE_DURATION))

        if len(frames) < 3:
            return False, meta

        # falling_warning 우선
        ok_fw, _ = self.is_falling_warning()
        if ok_fw:
            return False, meta

        # standing 비율 체크(standing_tilt는 제외)
        standing_count = sum(1 for f in frames if f.label == "standing")
        if (standing_count / len(frames)) < STANDING_MAJORITY_RATIO:
            return False, meta

        # 평균 움직임 산출
        class P:
            __slots__ = ("x", "y")
            def __init__(self, x: float, y: float):
                self.x, self.y = x, y

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
            return False, meta

        avg_motion = total / pair_count
        if avg_motion >= ANALYZER_MOTION_THRESHOLD:
            return False, meta

        return True, meta

    # ----------------- 구(레거시) 규칙(보존) ----------------- #
    def is_fall_detected(self) -> bool:
        """
        과거 규칙(standing→lying 전이 기반). 현재 get_events()에서는 미사용이지만 보존.
        - 마지막 라벨이 lying이고 ROI 밖이어야 함
        - standing→lying 전이 시간이 FALL_TRANSITION_TIME 이내
        - 중간에 sitting이 끼면 낙상으로 보지 않음
        """
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
        """
        엎드린 자세 경고(미사용 유지):
        - lying 상태에서 코(z) - 엉덩이(z) 차이가 임계 미만
        """
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
        """
        라벨 변화 횟수 기반의 불규칙 움직임(미사용 유지):
        - 변경 횟수 ≥ ANALYZER_IRREGULAR_THRESHOLD
        """
        changes = 0
        prev = None
        for f in self.buffer:
            if prev is not None and f.label != prev:
                changes += 1
            prev = f.label
        return changes >= ANALYZER_IRREGULAR_THRESHOLD

    # ----------------- meta 유틸 ----------------- #
    def _meta_template(self, base: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """메타 필드 스키마에 맞춰 기본값(None)으로 초기화 후 부분 덮어쓰기."""
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
        """메타 입력을 스키마에 맞게 정규화."""
        return self._meta_template(meta or {})

    # ----------------- 이벤트 수집/발행 ----------------- #
    def _append_event(
        self,
        ev_type: str,
        message: str,
        events: List[Dict[str, Any]],
        severity: str = "warning",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        쿨다운을 고려해 이벤트 리스트에 1건을 추가.
        - ev_type별 마지막 발행 이후 COOL_DOWN 이상 경과했을 때만 추가
        """
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
        요구사항 3가지 이벤트 + standing_freeze 경고를 반환.
        출력 형식(type/severity/timestamp/message/meta)은 통일 유지.
        """
        events: List[Dict[str, Any]] = []

        # 1) falling_warning
        ok, meta = self.is_falling_warning()
        if ok:
            self._append_event(
                "falling_warning",
                "낙상 경고: standing_tilt 1초 이상",
                events,
                severity="warning",
                meta=meta,
            )

        # 2) falling_detect (ROI 유무 규칙 반영)
        ok, meta = self.is_falling_detect()
        if ok:
            self._append_event(
                "falling_detect",
                "낙상 감지: ROI 유무 규칙 기반 sitting/lying",
                events,
                severity="critical",
                meta=meta,
            )

        # 3) patient_escape
        ok, meta = self.is_patient_escape()
        if ok:
            self._append_event(
                "patient_escape",
                "환자 이탈: 1초 이상 사람 미검출",
                events,
                severity="critical",
                meta=meta,
            )

        # 4) standing_freeze
        ok, meta = self.is_standing_freeze()
        if ok:
            self._append_event(
                "standing_freeze",
                f"서 있는 상태에서 {STANDING_FREEZE_DURATION:.0f}초 이상 움직임 없음",
                events,
                severity="warning",
                meta=meta,
            )

        return events
