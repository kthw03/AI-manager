# posture_wrapper.py
# posture_classifier 보완 코드(측면 자세, 정확도 등)

from collections import deque, Counter, namedtuple
from posture_classifier import PostureClassifierV6  # ✅ V6 분류기 사용
from utils import calculate_angle

Point = namedtuple("Point", ["x", "y"])

class SlidingWindow:
    def __init__(self, size=5):
        self.size = size
        self.buffer = deque()

    def add(self, label):
        self.buffer.append(label)
        if len(self.buffer) > self.size:
            self.buffer.popleft()

    def get_majority(self):
        if not self.buffer:
            return "unknown"
        count = Counter(self.buffer)
        return count.most_common(1)[0][0]

class PostureClassifierWrapper:
    def __init__(self, window_size=5, visibility_threshold=0.5):
        self.primary = PostureClassifierV6()  # ✅ V6 적용
        self.window = SlidingWindow(window_size)
        self.visibility_threshold = visibility_threshold

    def average_visibility(self, landmarks):
        return sum([lm[3] for lm in landmarks]) / len(landmarks) if landmarks else 0.0

    def determine_view_side(self, landmarks):
        left_idxs = [11, 23, 25, 27]
        right_idxs = [12, 24, 26, 28]
        left_vis = sum([landmarks[i][3] for i in left_idxs]) / len(left_idxs)
        right_vis = sum([landmarks[i][3] for i in right_idxs]) / len(right_idxs)

        if left_vis > 0.6 and right_vis < 0.3:
            return "left_side_view"
        elif right_vis > 0.6 and left_vis < 0.3:
            return "right_side_view"
        elif left_vis > 0.6 and right_vis > 0.6:
            return "front_view"
        else:
            return "uncertain"

    def side_posture(self, landmarks, side="right"):
        try:
            if side == "right":
                sh, hip, knee, ankle = [Point(*landmarks[i][:2]) for i in [12, 24, 26, 28]]
            else:
                sh, hip, knee, ankle = [Point(*landmarks[i][:2]) for i in [11, 23, 25, 27]]

            y_diff = hip.y - knee.y
            if 0 < y_diff < 0.15:
                return "sitting"
            if knee.y < hip.y:
                return "kneeling"

            hk_angle = calculate_angle(hip, knee, ankle)
            sh_angle = calculate_angle(sh, hip, knee)
            if hk_angle > 160 and sh_angle > 150:
                return "standing"
        except:
            pass
        return "irregular"

    def classify(self, landmarks):
        avg_vis = self.average_visibility(landmarks)
        view = self.determine_view_side(landmarks)
        print(f"[view]: {view} | avg_vis: {avg_vis:.2f}")

        if avg_vis < self.visibility_threshold:
            if view == "right_side_view":
                label = self.side_posture(landmarks, side="right")
            elif view == "left_side_view":
                label = self.side_posture(landmarks, side="left")
            else:
                label = "irregular"
        else:
            label = self.primary.classify(landmarks)

        self.window.add(label)
        return self.window.get_majority()
