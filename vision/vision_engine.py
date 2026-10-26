from vision.face_landmarks import FaceLandmarker
from vision.hand_landmarks import HandLandmarker
from vision.pose_landmarks import PoseLandmarker


class VisionEngine:
    def __init__(self):
        self.face = FaceLandmarker()
        self.hand = HandLandmarker()
        self.pose = PoseLandmarker()

    def process(self, frame):
        return {
            "face": self.face.detect(frame),
            "hand": self.hand.detect(frame),
            "pose": self.pose.detect(frame),
        }

    def draw(self, frame, results):
        output = frame.copy()

        face_results = results.get("face")
        hand_results = results.get("hand")
        pose_results = results.get("pose")

        if face_results is not None:
            output = self.face.draw(
                output,
                face_results,
            )

        if hand_results is not None:
            output = self.hand.draw(
                output,
                hand_results,
            )

        if pose_results is not None:
            output = self.pose.draw(
                output,
                pose_results,
            )

        return output

    def get_summary(self, results):
        face_results = results.get("face")
        hand_results = results.get("hand")
        pose_results = results.get("pose")

        faces = (
            len(face_results.face_landmarks)
            if face_results is not None
            and hasattr(face_results, "face_landmarks")
            else 0
        )

        hands = (
            len(hand_results.hand_landmarks)
            if hand_results is not None
            and hasattr(hand_results, "hand_landmarks")
            else 0
        )

        poses = (
            len(pose_results.pose_landmarks)
            if pose_results is not None
            and hasattr(pose_results, "pose_landmarks")
            else 0
        )

        return {
            "faces": faces,
            "hands": hands,
            "poses": poses,
        }

    def close(self):
        self.face.close()
        self.hand.close()
        self.pose.close()
