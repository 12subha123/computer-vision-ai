import math


class GestureRecognizer:
    """
    Recognizes simple hand gestures from MediaPipe's 21 hand landmarks.

    Landmark indexes:
    0  = wrist
    4  = thumb tip
    8  = index finger tip
    12 = middle finger tip
    16 = ring finger tip
    20 = pinky tip
    """

    def _distance(self, a, b):
        return math.sqrt(
            (a.x - b.x) ** 2
            + (a.y - b.y) ** 2
            + (a.z - b.z) ** 2
        )

    def _finger_extended(self, landmarks, tip, pip):
        wrist = landmarks[0]

        tip_distance = self._distance(landmarks[tip], wrist)
        pip_distance = self._distance(landmarks[pip], wrist)

        return tip_distance > pip_distance * 1.15

    def recognize(self, landmarks, handedness="Right"):
        index_extended = self._finger_extended(
            landmarks, 8, 6
        )

        middle_extended = self._finger_extended(
            landmarks, 12, 10
        )

        ring_extended = self._finger_extended(
            landmarks, 16, 14
        )

        pinky_extended = self._finger_extended(
            landmarks, 20, 18
        )

        extended_count = sum(
            [
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended,
            ]
        )

        if extended_count == 0:
            return "Fist"

        if (
            index_extended
            and middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "Peace"

        if (
            index_extended
            and not middle_extended
            and not ring_extended
            and not pinky_extended
        ):
            return "Point"

        if extended_count == 4:
            return "Open Palm"

        return "Unknown"
