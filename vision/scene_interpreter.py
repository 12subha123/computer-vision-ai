class SceneInterpreter:
    def describe(self, results):
        parts = []

        faces = results.get("faces")
        hands = results.get("hands")
        poses = results.get("poses")
        objects = results.get("objects", [])

        face_count = (
            len(faces.face_landmarks)
            if faces is not None and hasattr(faces, "face_landmarks")
            else 0
        )

        hand_count = (
            len(hands.hand_landmarks)
            if hands is not None and hasattr(hands, "hand_landmarks")
            else 0
        )

        pose_count = (
            len(poses.pose_landmarks)
            if poses is not None and hasattr(poses, "pose_landmarks")
            else 0
        )

        if face_count:
            parts.append(
                f"{face_count} face{'s' if face_count != 1 else ''} detected"
            )

        if hand_count:
            parts.append(
                f"{hand_count} hand{'s' if hand_count != 1 else ''} detected"
            )

        if pose_count:
            parts.append(
                f"{pose_count} pose{'s' if pose_count != 1 else ''} detected"
            )

        for obj in objects:
            label = obj["label"]
            confidence = obj["confidence"]

            parts.append(
                f"{label} ({confidence * 100:.1f}%)"
            )

        if not parts:
            return "No recognizable elements detected."

        return ", ".join(parts) + "."


    def summarize(self, results):
        description = self.describe(results)

        return {
            "description": description,
            "object_count": len(results.get("objects", [])),
        }
