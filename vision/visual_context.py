class VisualContextBuilder:
    def build(self, vision_results, caption=None, classification=None, question=None):
        context = {
            "faces": 0,
            "hands": 0,
            "poses": 0,
            "objects": [],
            "caption": caption or "",
            "classification": classification or [],
            "question": question or "",
        }

        if vision_results:
            faces = vision_results.get("faces")
            hands = vision_results.get("hands")
            poses = vision_results.get("poses")
            objects = vision_results.get("objects")

            context["faces"] = self._count_faces(faces)
            context["hands"] = self._count_hands(hands)
            context["poses"] = self._count_poses(poses)

            if objects:
                context["objects"] = self._extract_objects(objects)

        return context

    def _count_faces(self, value):
        if value is None:
            return 0

        if hasattr(value, "face_landmarks"):
            return len(value.face_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def _count_hands(self, value):
        if value is None:
            return 0

        if hasattr(value, "hand_landmarks"):
            return len(value.hand_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def _count_poses(self, value):
        if value is None:
            return 0

        if hasattr(value, "pose_landmarks"):
            return len(value.pose_landmarks)

        return len(value) if hasattr(value, "__len__") else 0

    def _extract_objects(self, objects):
        extracted = []

        for obj in objects:
            if isinstance(obj, dict):
                name = obj.get("class_name") or obj.get("name") or obj.get("label")

                if name:
                    extracted.append(name)

            elif isinstance(obj, str):
                extracted.append(obj)

        return extracted

    def to_prompt(self, context):
        lines = [
            "Visual information:",
            f"- Faces detected: {context['faces']}",
            f"- Hands detected: {context['hands']}",
            f"- Poses detected: {context['poses']}",
        ]

        if context["objects"]:
            lines.append(
                "- Objects detected: " +
                ", ".join(context["objects"])
            )

        if context["caption"]:
            lines.append(
                f"- Image caption: {context['caption']}"
            )

        if context["classification"]:
            lines.append(
                f"- Classification: {context['classification']}"
            )

        if context["question"]:
            lines.append(
                f"- User question: {context['question']}"
            )

        return "\n".join(lines)
