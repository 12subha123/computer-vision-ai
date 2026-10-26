class LocalVisionFallback:
    def answer(self, context):
        faces = context.get("faces", 0)
        hands = context.get("hands", 0)
        poses = context.get("poses", 0)
        objects = context.get("objects", [])
        caption = context.get("caption", "")

        if caption:
            return caption

        parts = []

        if faces:
            parts.append(
                f"{faces} face{'s' if faces != 1 else ''}"
            )

        if hands:
            parts.append(
                f"{hands} hand{'s' if hands != 1 else ''}"
            )

        if poses:
            parts.append(
                f"{poses} pose{'s' if poses != 1 else ''}"
            )

        if objects:
            parts.append(
                "objects: " + ", ".join(objects)
            )

        if not parts:
            return "I could not identify enough visual information to answer."

        return "Visual analysis detected " + "; ".join(parts) + "."
