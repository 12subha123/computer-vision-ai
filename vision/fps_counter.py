import time


class FPSCounter:
    def __init__(self):
        self.previous_time = time.perf_counter()
        self.fps = 0.0

    def update(self):
        current_time = time.perf_counter()
        elapsed = current_time - self.previous_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed

            self.fps = (
                self.fps * 0.9
                + instant_fps * 0.1
            )

        self.previous_time = current_time

        return self.fps

    def get_fps(self):
        return self.fps
