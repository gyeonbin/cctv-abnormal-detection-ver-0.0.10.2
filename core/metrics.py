import time

# ===================== 유틸 =====================
class LoopTimer:
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        self._last = None
        self.ema = None
    def tick(self):
        now = time.perf_counter()
        if self._last is None:
            self._last = now
            return None, None
        delta = now - self._last
        self._last = now
        if delta <= 0:
            return None, self.ema
        inst = 1.0 / delta
        self.ema = inst if self.ema is None else (self.alpha * inst + (1 - self.alpha) * self.ema)
        return inst, self.ema