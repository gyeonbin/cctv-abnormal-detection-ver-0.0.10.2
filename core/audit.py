import time

class ContinuityCounter:
    def __init__(self, name):
        self.name = name
        self.expected = None
        self.lost = 0
        self.dup = 0
        self.total = 0

    def seen(self, fid: int):
        self.total += 1
        if self.expected is None:
            self.expected = fid + 1
            return
        if fid == self.expected:
            self.expected += 1
        elif fid < self.expected:
            # 같은 프레임이 다시 들어오거나 역행(이상 징후)
            self.dup += 1
        else:
            # 건너뛴 개수만큼 누락
            self.lost += (fid - self.expected)
            self.expected = fid + 1

    def snapshot(self):
        return f"[AUDIT][{self.name}] total={self.total} lost={self.lost} dup={self.dup}"