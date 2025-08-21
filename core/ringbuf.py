# core/ringbuf.py
import numpy as np
from multiprocessing import shared_memory as shm

# ===================== 공유메모리 링버퍼 =====================
class SharedFrameRing:
    def __init__(self, name: str, n_slots: int, h: int, w: int, c: int = 3, create: bool = False):
        self.name = name
        self.n_slots = int(n_slots)
        self.h = int(h)
        self.w = int(w)
        self.c = int(c)
        self.dtype = np.uint8
        self.frame_bytes = self.h * self.w * self.c
        self.nbytes = self.n_slots * self.frame_bytes

        if create:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.name)
        else:
            self.shm = shm.SharedMemory(name=self.name, create=False)

        self._np = np.ndarray((self.n_slots, self.h, self.w, self.c), dtype=self.dtype, buffer=self.shm.buf)

    def write(self, slot: int, img: np.ndarray):
        if img.shape[0] != self.h or img.shape[1] != self.w or img.shape[2] != self.c or img.dtype != self.dtype:
            raise ValueError("SharedFrameRing.write: shape/dtype mismatch")
        self._np[slot, :, :, :] = img

    def read_copy(self, slot: int) -> np.ndarray:
        return self._np[slot].copy()

    def close(self):
        try: self.shm.close()
        except Exception: pass

    def unlink(self):
        try: self.shm.unlink()
        except Exception: pass