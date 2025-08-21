# -*- coding: utf-8 -*-
"""
Batched SHM-style CSV logger with **backward-compatible API**
- Accepts old constructor: ShmCsvProducer(name=..., nslots=..., slot_size=...)
- Non-blocking: producer never waits; overwrites oldest FULL slot when ring is full
- Works even without a separate log server/consumer (records will cycle in-memory)

If you have a real SHM ring, wrap it with RingAdapter and pass via `ring=`.
Otherwise, this module's DummyRing stores slots in-process (fast, safe fallback).
"""
from __future__ import annotations
import struct
from time import perf_counter
from typing import Callable, List, Tuple, Optional, Iterable

# ---------------------- Record/Header formats ----------------------
REC_FMT  = struct.Struct("<q d q d")   # (seq:int64, t_wall:double, fid:int64, pts:double) = 32 bytes
HDR_FMT  = struct.Struct("<I I q")     # (count:uint32, reserved:uint32, seq_base:int64) = 16 bytes
REC_SIZE = REC_FMT.size
HDR_SIZE = HDR_FMT.size

# ---------------------- Slot flags ----------------------
VALID_EMPTY   = 0
VALID_FILLING = 1
VALID_FULL    = 2

# ---------------------- Ring interface + adapters ----------------------
class ShmRingIface:
    def __init__(self, nslots: int, slot_size: int):
        self.nslots = int(nslots)
        self.slot_size = int(slot_size)
    def _rec_ptr(self, slot: int) -> memoryview:  # legacy-style
        raise NotImplementedError
    def _flag_ptr(self, slot: int) -> memoryview:  # 1 byte view
        raise NotImplementedError
    def _get_widx(self) -> int:
        raise NotImplementedError
    def _set_widx(self, v: int) -> None:
        raise NotImplementedError
    def _get_ridx(self) -> int:
        raise NotImplementedError
    def _set_ridx(self, v: int) -> None:
        raise NotImplementedError

class DummyRing(ShmRingIface):
    """In-process ring for compatibility when real SHM isn't required/present."""
    def __init__(self, nslots: int, slot_size: int):
        super().__init__(nslots, slot_size)
        self._slots = [bytearray(self.slot_size) for _ in range(self.nslots)]
        self._flags = bytearray([VALID_EMPTY] * self.nslots)
        self._widx = 0
        self._ridx = 0
    def _rec_ptr(self, slot: int) -> memoryview:
        return memoryview(self._slots[slot])
    def _flag_ptr(self, slot: int) -> memoryview:
        return memoryview(self._flags[slot:slot+1])
    def _get_widx(self) -> int: return self._widx
    def _set_widx(self, v: int) -> None: self._widx = int(v) % self.nslots
    def _get_ridx(self) -> int: return self._ridx
    def _set_ridx(self, v: int) -> None: self._ridx = int(v) % self.nslots

class RingAdapter:
    """Adapter exposing minimal methods used by the batched producer/consumer."""
    def __init__(self, ring: ShmRingIface):
        self._ring = ring
        self.nslots = ring.nslots
        self.slot_size = ring.slot_size
    def rec_ptr(self, slot: int) -> memoryview:
        return self._ring._rec_ptr(slot)
    def flag(self, slot: int) -> int:
        return int(self._ring._flag_ptr(slot)[0])
    def set_flag(self, slot: int, val: int) -> None:
        self._ring._flag_ptr(slot)[:] = bytes([val])
    def get_widx(self) -> int:
        return int(self._ring._get_widx())
    def set_widx(self, v: int) -> None:
        self._ring._set_widx(int(v))
    def get_ridx(self) -> int:
        return int(self._ring._get_ridx())
    def set_ridx(self, v: int) -> None:
        self._ring._set_ridx(int(v))

# ---------------------- Batched Producer ----------------------
class ShmCsvProducerBatch:
    def __init__(
        self,
        ring: RingAdapter,
        *,
        batch: int = 128,
        max_delay_sec: float = 0.020,
        overwrite_when_full: bool = True,
    ):

        self.ring = ring
        self.batch = int(batch)
        self.max_delay_sec = float(max_delay_sec)
        self.overwrite_when_full = bool(overwrite_when_full)

        self.wslot = 0
        self.count_in_slot = 0
        self.seq_base = 0
        self.slot_t0 = 0.0
        self.payload_mv: Optional[memoryview] = None

        need = HDR_SIZE + self.batch * REC_SIZE
        if self.ring.slot_size < need:
            raise ValueError(f"ring.slot_size {self.ring.slot_size} < required {need}")

    def _begin_slot(self, seq0: int) -> None:
        s = self.wslot
        f = self.ring.flag(s)
        if f == VALID_FULL:
            if self.overwrite_when_full:
                self.ring.set_flag(s, VALID_EMPTY)
            else:
                return  # drop
        self.ring.set_flag(s, VALID_FILLING)
        self.count_in_slot = 0
        self.seq_base = int(seq0)
        self.slot_t0 = perf_counter()
        slot_view = self.ring.rec_ptr(s)
        self.payload_mv = memoryview(slot_view)[HDR_SIZE:HDR_SIZE + self.batch * REC_SIZE]

    def _commit_slot(self) -> None:
        s = self.wslot
        if self.count_in_slot <= 0:
            self.ring.set_flag(s, VALID_EMPTY)
        else:
            hdr = HDR_FMT.pack(self.count_in_slot, 0, int(self.seq_base))
            slot_view = self.ring.rec_ptr(s)
            slot_view[:HDR_SIZE] = hdr
            self.ring.set_flag(s, VALID_FULL)
        self.wslot = (self.wslot + 1) % self.ring.nslots
        self.count_in_slot = 0
        self.payload_mv = None

    def write_record(self, seq: int, t_wall: float, fid: int, pts: float) -> None:
        if self.count_in_slot == 0 or self.payload_mv is None:
            self._begin_slot(seq0=seq)
            if self.payload_mv is None:
                return
        off = self.count_in_slot * REC_SIZE
        REC_FMT.pack_into(self.payload_mv, off, int(seq), float(t_wall), int(fid), float(pts))
        self.count_in_slot += 1
        if self.count_in_slot >= self.batch or (perf_counter() - self.slot_t0) >= self.max_delay_sec:
            self._commit_slot()

    def flush(self) -> None:
        if self.count_in_slot > 0:
            self._commit_slot()

# ---------------------- Consumer ----------------------
class ShmCsvConsumerBatch:
    def __init__(self, ring: RingAdapter, process_batch: Callable[[List[Tuple[int, float, int, float]]], None]):
        self.ring = ring
        self.rslot = 0
        self.process_batch = process_batch
    def poll_once(self) -> bool:
        s = self.rslot
        # We check raw flag via adapter
        # Note: for DummyRing-only setups without external consumer, nothing will ever be FULL
        # unless producer is active.
        # A real SHM reader would see FULL set by another process.
        if RingAdapter.flag.__get__(self.ring, RingAdapter)(s) != VALID_FULL:
            return False
        slot_view = self.ring.rec_ptr(s)
        count, _resv, _seq_base = HDR_FMT.unpack(slot_view[:HDR_SIZE])
        payload = memoryview(slot_view)[HDR_SIZE:HDR_SIZE + count * REC_SIZE]
        recs: List[Tuple[int, float, int, float]] = []
        off = 0
        for _ in range(count):
            seq, t_wall, fid, pts = REC_FMT.unpack_from(payload, off)
            recs.append((int(seq), float(t_wall), int(fid), float(pts)))
            off += REC_SIZE
        self.process_batch(recs)
        self.ring.set_flag(s, VALID_EMPTY)
        self.rslot = (self.rslot + 1) % self.ring.nslots
        return True
    def drain(self) -> int:
        n = 0
        while self.poll_once():
            n += 1
        return n

# ---------------------- CSV sink (batched flush) ----------------------
class CsvBatchSink:
    def __init__(self, fh=None, flush_interval: float = 0.5):
        self.fh = fh
        self.buf: List[str] = []
        self.last = perf_counter()
        self.interval = float(flush_interval)
    def consume(self, recs: Iterable[Tuple[int, float, int, float]]):
        for (seq, t_wall, fid, pts) in recs:
            self.buf.append(f"{seq},{t_wall:.6f},{fid},{pts:.6f}")
        now = perf_counter()
        if (now - self.last) >= self.interval and self.buf:
            data = "\n".join(self.buf) + "\n"
            if self.fh is not None:
                self.fh.write(data)
                self.fh.flush()
            else:
                print(data, end="")
            self.buf.clear()
            self.last = now

# ---------------------- Backward-compatible Producer ----------------------
class ShmCsvProducer:
    """
    Back-compat constructor:
        ShmCsvProducer(name=..., nslots=32, slot_size=None, batch=128, max_delay_sec=0.02, overwrite_when_full=True, ring=None)
    Old write API (observed in legacy code):
        .write((t_wall, fid, pts))  # seq is internal counter
    Also exposes:
        .write_record(seq, t_wall, fid, pts)
        .flush()
    """
    def __init__(self, *args, **kwargs):
        # Extract known kwargs with defaults
        name = kwargs.pop('name', None)  # accepted but unused in DummyRing
        ring = kwargs.pop('ring', None)
        nslots = int(kwargs.pop('nslots', 1024))
        batch = int(kwargs.pop('batch', 128))
        max_delay_sec = float(kwargs.pop('max_delay_sec', 0.02))
        overwrite_when_full = bool(kwargs.pop('overwrite_when_full', True))
        slot_size = kwargs.pop('slot_size', None)
        # Allow positional (rare) : (ring,) or ()
        if args:
            # If first arg looks like a ring, use it
            maybe_ring = args[0]
            if hasattr(maybe_ring, '_rec_ptr') and hasattr(maybe_ring, '_flag_ptr'):
                ring = maybe_ring
        need_slot = HDR_SIZE + batch * REC_SIZE
        if slot_size is None:
            slot_size = need_slot
        if ring is None:
            ring = DummyRing(nslots=nslots, slot_size=slot_size)
        else:
            # Ensure slot size is large enough
            if getattr(ring, 'slot_size', need_slot) < need_slot:
                raise ValueError(f"ring.slot_size {getattr(ring, 'slot_size', '<?>')} < required {need_slot}")
        self._ring = ring
        self._adapter = RingAdapter(ring)
        self._prod = ShmCsvProducerBatch(self._adapter, batch=batch, max_delay_sec=max_delay_sec,
                                         overwrite_when_full=overwrite_when_full)
        self._seq = 0

    # Legacy-style API
    def write(self, values: Tuple[float, int, float]):
        t_wall, fid, pts = values
        self._seq += 1
        self._prod.write_record(self._seq, float(t_wall), int(fid), float(pts))

    # Modern explicit API
    def write_record(self, seq: int, t_wall: float, fid: int, pts: float):
        self._prod.write_record(int(seq), float(t_wall), int(fid), float(pts))

    def flush(self):
        self._prod.flush()

# Keep names for external imports
ShmCsvProducerBatchAdapter = ShmCsvProducerBatch
ShmCsvConsumer = ShmCsvConsumerBatch
ShmCsvProducerCompat = ShmCsvProducer

# Optional (not provided here): run_shm_csv_server
# If some modules import it optionally, they should guard for ImportError or None.
