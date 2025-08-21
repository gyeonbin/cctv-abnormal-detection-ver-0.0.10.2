# core/bus.py
import queue
from dataclasses import dataclass

@dataclass
class Bus:
    meta_q: queue.Queue       # FrameMeta (디코더 → 배처)
    infer_out_q: queue.Queue  # (stream_id, fid, pts, img_src, img_yolo, scales, dets, confs)
    track_out_q: queue.Queue  # (stream_id, fid, pts, img_src, tags)

_bus = None

def get_bus():
    global _bus
    if _bus is None:
        _bus = Bus(
            meta_q=queue.Queue(maxsize=2048),
            infer_out_q=queue.Queue(maxsize=1024),
            track_out_q=queue.Queue(maxsize=1024),
        )
    return _bus
