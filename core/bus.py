# core/bus.py
import queue
from dataclasses import dataclass
from typing import Tuple, Any, Queue  # typing에서 Queue를 가져옵니다.
from core.types import FrameMeta, InferOut

@dataclass
class Bus:
    """
    파이프라인 단계 간 데이터 전송을 위한 중앙 큐 허브.
    - meta_q: 디코더 -> 배처 (프레임 메타데이터)
    - infer_out_q: 배처 -> 트래커 (추론 결과 객체)
    - track_out_q: 트래커 -> 렌더러 (시각화용 태그 포함 최종 결과)
    """
    meta_q: Queue[FrameMeta]
    infer_out_q: Queue[InferOut]
    track_out_q: Queue[Tuple[str, int, float, Any, list]]  # (sid, fid, pts, img, tags)

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