# core/bus.py
import queue
import multiprocessing as mp  # multiprocessing을 import 합니다.
from dataclasses import dataclass

@dataclass
class Bus:
    """
    파이프라인 단계 간 데이터 전송을 위한 중앙 큐 허브.
    - meta_q: 디코더(Process) -> 배처(Thread) (프로세스 간 통신)
    - infer_out_q: 배처(Thread) -> 트래커(Thread) (스레드 간 통신)
    - track_out_q: 트래커(Thread) -> 렌더러(MainThread) (스레드 간 통신)
    """
    meta_q: mp.Queue
    infer_out_q: queue.Queue
    track_out_q: queue.Queue

_bus = None

def get_bus():
    """전역 Bus 인스턴스를 반환합니다."""
    global _bus
    if _bus is None:
        # get_context("spawn")은 윈도우/macOS에서 안정성을 보장합니다.
        ctx = mp.get_context("spawn")
        _bus = Bus(
            # 프로세스 간 통신용 큐
            meta_q=ctx.Queue(maxsize=2048),
            # 스레드 간 통신용 큐 (기존과 동일)
            infer_out_q=queue.Queue(maxsize=1024),
            track_out_q=queue.Queue(maxsize=1024),
        )
    return _bus