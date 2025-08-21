# core/routing.py
# stream_id -> SharedFrameRing 이름/객체 매핑 관리

from typing import Dict, Optional
from core.ringbuf import SharedFrameRing

class RingRouting:
    def __init__(self):
        # stream_id -> ring_name
        self._map: Dict[str, str] = {}
        # ring_name -> ring_instance
        self._cache: Dict[str, SharedFrameRing] = {}

    def register_ring(self, stream_id: str, ring_name: str):
        """스트림 ID와 공유 메모리 링의 이름을 등록합니다."""
        self._map[str(stream_id)] = str(ring_name)

    def get_ring(self, stream_id: str) -> Optional[SharedFrameRing]:
        """스트림 ID에 해당하는 공유 메모리 링 객체를 반환합니다."""
        ring_name = self._map.get(str(stream_id))
        if ring_name is None:
            return None

        # 캐시 확인
        ring = self._cache.get(ring_name)
        if ring is None:
            try:
                # 캐시에 없으면 객체 생성
                ring = SharedFrameRing(name=ring_name, create=False)
                self._cache[ring_name] = ring
            except Exception:
                # 링이 아직 생성되지 않았을 수 있음
                return None
        return ring

# --- 싱글턴 인스턴스 ---
_ROUTING_INSTANCE = None

def get_routing() -> RingRouting:
    """전역 라우팅 인스턴스를 반환합니다."""
    global _ROUTING_INSTANCE
    if _ROUTING_INSTANCE is None:
        _ROUTING_INSTANCE = RingRouting()
    return _ROUTING_INSTANCE

def register_ring(stream_id: str, ring_name: str):
    """라우팅 테이블에 링을 등록하는 전역 헬퍼 함수."""
    get_routing().register_ring(stream_id, ring_name)