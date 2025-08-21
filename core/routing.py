# core/routing.py
from typing import Dict, Optional

class RoutingTable:
    def __init__(self):
        self._ring: Dict[str, object] = {}    # stream_id -> SharedFrameRing
        self._panel: Dict[str, object] = {}   # stream_id -> VideoPanel/Widget

    def register(self, stream_id: str, ring, panel=None):
        self._ring[stream_id] = ring
        if panel is not None:
            self._panel[stream_id] = panel

    def unregister(self, stream_id: str):
        self._ring.pop(stream_id, None)
        self._panel.pop(stream_id, None)

    def ring(self, stream_id: str):
        return self._ring.get(stream_id)

    def panel(self, stream_id: str):
        return self._panel.get(stream_id)

_rt = RoutingTable()

def get_routing():
    return _rt
