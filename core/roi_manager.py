# core.roi_manager.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

Point = Tuple[float, float]

class ROI:
    def __init__(self, name: str, points: List[Point], zone_type: str = "intrusion", meta: dict = None):
        self.name = name
        self.points = points      # [(x,y), ...] 3개 이상
        self.zone_type = zone_type
        self.meta = meta or {}

    def contains(self, pt: Point) -> bool:
        if len(self.points) < 3:
            return False
        contour = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(contour, pt, False) >= 0

    def edges(self):
        n = len(self.points)
        for i in range(n):
            a = (float(self.points[i][0]), float(self.points[i][1]))
            b = (float(self.points[(i+1) % n][0]), float(self.points[(i+1) % n][1]))
            yield a, b


class ROIManager:
    def __init__(self):
        self._zones: Dict[str, ROI] = {}

    def set(self, name: str, points: List[Point], zone_type: str = "intrusion", meta: dict = None):
        self._zones[name] = ROI(name, points, zone_type, meta)

    def remove(self, name: str):
        self._zones.pop(name, None)

    def clear(self):
        self._zones.clear()

    def get(self, name: str) -> Optional[ROI]:
        return self._zones.get(name)

    def all(self, zone_type: Optional[str] = None) -> List[ROI]:
        if zone_type is None:
            return list(self._zones.values())
        return [z for z in self._zones.values() if z.zone_type == zone_type]
