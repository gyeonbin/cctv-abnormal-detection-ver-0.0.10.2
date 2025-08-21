# -*- coding: utf-8 -*-
# core/types.py — dataclasses with tuple-unpacking compatibility

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional
import numpy as np


@dataclass
class FrameMeta:
    """메타큐로 전달되는 디코드 프레임 메타데이터"""
    stream_id: str
    slot: int                 # SharedFrameRing slot index
    fid: int                  # frame id (0..)
    pts: Optional[float]      # presentation timestamp in seconds

    # 과거 코드에서 튜플 언패킹을 사용했을 수 있으므로, 안전한 이터레이터 제공
    def __iter__(self) -> Iterator:
        yield self.stream_id
        yield self.slot
        yield self.fid
        yield self.pts

    def to_tuple(self) -> Tuple[str, int, int, Optional[float]]:
        return (self.stream_id, self.slot, self.fid, self.pts)


@dataclass
class InferOut:
    """
    YOLO(등) 인퍼런스 결과. 배열은 numpy.ndarray 가정.
    det_xyxy: (N,4) float32
    det_conf: (N,) float32
    det_cls : (N,) float32

    언패킹 호환: (stream_id, fid, pts, det_xyxy, det_conf, det_cls)
    """
    stream_id: str
    fid: int
    pts: Optional[float]
    det_xyxy: np.ndarray
    det_conf: np.ndarray
    det_cls: np.ndarray

    def __iter__(self) -> Iterator:
        yield self.stream_id
        yield self.fid
        yield (0.0 if self.pts is None else float(self.pts))
        yield self.det_xyxy
        yield self.det_conf
        yield self.det_cls

    def __len__(self) -> int:
        # 언패킹 길이를 명시 (6개)
        return 6

    def to_tuple(self) -> Tuple[str, int, float, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.stream_id,
            self.fid,
            0.0 if self.pts is None else float(self.pts),
            self.det_xyxy,
            self.det_conf,
            self.det_cls,
        )


@dataclass
class TrackOut:
    """트래킹/로직 결과(시각화용 태그 포함)."""
    stream_id: str
    fid: int
    pts: Optional[float]
    # tags: [(x1,y1,x2,y2,"label"), ...]
    tags: List[Tuple[int, int, int, int, str]]

    def __iter__(self) -> Iterator:
        yield self.stream_id
        yield self.fid
        yield (0.0 if self.pts is None else float(self.pts))
        yield self.tags

    def to_tuple(self) -> Tuple[str, int, float, List[Tuple[int, int, int, int, str]]]:
        return (
            self.stream_id,
            self.fid,
            0.0 if self.pts is None else float(self.pts),
            self.tags,
        )
