# -*- coding: utf-8 -*-
# core/decode_proc.py
# InMemoryCsvLoggerNP 기반 로깅 + 실시간 페이싱(옵션) + 백프레셔 안정화

import os
import time
import av
import queue

from core.metrics import LoopTimer
from core.ringbuf import SharedFrameRing
from core.types import FrameMeta
from cfg.defaults import (
    SLOTS, PRINT_EVERY, LIMIT_DECODE_SPEED, WAIT_WHEN_QUEUES_FULL, BACKPRESSURE_HWM
)
from utils.inmem_logger import InMemoryCsvLoggerNP


def _open_container(video_path: str, use_hwaccel: bool):
    opts = {}
    if use_hwaccel:
        # 환경에 따라 무시될 수 있음
        opts.update({"hwaccel": "cuda"})
    return av.open(video_path, options=opts or None)


def decoder_process(stream_id: str, video_path: str, init_q, meta_q, stop_event,
                    use_hwaccel: bool, ring_name: str):
    """
    Decode → SharedFrameRing에 프레임 쓰기 → FrameMeta(meta_q) 전달.
    - 각 프레임 (t_wall, fid, pts, seq=fid) 로깅을 in-memory에 누적 후 종료 시 CSV로 저장.
    - LIMIT_DECODE_SPEED=True면 PTS 기준으로 실시간 페이싱.
    - WAIT_WHEN_QUEUES_FULL=True면 meta_q.put(block=True)로 백프레셔.
    """
    # ---------- Audit (In-Memory logger) ----------
    audit_dir = os.getenv("AUDIT_DIR", "./audit")
    os.makedirs(audit_dir, exist_ok=True)
    csv_path = os.path.join(audit_dir, f"{stream_id}_decode.csv")
    log = InMemoryCsvLoggerNP(header=["t_wall", "fid", "pts", "seq"], initial_capacity=200_000)

    container = None
    ring = None
    video_stream = None

    timer = LoopTimer(alpha=0.05)
    last_print_fid = -1

    try:
        container = _open_container(video_path, use_hwaccel)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        # 첫 프레임으로 해상도/초기 PTS 확보
        first_frame = next(container.decode(video=0))
        img0 = first_frame.to_ndarray(format="bgr24")
        H, W = img0.shape[:2]

        # 공유메모리 링 생성
        ring = SharedFrameRing(name=ring_name, n_slots=SLOTS, h=H, w=W, c=3, create=True)

        # 페이싱 기준선
        base_pts = float(first_frame.pts * float(video_stream.time_base)) if first_frame.pts is not None else 0.0
        base_wall = time.time()

        # 초기 알림(기존 호환: (W,H,SLOTS))
        if init_q is not None:
            try:
                init_q.put((W, H, SLOTS))
            except Exception:
                pass

        # 첫 프레임 처리
        fid = 0
        pts = base_pts
        ring.write(fid % SLOTS, img0)
        # meta_q 백프레셔 모드에 따라 put
        try:
            meta_q.put(FrameMeta(stream_id=stream_id, slot=(fid % SLOTS), fid=fid, pts=pts),
                       block=bool(WAIT_WHEN_QUEUES_FULL), timeout=0.5)
        except queue.Full:
            # 드롭은 정합성에 영향 → block 모드 미사용 시 희귀하게 발생 가능
            pass
        log.row((time.time(), fid, pts, fid))
        timer.tick()
        last_print_fid = fid

        # 나머지 프레임 루프
        for packet in container.demux(video=0):
            if stop_event.is_set():
                break
            for frame in packet.decode():
                fid += 1
                img = frame.to_ndarray(format="bgr24")
                pts = float(frame.pts * float(video_stream.time_base)) if frame.pts is not None else (
                    fid / float(video_stream.average_rate or 30.0)
                )

                # 실시간 페이싱(옵션): 현재 벽시계가 목표시간보다 빠르면 대기
                if LIMIT_DECODE_SPEED:
                    target_wall = base_wall + max(0.0, pts - base_pts)
                    now = time.time()
                    if target_wall > now:
                        time.sleep(target_wall - now)

                # 링에 쓰기
                slot = fid % SLOTS
                ring.write(slot, img)

                # meta 전송 (백프레셔)
                try:
                    meta_q.put(FrameMeta(stream_id=stream_id, slot=slot, fid=fid, pts=pts),
                               block=bool(WAIT_WHEN_QUEUES_FULL), timeout=0.5)
                except queue.Full:
                    # 드롭 방지 위해 마지막으로 한 번 더 대기 후 재시도
                    try:
                        time.sleep(0.002)
                        meta_q.put(FrameMeta(stream_id=stream_id, slot=slot, fid=fid, pts=pts), block=True, timeout=1.0)
                    except Exception:
                        pass

                # 로깅
                log.row((time.time(), fid, pts, fid))

                # 통계
                if fid % PRINT_EVERY == 0 and fid != last_print_fid:
                    print(f"[DECODE] {stream_id} fid={fid} (EMA FPS={timer.fps:.2f})")
                    last_print_fid = fid
                timer.tick()

                if stop_event.is_set():
                    break

    except StopIteration:
        pass
    except Exception as e:
        print(f"[ERROR][DECODE:{stream_id}] {e}")
    finally:
        # 자원 정리
        try:
            if container:
                container.close()
        except Exception:
            pass
        try:
            if ring:
                ring.close()
        except Exception:
            pass
        # 로그 덤프
        try:
            log.dump(csv_path)
            print(f"[DECODE] audit saved → {csv_path}")
        except Exception as e:
            print(f"[DECODE] audit dump failed: {e}")
