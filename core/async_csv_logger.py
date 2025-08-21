# core/async_csv_logger.py
# 비동기 CSV 로거 (백그라운드 Thread/Process 자동 선택)
# - 핫패스에서는 row()/rows()만 호출: 절대 블로킹 없음 (큐 포화 시 드롭/카운트)
# - 디코더 같은 daemon 프로세스 안에서는 자동으로 Thread 로거 사용
# - 메인 등 non-daemon이면 프로세스 로거 사용(기본), 환경변수로 강제 전환 가능

import os
import csv
import time
import atexit
import queue as qmod
from typing import Iterable, List, Optional

import multiprocessing as mp
import threading

_DEFAULT_FLUSH_SEC = 0.5
_DEFAULT_BATCH = 256
_DEFAULT_QSIZE = 10000

def _should_use_process() -> bool:
    """daemon 프로세스 내부면 False. 환경변수로 강제 가능."""
    force_thread = os.getenv("ASYNC_CSV_FORCE_THREAD", "0") == "1"
    force_proc   = os.getenv("ASYNC_CSV_FORCE_PROCESS", "0") == "1"
    use_proc = False
    try:
        curr = mp.current_process()
        if not curr.daemon:  # daemon이면 자식 프로세스 금지 → Thread 사용
            use_proc = True
    except Exception:
        use_proc = False
    if force_thread:
        return False
    if force_proc:
        return True
    return use_proc


class _BaseWriter:
    def __init__(self, path: str, header: Optional[List[str]],
                 flush_sec: float, batch_size: int):
        self.path = path
        self.header = header
        self.flush_sec = max(0.05, float(flush_sec))
        self.batch_size = max(1, int(batch_size))
        self._dropped = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    def inc_dropped(self, n: int = 1):
        self._dropped += max(0, n)


# ------------------------- Thread Writer -------------------------
class _ThreadWriter(_BaseWriter):
    def __init__(self, path: str, header: Optional[List[str]],
                 flush_sec: float, batch_size: int, qsize: int):
        super().__init__(path, header, flush_sec, batch_size)
        self.q: qmod.Queue = qmod.Queue(maxsize=qsize)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _open_writer(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        f = open(self.path, "a", newline="", encoding="utf-8", buffering=1024*1024)
        w = csv.writer(f)
        if (not exists) and self.header:
            w.writerow(self.header)
            f.flush()
        return f, w

    def _run(self):
        f, w = self._open_writer()
        batch = []
        last_flush = time.perf_counter()
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                timeout = max(0.01, self.flush_sec - (now - last_flush))
                try:
                    msg = self.q.get(timeout=timeout)
                except qmod.Empty:
                    msg = None

                if msg is not None:
                    kind = msg[0]
                    if kind == "row":
                        batch.append(msg[1])
                    elif kind == "rows":
                        rows = msg[1]
                        if rows:
                            batch.extend(rows)
                    elif kind == "flush":
                        pass
                    elif kind == "stop":
                        # drain
                        while True:
                            try:
                                k = self.q.get_nowait()
                                if k[0] == "row":
                                    batch.append(k[1])
                                elif k[0] == "rows" and k[1]:
                                    batch.extend(k[1])
                            except qmod.Empty:
                                break
                        if batch:
                            w.writerows(batch); batch.clear(); f.flush()
                        break

                now = time.perf_counter()
                if batch and (len(batch) >= self.batch_size or (now - last_flush) >= self.flush_sec):
                    w.writerows(batch); batch.clear(); f.flush(); last_flush = now
        finally:
            try:
                if batch:
                    w.writerows(batch); f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass

    # 외부에서 호출되는 인터페이스
    def put_row(self, row):
        try:
            self.q.put_nowait(("row", row))
        except qmod.Full:
            self.inc_dropped(1)

    def put_rows(self, rows: List[List]):
        if not rows:
            return
        # 1024개 청크로 분할
        for i in range(0, len(rows), 1024):
            chunk = rows[i:i+1024]
            try:
                self.q.put_nowait(("rows", chunk))
            except qmod.Full:
                self.inc_dropped(len(chunk))

    def flush(self):
        try:
            self.q.put_nowait(("flush", None))
        except qmod.Full:
            pass

    def close(self, timeout: float = 3.0):
        try:
            self.q.put(("stop", None))
        except Exception:
            pass
        self._t.join(timeout=timeout)


# ------------------------- Process Writer -------------------------
class _ProcImpl(mp.Process):
    def __init__(self, path: str, header: Optional[List[str]],
                 q: mp.Queue, flush_sec: float, batch_size: int):
        super().__init__(daemon=False)
        self.path = path
        self.header = header
        self.q = q
        self.flush_sec = flush_sec
        self.batch_size = batch_size

    def _open_writer(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
        f = open(self.path, "a", newline="", encoding="utf-8", buffering=1024*1024)
        w = csv.writer(f)
        if (not exists) and self.header:
            w.writerow(self.header)
            f.flush()
        return f, w

    def run(self):
        f, w = self._open_writer()
        batch = []
        last_flush = time.perf_counter()
        try:
            while True:
                now = time.perf_counter()
                timeout = max(0.01, self.flush_sec - (now - last_flush))
                try:
                    msg = self.q.get(timeout=timeout)
                except Exception:
                    msg = None
                if msg is not None:
                    kind = msg[0]
                    if kind == "row":
                        batch.append(msg[1])
                    elif kind == "rows":
                        rows = msg[1]
                        if rows:
                            batch.extend(rows)
                    elif kind == "flush":
                        pass
                    elif kind == "stop":
                        # drain
                        while True:
                            try:
                                k = self.q.get_nowait()
                                if k[0] == "row":
                                    batch.append(k[1])
                                elif k[0] == "rows" and k[1]:
                                    batch.extend(k[1])
                            except qmod.Empty:
                                break
                        if batch:
                            w.writerows(batch); f.flush()
                        break

                now = time.perf_counter()
                if batch and (len(batch) >= self.batch_size or (now - last_flush) >= self.flush_sec):
                    w.writerows(batch); batch.clear(); f.flush(); last_flush = now
        finally:
            try:
                if batch:
                    w.writerows(batch); f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass


class _ProcessWriter(_BaseWriter):
    def __init__(self, path: str, header: Optional[List[str]],
                 flush_sec: float, batch_size: int, qsize: int):
        super().__init__(path, header, flush_sec, batch_size)
        self.q: mp.Queue = mp.Queue(maxsize=qsize)
        self._p = _ProcImpl(path=path, header=header, q=self.q,
                            flush_sec=self.flush_sec, batch_size=self.batch_size)
        self._p.start()

    def put_row(self, row):
        try:
            self.q.put_nowait(("row", row))
        except qmod.Full:
            self.inc_dropped(1)

    def put_rows(self, rows: List[List]):
        if not rows:
            return
        for i in range(0, len(rows), 1024):
            chunk = rows[i:i+1024]
            try:
                self.q.put_nowait(("rows", chunk))
            except qmod.Full:
                self.inc_dropped(len(chunk))

    def flush(self):
        try:
            self.q.put_nowait(("flush", None))
        except qmod.Full:
            pass

    def close(self, timeout: float = 3.0):
        try:
            self.q.put(("stop", None))
        except Exception:
            pass
        try:
            self._p.join(timeout=timeout)
        except Exception:
            pass


# ------------------------- Public API -------------------------
class AsyncCsvLogger:
    def __init__(self, path: str, header: Optional[List[str]] = None,
                 flush_interval_sec: float = _DEFAULT_FLUSH_SEC,
                 batch_size: int = _DEFAULT_BATCH,
                 queue_max: int = _DEFAULT_QSIZE):
        self.path = path
        self.header = header
        self._impl: _BaseWriter
        if _should_use_process():
            self._impl = _ProcessWriter(path, header, flush_interval_sec, batch_size, queue_max)
        else:
            self._impl = _ThreadWriter(path, header, flush_interval_sec, batch_size, queue_max)
        atexit.register(self.close)

    @property
    def dropped(self) -> int:
        return self._impl.dropped

    def row(self, values: List):
        self._impl.put_row(values)

    def rows(self, rows: Iterable[List]):
        rows_list = list(rows)
        self._impl.put_rows(rows_list)

    def flush(self):
        self._impl.flush()

    def close(self, timeout: float = 3.0):
        try:
            self._impl.close(timeout=timeout)
        except Exception:
            pass


_LOGGERS = {}

def get_logger(path: str, header: Optional[List[str]] = None,
               flush_interval_sec: float = _DEFAULT_FLUSH_SEC,
               batch_size: int = _DEFAULT_BATCH,
               queue_max: int = _DEFAULT_QSIZE) -> AsyncCsvLogger:
    lg = _LOGGERS.get(path)
    if lg is None:
        lg = AsyncCsvLogger(path=path, header=header,
                            flush_interval_sec=flush_interval_sec,
                            batch_size=batch_size, queue_max=queue_max)
        _LOGGERS[path] = lg
    return lg


def close_all():
    for p, lg in list(_LOGGERS.items()):
        try:
            lg.close()
        except Exception:
            pass
        _LOGGERS.pop(p, None)
