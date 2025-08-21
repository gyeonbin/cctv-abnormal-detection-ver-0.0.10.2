# CCTV 프로젝트 — 작업 로그 & 설계/구조 노트 (Live)

> 이 문서는 **작업 진행 상황·설계 결정·디렉터리 구조**를 한 곳에 모아, 방을 새로 열어도 맥락이 끊기지 않도록 하기 위한 **단일 진실 소스(Single Source of Truth)** 입니다. 새 세션을 열 때는 이 문서의 **세션 핑 템플릿**을 복붙해서 시작해 주세요.

---

## 0) 세션 핑 템플릿 (새 방 시작 시 여기에 복붙)

```
[세션 핑]
- 브랜치/버전:
- 오늘의 목표(3줄):
- 현재 구조(요약): 프로세스(디코더) ↔ 메인(GPU: YOLO 배치+트래커) ↔ 패널(GL)
- 변경 중인 파일:
- 새 결정/가설:
- 막힌 곳(필요한 답):
```

---

## 1) 현재 아키텍처 스냅샷 (As-Is)

- **프로세스 분리:** Decoder Proc(프레임 생산, SharedMemory 링버퍼) ↔ Main GUI Proc(YOLO 배치 추론, 추적, 렌더)
- **추론 파이프:** `BATCH_SIZE`, `MAX_BATCH_DELAY_MS`, `YOLO_MAX_SIDE` 기반 배치 예측 → 스트림별 결과 분배
- **렌더:** QOpenGLWidget(PBO 더블버퍼) + Qt 텍스트 오버레이, drift/FPS 표시
- **ROI:** 재생 중 지정(패널 내 모드), 사전 지정(도입 예정)
- **스냅샷:** 학습용 프레임/크롭 저장(도입 예정)

---

## 2) 목표(To-Be)

- **워크스페이스 기반 멀티 패널**(여러 영상 동시 재생/추론/추적)
- **공용 InferenceBatcher**(GPU 모델 1개로 다중 스트림 배치 처리)
- **TrackerManager**(스트림별 상태/ROI)
- **SnapshotService**(정책 기반 프레임/크롭 저장 + manifest.csv)
- **Settings/Workspace**(전역·스트림 설정 + 레이아웃 저장/복원)
- **이상탐지 모델 플러그인**(.pt 교체 가능)

---

## 3) 디렉터리/파일 구조 제안

```
gui/
  app.py                  # QApplication 부트스트랩
  main_window.py          # 워크스페이스/도킹/툴바/우클릭 메뉴
  video_panel.py          # GLVideoWidget + Panel 래퍼(패널별 드리프트/FPS)
  dialogs.py              # OpenSourceDialog, SettingsDialog, ROIEditor
  actions.py              # QAction 묶음 생성/바인딩

core/
  decode_proc.py          # DecoderProcess(stream_id) + Shared ring publish
  ringbuf.py              # SharedFrameRing (현재 클래스 이동)
  batcher.py              # InferenceBatcher(다중 스트림 → 단일 배치)
  tracker_manager.py      # LogicDetector 스트림별 관리/ROI API
  bus.py                  # 전역 큐(meta_q_global, infer_out_q, track_out_q)
  routing.py              # stream_id ↔ ring/패널 등록/해제 유틸
  types.py                # dataclass: FrameMeta/InferReq/InferOut/TrackOut
  metrics.py              # LoopTimer/EMA/정기 로그
  snapshot.py             # 크롭/프레임 저장 + manifest.csv
  settings.py             # AppSettings/StreamSettings 직렬화
  workspace.py            # 워크스페이스 저장/복원(.cctvproj)

detectors/
  yolo_ultralytics.py     # YOLO 어댑터(ultralytics predict 래핑)
  anomaly_base.py         # 이상탐지 플러그인 인터페이스
  anomaly_mil.py          # (예시) MIL 기반 .pt 로더/추론

cfg/
  defaults.py             # BATCH_SIZE/YOLO_MAX_SIDE/IOU/CONF 등 기본값
  workspace.schema.json   # (선택) 워크스페이스 스키마

main.py                   # 진입점(gui/app.run_app())
```

> **호환성 원칙:** 지금의 `video_player.py`에서 클래스를 뽑아 위 파일들로 **옮기기만** 하고, 동작은 그대로 유지하면서 외부에서 조립합니다.

---

## 4) 리팩터링 플랜 (체크리스트)

### Phase 1 — 파일 분리, 동작 동일

-

### Phase 2 — 멀티 소스 + 공용 배치 추론

-

### Phase 3 — 설정·워크스페이스·스냅샷·플러그인

-

---

## 5) 인터페이스 계약 (필수 타입)

```python
# core/types.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class FrameMeta:
    stream_id: str
    slot: int
    fid: int
    pts: float

@dataclass
class InferReq:
    stream_id: str
    fid: int
    pts: float
    img_src: np.ndarray
    img_yolo: np.ndarray
    scale_xy: Tuple[float, float]

@dataclass
class InferOut:
    stream_id: str
    fid: int
    pts: float
    det_xyxy: np.ndarray  # (N,4)
    det_conf: np.ndarray  # (N,)
    det_cls: np.ndarray   # (N,)

@dataclass
class TrackOut:
    stream_id: str
    fid: int
    pts: float
    tags: List[Tuple[int,int,int,int,str]]  # src 좌표계
```

**버스 채널 (core/bus.py):**

- `meta_q_global: Queue[FrameMeta]`
- `infer_out_q: Queue[InferOut]`
- `track_out_q: Queue[TrackOut]`

---

## 6) 라우팅 규칙

- `stream_id` 형식: `str` (예: `cam01`, `file:/path/...`는 별칭으로 매핑)
- 등록: `routing.register_stream(stream_id, ring, panel)`
- 해제: `routing.unregister_stream(stream_id)` → ring close/unlink, panel dispose

---

## 7) 배치 추론 규칙(SLA/파라미터)

- `BATCH_SIZE`(기본 8), `MAX_BATCH_DELAY_MS`(기본 40ms)
- YOLO 입력: 긴 변 `YOLO_MAX_SIDE`(기본 1280) 리사이즈, `scale_xy`로 복원
- 클래스 필터: 기본 `[0]`(person), 스트림별 override 허용
- 예외 시 CPU 폴백 허용(로그 남김)

---

## 8) ROI & 스냅샷 정책

- ROI: 패널 내 즉시 지정 + **사전 지정(ROIEditor)** 지원
- 스냅샷 모드: `frame | crop`
- 트리거: `always | on_new_id | on_event`
- 레이트 리밋: 스트림/ID 단위 cool-down
- `manifest.csv` 스키마:

```
path,stream_id,fid,pts,tid,x1,y1,x2,y2,conf,cls,roi_tag,timestamp
```

---

## 9) 개발 규칙(요약)

- **로깅 태그:** `[DECODE]`, `[BATCH]`, `[TRACK]`, `[PANEL:<id>]`, `[SNAP]`
- **시간 의미:** 모든 지연/드리프트는 **PTS 기준** 명시
- **큐 백프레셔:** 디코더는 `meta_q_global` 하이워터마크 관찰 → PTS 페이싱 유지
- **GL 호출:** UI 스레드 한정, 패널 외부에서 GL 금지

---

## 10) 벤치마크 & 품질 지표 템플릿

- 해상도/스트림 수/배치/지연 조합별 **Decode/YOLO/Track/Render EMA FPS**, drift(ms), drop(%)

```
[벤치 표]
- 소스: cam01(1080p), fileA(4K)
- 설정: BATCH=8, MAX_DELAY=40ms, YOLO_MAX_SIDE=1280
- 결과(3000f): Decode 27.4 | YOLO 24.9 | Track 24.6 | Render 25.1 | drift +18.3ms | drop 3.2%
```

---

## 11) 변경 이력 (Changelog)

- 2025-08-17: 문서 초안 생성. 리팩터링 Phase 1/2/3 정의.

---

## 12) ADR 인덱스 & 템플릿

- ADR-0001: **단일 InferenceBatcher로 다중 스트림 처리**
- ADR-0002: **ROI 사전 지정 플로우 도입**

**ADR 템플릿**

```
# ADR-XXXX: 제목
- 상태: proposed | accepted | superseded
- 날짜: YYYY-MM-DD
## 문맥
## 결정
## 결과(Trade-offs)
## 대안
```

---

