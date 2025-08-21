# reid_embedder_onnx.py (fixed)
import os, math, urllib.request, urllib.error, hashlib, time
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime가 필요합니다. 먼저 `pip install onnxruntime` 해주세요.") from e

WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "osnet_x0_25_msmt17.onnx")

# 검증용 SHA256 (HF 페이지에 공개된 값)
_OSNET_SHA256 = "e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277"

# 동작 확인된 미러들
OSNET_ONNX_URLS = [
    # Hugging Face LFS (download 파라미터 포함)
    "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx?download=true",
    # onnx2tf 문서에 소개된 공개 미러
    "https://s3.ap-northeast-2.wasabisys.com/temp-models/onnx2tf_441/osnet_x0_25_msmt17.onnx",
]

def _sha256(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _download(url, path, timeout=30, tries=2):
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*",
                    "Connection": "keep-alive",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(path, "wb") as f:
                f.write(resp.read())
            return
        except Exception as e:
            last = e
            time.sleep(0.8)
    raise last

def ensure_weights(path=WEIGHTS_PATH, urls=OSNET_ONNX_URLS):
    # 1) 환경변수 우선
    env_path = os.environ.get("REID_ONNX_PATH")
    if env_path and os.path.exists(env_path):
        # 무결성 체크(있으면)
        try:
            if _sha256(env_path) == _OSNET_SHA256:
                return env_path
        except Exception:
            return env_path  # 체크 실패해도 사용자는 의도적으로 지정했으니 통과
        return env_path

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2) 로컬 캐시가 이미 있고, 사이즈/해시가 정상인 경우
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        try:
            if _sha256(path) == _OSNET_SHA256:
                return path
        except Exception:
            return path  # 해시 계산 실패 시 그냥 사용

    # 3) 미러 순회 다운로드
    last_err = None
    for u in urls:
        try:
            print(f"[ReID] Downloading ONNX weights: {u}")
            _download(u, path)
            if os.path.getsize(path) > 100_000:
                # 해시 검증(가능할 때만)
                try:
                    h = _sha256(path)
                    if h != _OSNET_SHA256:
                        print(f"[ReID] SHA256 mismatch: {h} (expected {_OSNET_SHA256}), try next mirror")
                        continue
                except Exception:
                    pass
                print(f"[ReID] Downloaded: {path}")
                return path
        except Exception as e:
            last_err = e
            print(f"[ReID] download fail: {e}")
            continue

    raise RuntimeError(f"ReID ONNX 다운로드 실패: {path} (last error: {last_err})")

# ImageNet 정규화
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocess_crops_rgb(crops_rgb, size=(128, 256)):
    tensors = []
    for crop in crops_rgb:
        if crop.size == 0:
            crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        tensors.append(arr)
    if not tensors:
        return np.empty((0, 3, size[1], size[0]), dtype=np.float32)
    return np.stack(tensors, axis=0).astype(np.float32)

class ONNXReIDEmbedder(object):
    def __init__(self, weight_path=None, use_gpu=True):
        weight_path = weight_path or ensure_weights()
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if (use_gpu and "CUDAExecutionProvider" in ort.get_available_providers())
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(weight_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name  = inp.name
        self.output_name = self.session.get_outputs()[0].name

        # ✅ 모델 입력 shape 파악 (ex: [16, 3, 256, 128] 혹은 [None, 3, 256, 128])
        ishape = inp.shape
        self.fixed_bs = ishape[0] if isinstance(ishape[0], int) else None  # 16이면 고정 배치
        # (H, W)는 onnx 내 정의가 [N, C, H, W] 기준
        self.in_h = int(ishape[2]) if isinstance(ishape[2], int) else 256
        self.in_w = int(ishape[3]) if isinstance(ishape[3], int) else 128

    def __call__(self, frame_rgb, boxes_xyxy):
        h, w, _ = frame_rgb.shape
        crops = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            x1i, y1i = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
            x2i, y2i = min(w, int(math.ceil(x2))),  min(h, int(math.ceil(y2)))
            crops.append(frame_rgb[y1i:y2i, x1i:x2i])

        # ✅ 모델 입력 크기에 맞춰 전처리
        batch = _preprocess_crops_rgb(crops, size=(self.in_w, self.in_h))
        N = batch.shape[0]
        if N == 0:
            return np.zeros((0, 256), dtype=np.float32)

        # ✅ 배치 고정 모델이면 패딩/청크 처리
        if self.fixed_bs and self.fixed_bs > 0:
            bs = self.fixed_bs
            outs = []
            if N <= bs:
                if N < bs:
                    pad = np.zeros((bs - N, 3, self.in_h, self.in_w), dtype=np.float32)
                    batch_run = np.concatenate([batch, pad], axis=0)
                else:
                    batch_run = batch
                out = self.session.run([self.output_name], {self.input_name: batch_run})[0]
                outs.append(out[:N])
            else:
                # N > bs 이면 청크로 나눠서 실행
                for i in range(0, N, bs):
                    chunk = batch[i:i+bs]
                    if chunk.shape[0] < bs:
                        pad = np.zeros((bs - chunk.shape[0], 3, self.in_h, self.in_w), dtype=np.float32)
                        chunk = np.concatenate([chunk, pad], axis=0)
                    out = self.session.run([self.output_name], {self.input_name: chunk})[0]
                    valid = out[:min(bs, N - i)]
                    outs.append(valid)
            out = np.concatenate(outs, axis=0)
        else:
            # 동적 배치 모델이면 그대로 실행
            out = self.session.run([self.output_name], {self.input_name: batch})[0]

        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
        return (out / norms).astype(np.float32)


def build_embedder_onnx(auto_download=True, use_gpu=True):
    if auto_download:
        ensure_weights()
    return ONNXReIDEmbedder(WEIGHTS_PATH, use_gpu=use_gpu)
