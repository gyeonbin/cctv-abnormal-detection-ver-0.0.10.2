# core.person_tracker.py  (AH-state Kalman + ReID-stabilized)
import numpy as np
from collections import defaultdict, deque

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# --------------------------- 기본 유틸 ---------------------------

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (ay2 - ay1))
    union = area_a + area_b - inter + 1e-6
    return inter / union


def xyxy_to_cyah(box):
    """ [x1,y1,x2,y2] -> [cx, cy, a, h]  (a = w/h, h = height) """
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=float)


def cyah_to_xyxy(state):
    """ [cx,cy,a,h] -> [x1,y1,x2,y2] """
    cx, cy, a, h = state[:4]
    a = float(np.clip(a, 0.2, 6.0))       # 안전 클램프
    h = float(max(1.0, h))
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


def cosine_distance(a, b):
    an = a / (np.linalg.norm(a) + 1e-6)
    bn = b / (np.linalg.norm(b) + 1e-6)
    return 1.0 - float(np.dot(an, bn))


# --------------------------- 칼만 필터 (AH 상태) ---------------------------

class KalmanBoxAH:
    """
    상태: [cx, cy, a, h, vx, vy, va, vh]
    관측: [cx, cy, a, h]
    예측: cx+=vx*dt, cy+=vy*dt, a+=va*dt, h+=vh*dt
    """
    def __init__(self, dt=1.0):
        self.dt = float(dt)

        self._motion_mat = np.eye(8)
        # 위치/형상에 속도 결합
        self._motion_mat[0, 4] = self.dt  # cx += vx
        self._motion_mat[1, 5] = self.dt  # cy += vy
        self._motion_mat[2, 6] = self.dt  # a  += va
        self._motion_mat[3, 7] = self.dt  # h  += vh

        self._update_mat = np.zeros((4, 8))
        self._update_mat[0, 0] = 1.0  # cx
        self._update_mat[1, 1] = 1.0  # cy
        self._update_mat[2, 2] = 1.0  # a
        self._update_mat[3, 3] = 1.0  # h

        # 스케일링 계수 (DeepSORT 계열 감각)
        self.pos_xy_factor = 1.0 / 20.0   # cx,cy 관측/상태 잡음은 h에 비례
        self.pos_h_factor  = 1.0 / 20.0   # h 관측/상태 잡음
        self.pos_a_sigma   = 0.04         # a 관측/상태 잡음(작게)

        self.vel_xy_factor = 1.0 / 160.0  # vx,vy 프로세스 잡음(h에 비례)
        self.vel_h_factor  = 1.0 / 80.0   # vh 프로세스 잡음
        self.vel_a_sigma   = 0.004        # va 프로세스 잡음(아주 작게)

        self.vel_damping   = 0.90         # 속도 감쇠(발산 방지)

        self.mean = None  # (8,)
        self.cov  = None  # (8,8)

        # 안전 한계
        self.a_min, self.a_max = 0.25, 4.0
        self.h_min = 4.0

    def _std_pos(self, a, h):
        # cx,cy는 h 규모에 비례, a는 작은 고정, h는 h에 비례
        sxy = h * self.pos_xy_factor
        sa  = self.pos_a_sigma
        sh  = h * self.pos_h_factor
        return np.array([sxy, sxy, sa, sh], dtype=float)

    def _std_vel(self, a, h):
        vxy = h * self.vel_xy_factor
        va  = self.vel_a_sigma
        vh  = h * self.vel_h_factor
        return np.array([vxy, vxy, va, vh], dtype=float)

    def initiate(self, meas_cyah):
        # meas_cyah: [cx,cy,a,h]
        cx, cy, a, h = meas_cyah
        a = float(np.clip(a, self.a_min, self.a_max))
        h = float(max(self.h_min, h))

        self.mean = np.zeros(8, dtype=float)
        self.mean[:4] = [cx, cy, a, h]

        sp = self._std_pos(a, h)
        sv = self._std_vel(a, h)
        self.cov = np.zeros((8, 8), dtype=float)
        self.cov[0, 0] = sp[0] ** 2
        self.cov[1, 1] = sp[1] ** 2
        self.cov[2, 2] = sp[2] ** 2
        self.cov[3, 3] = sp[3] ** 2
        self.cov[4, 4] = sv[0] ** 2
        self.cov[5, 5] = sv[1] ** 2
        self.cov[6, 6] = sv[2] ** 2
        self.cov[7, 7] = sv[3] ** 2

    def predict(self):
        if self.mean is None:
            return
        cx, cy, a, h = self.mean[:4]
        a = float(np.clip(a, self.a_min, self.a_max))
        h = float(max(self.h_min, h))

        # 동적 Q (상태 크기에 따른 프로세스 잡음)
        sp = self._std_pos(a, h)
        sv = self._std_vel(a, h)
        Q = np.zeros((8, 8), dtype=float)
        Q[0, 0] = sp[0] ** 2
        Q[1, 1] = sp[1] ** 2
        Q[2, 2] = sp[2] ** 2
        Q[3, 3] = sp[3] ** 2
        Q[4, 4] = sv[0] ** 2
        Q[5, 5] = sv[1] ** 2
        Q[6, 6] = sv[2] ** 2
        Q[7, 7] = sv[3] ** 2

        self.mean = self._motion_mat @ self.mean
        self.cov  = self._motion_mat @ self.cov @ self._motion_mat.T + Q

        # 속도 감쇠(발산 방지)
        self.mean[4] *= self.vel_damping
        self.mean[5] *= self.vel_damping
        self.mean[6] *= self.vel_damping
        self.mean[7] *= self.vel_damping

        # 안전 클램프
        self.mean[2] = float(np.clip(self.mean[2], self.a_min, self.a_max))
        self.mean[3] = float(max(self.h_min, self.mean[3]))

    def update(self, meas_cyah):
        if self.mean is None:
            self.initiate(meas_cyah)
            return
        cx, cy, a, h = meas_cyah
        a = float(np.clip(a, self.a_min, self.a_max))
        h = float(max(self.h_min, h))

        # 관측 잡음 R
        sp = self._std_pos(a, h)
        R = np.diag(sp ** 2)

        H = self._update_mat
        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)
        z = np.array([cx, cy, a, h], dtype=float)
        y = z - H @ self.mean

        self.mean = self.mean + K @ y
        I = np.eye(8)
        self.cov = (I - K @ H) @ self.cov

        # 안전 클램프
        self.mean[2] = float(np.clip(self.mean[2], self.a_min, self.a_max))
        self.mean[3] = float(max(self.h_min, self.mean[3]))

    def to_xyxy(self):
        return cyah_to_xyxy(self.mean)


# --------------------------- 트랙 ---------------------------

class Track:
    def __init__(self, track_id, init_state_cyah, feature=None,
                 max_feat=30, feat_ema_alpha=0.4, gallery_k=10):
        self.id = track_id
        self.kf = KalmanBoxAH()
        self.kf.initiate(init_state_cyah)

        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.conf = None
        self.dead = False

        # Feature EMA + 갤러리
        self.features = deque(maxlen=max_feat)
        self.feat_ema = None
        self.feat_ema_alpha = feat_ema_alpha
        self.gallery = deque(maxlen=gallery_k)

        if feature is not None:
            self.features.append(feature)
            self.gallery.append(feature.copy())
            self.feat_ema = feature.copy()

        self.confirmed = False
        self.last_update_age = 1

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, meas_state_cyah, conf=None, feature=None, n_init=2):
        self.kf.update(meas_state_cyah)
        self.hits += 1
        self.time_since_update = 0
        self.last_update_age = self.age
        if conf is not None:
            self.conf = conf
        if feature is not None:
            self.features.append(feature)
            self.gallery.append(feature.copy())
            if self.feat_ema is None:
                self.feat_ema = feature.copy()
            else:
                a = self.feat_ema_alpha
                self.feat_ema = a * self.feat_ema + (1 - a) * feature

        if self.hits >= n_init:
            self.confirmed = True

    def bbox_xyxy(self):
        return self.kf.to_xyxy()

    def centroid(self):
        x1, y1, x2, y2 = self.bbox_xyxy()
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def best_app_distance(self, f):
        d_best = 1.0
        if self.feat_ema is not None:
            d_best = min(d_best, cosine_distance(self.feat_ema, f))
        for g in self.gallery:
            d_best = min(d_best, cosine_distance(g, f))
        return d_best


# --------------------------- 트래커 ---------------------------

class PersonTracker:
    """
    DeepSORT-like + ReID 안정화 + 재결합 패스
    - feature_extractor(frame, boxes_xyxy) -> np.ndarray [N, D] 있으면 ReID 사용
    """
    def __init__(
        self,
        feature_extractor=None,
        max_age=60,
        n_init=2,
        iou_threshold=0.2,
        max_disappeared=60,
        max_feature_cosine=0.65,
        appearance_weight=0.65,
        reid_override_cos=0.28,
        reassoc_ttl=10,
        reassoc_motion_w=0.15
    ):
        self.feature_extractor = feature_extractor
        self.next_id = 0
        self.tracks = []

        self.objects = {}
        self.bboxes = {}
        self.paths = defaultdict(lambda: deque(maxlen=100))
        self.disappeared = {}

        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.max_feature_cosine = max_feature_cosine
        self.appearance_weight = appearance_weight

        self.reid_override_cos = reid_override_cos
        self.reassoc_ttl = reassoc_ttl
        self.reassoc_motion_w = reassoc_motion_w

    # ---------- 내부 유틸 ----------

    def _start_track(self, box_xyxy, conf=None, feat=None):
        state = xyxy_to_cyah(box_xyxy)
        t = Track(self.next_id, state, feat)
        t.conf = conf
        self.tracks.append(t)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _remove_dead(self):
        alive = []
        for t in self.tracks:
            if t.time_since_update > self.max_age:
                t.dead = True
            if not t.dead:
                alive.append(t)
            else:
                self.objects.pop(t.id, None)
                self.bboxes.pop(t.id, None)
                self.paths.pop(t.id, None)
                self.disappeared.pop(t.id, None)
        self.tracks = alive

    def _match(self, boxes, feats):
        if len(self.tracks) == 0 or len(boxes) == 0:
            return [], list(range(len(self.tracks))), list(range(len(boxes)))

        T, D = len(self.tracks), len(boxes)
        # IoU
        iou_mat = np.zeros((T, D), dtype=float)
        for i, trk in enumerate(self.tracks):
            tb = trk.bbox_xyxy()
            for j, db in enumerate(boxes):
                iou_mat[i, j] = iou_xyxy(tb, db)
        iou_dist = 1.0 - iou_mat

        # Appearance
        has_app_input = (feats is not None and len(feats) > 0)
        if has_app_input:
            app_dist = np.ones((T, D), dtype=float)
            has_any_track_feat = False
            for i, trk in enumerate(self.tracks):
                if (trk.feat_ema is None) and (len(trk.gallery) == 0):
                    continue
                has_any_track_feat = True
                for j, f in enumerate(feats):
                    app_dist[i, j] = trk.best_app_distance(f)
            w_app = self.appearance_weight if has_any_track_feat else 0.2
        else:
            app_dist = np.ones((T, D), dtype=float)
            w_app = 0.2

        cost = w_app * app_dist + (1.0 - w_app) * iou_dist

        # 게이트
        gate_mask = (iou_mat >= self.iou_threshold)
        if has_app_input:
            gate_mask &= (app_dist <= self.max_feature_cosine)
            # appearance override
            override_mask = (app_dist <= self.reid_override_cos)
            gate_mask = np.logical_or(gate_mask, override_mask)

        BIG = 1e6
        cost = np.where(gate_mask, cost, BIG)

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
            matches, used_cols = [], set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= BIG:
                    continue
                matches.append((r, c))
                used_cols.add(c)
            um_trk = [i for i in range(T) if all(i != r for r, _ in matches)]
            um_det = [j for j in range(D) if j not in used_cols]
            return matches, um_trk, um_det
        else:
            pairs = [(i, j, cost[i, j]) for i in range(T) for j in range(D)]
            pairs.sort(key=lambda x: x[2])
            used_t, used_d, matches = set(), set(), []
            for i, j, c in pairs:
                if c >= BIG: break
                if i in used_t or j in used_d: continue
                used_t.add(i); used_d.add(j)
                matches.append((i, j))
            um_trk = [i for i in range(T) if i not in used_t]
            um_det = [j for j in range(D) if j not in used_d]
            return matches, um_trk, um_det

    def _reassociate_pass(self, boxes, feats, um_trk_idx, um_det_idx):
        if len(um_trk_idx) == 0 or len(um_det_idx) == 0 or feats is None or len(feats) == 0:
            return [], um_trk_idx, um_det_idx

        cand_t = [ti for ti in um_trk_idx
                  if (self.tracks[ti].age - self.tracks[ti].last_update_age) <= self.reassoc_ttl]
        if not cand_t:
            return [], um_trk_idx, um_det_idx

        T = len(cand_t)
        D = len(um_det_idx)
        cost = np.ones((T, D), dtype=float) * 1e3

        for ti_i, ti in enumerate(cand_t):
            trk = self.tracks[ti]
            tb = trk.bbox_xyxy()
            tcx, tcy = (tb[0]+tb[2])/2.0, (tb[1]+tb[3])/2.0
            for dj_j, dj in enumerate(um_det_idx):
                f = feats[dj]
                ad = trk.best_app_distance(f)
                db = boxes[dj]
                dcx, dcy = (db[0]+db[2])/2.0, (db[1]+db[3])/2.0
                motion = np.hypot(dcx - tcx, dcy - tcy)
                cost[ti_i, dj_j] = ad + self.reassoc_motion_w * (motion / 150.0)

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
            matches = []
            used_trk, used_det = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 0.75:
                    continue
                ti = cand_t[r]
                dj = um_det_idx[c]
                matches.append((ti, dj))
                used_trk.add(ti); used_det.add(dj)
            new_um_trk = [i for i in um_trk_idx if i not in used_trk]
            new_um_det = [j for j in um_det_idx if j not in used_det]
            return matches, new_um_trk, new_um_det
        else:
            pairs = []
            for r in range(T):
                for c in range(D):
                    pairs.append((r, c, cost[r, c]))
            pairs.sort(key=lambda x: x[2])
            matches = []
            used_r, used_c = set(), set()
            for r, c, v in pairs:
                if v >= 0.75: break
                if r in used_r or c in used_c: continue
                used_r.add(r); used_c.add(c)
                matches.append((cand_t[r], um_det_idx[c]))
            new_um_trk = [i for i in um_trk_idx if i not in [cand_t[r] for r in used_r]]
            new_um_det = [j for j in um_det_idx if j not in [um_det_idx[c] for c in used_c]]
            return matches, new_um_trk, new_um_det

    # ---------- 공개 API ----------

    def update(self, detections_xyxy, frame=None, confidences=None):
        boxes = [np.array(d, dtype=float) for d in detections_xyxy]
        confs = confidences if confidences is not None else [None] * len(boxes)

        # 특징 추출
        feats = None
        if frame is not None and self.feature_extractor is not None and len(boxes) > 0:
            feats = self.feature_extractor(frame, boxes)  # [N, D]

        # 예측
        for t in self.tracks:
            t.predict()
            self.disappeared[t.id] = self.disappeared.get(t.id, 0) + 1

        # 1차 매칭
        matches, um_trk_idx, um_det_idx = self._match(boxes, feats)

        # 2차 재결합
        if feats is not None and len(um_trk_idx) and len(um_det_idx):
            rematch, um_trk_idx, um_det_idx = self._reassociate_pass(boxes, feats, um_trk_idx, um_det_idx)
            if rematch:
                matches.extend(rematch)

        # 매칭 업데이트
        for ti, di in matches:
            trk = self.tracks[ti]
            meas = xyxy_to_cyah(boxes[di])
            feat = None if feats is None else feats[di]
            trk.update(meas, confs[di], feat, n_init=self.n_init)
            self.disappeared[trk.id] = 0

        # 신규 트랙
        for di in um_det_idx:
            feat = None if feats is None else feats[di]
            self._start_track(boxes[di], confs[di], feat)

        # 오래된 트랙 제거
        self._remove_dead()

        # 결과(확정된 것만)
        self.objects.clear()
        self.bboxes.clear()
        for t in self.tracks:
            if not t.confirmed:
                continue
            bx = t.bbox_xyxy()
            self.bboxes[t.id] = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
            c = t.centroid()
            self.objects[t.id] = c
            self.paths[t.id].append(c)

        return self.objects

    def iter_active(self, confirmed_only=True):
        for t in self.tracks:
            if confirmed_only and not t.confirmed:
                continue
            yield t

    def get_paths(self, confirmed_only=True):
        out = {}
        for obj_id, path in self.paths.items():
            if confirmed_only:
                if any((t.id == obj_id and t.confirmed) for t in self.tracks):
                    out[obj_id] = list(path)
            else:
                out[obj_id] = list(path)
        return out
