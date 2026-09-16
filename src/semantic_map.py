#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantic_map.py — 의미지도에서 후보 pose별 '여기서 뭐가 보여야 하는가' 계산.

입력: 안내도에서 만든 4F 의미지도(복도/방/문/계단/화장실 마스크) + 위치추정용 레이어
    (seed_allowed / no_go / building_footprint / elevator). 모두 안내도 픽셀 프레임이라 정합 불필요.
출력: 후보격자 x yaw 별 expected[cue] in [0,1].  VLM 관측과 베르누이로 비교.
"""
import os, json
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

SEM = os.environ.get('SEMLOC_SEM_DIR', 'maps/4F/semantic')
MCL = os.environ.get('SEMLOC_MCL_DIR', 'maps/4F/mcl')
LOCMAP = os.environ.get('SEMLOC_LOCMAP', 'maps/4F/4F_map.png')
CORRIDOR_M = 94.6

# 융합에 쓰는 cue = 지도에 대응물이 있는 것만. (lockers/notice_boards/glass_door 는 지도에 없음)
FUSED_CUES = ['corridor', 'open_area', 'inside_room', 'dead_end', 'long_view',
              'left_open', 'right_open',
              'door_left', 'door_right', 'door_ahead', 'stairs', 'elevator',
              'restroom_sign', 'desks_chairs']

# ─────────────────────────────────────────────────────────────────────────────
# ★카메라 장착
#   로즈백 /tf_static 에 **base_link -> camera_link 가 없다**. base_link 에서 나가는 변환은
#   hesai_lidar(yaw+90, t=0.145,0,0.110) / imu / radar(yaw180) / Head_upper(t=0.285,0,0.010)
#   네 개뿐이고 카메라는 camera_link 내부 체인만 있다. 즉 카메라가 로봇 어디에 어느 방향으로
#   붙었는지는 데이터로 알 수 없다.  → 아래를 **가정이 아니라 선언된 사양**으로 둔다.
#   실측(장착각·틸트)이 확보되면 이 값만 바꾸면 된다.
CAMERA_MOUNT = dict(
    # ── 방향 ──────────────────────────────────────────────────────────────
    yaw_deg   = 0.0,     # 항상 전방 주시 (base_link +x). 운용자 확인 + 데이터 검증됨.
    pitch_deg = 3.0,     # ★거의 전방이나 아주 살짝 위. 운용자 선언값.
                         #   측정(참고): 카메라 광축 vs 바닥 = -0.81도 (거의 수평, 보행 중
                         #   ±1~2도 변동). 몸체가 보행 중 앞으로 숙여 상쇄되는 것으로 보이나,
                         #   /utlidar/imu 는 데이터 축이 뒤집혀 있어
                         #   '카메라 vs 몸체'를 데이터로 분리할 수 없다.
                         #   정확한 장착각을 알면 이 값을 교체할 것.
    roll_deg  = 0.0,
    # ── 위치 ──────────────────────────────────────────────────────────────
    offset_m  = 0.285,   # base_link 앞 0.285 m (Head_upper 기준 추정)
    height_m  = 0.396,   # ★측정: depth 바닥평면 피팅 중앙값 (Go2 실제 높이와 부합)
    # ── 광학 ──────────────────────────────────────────────────────────────
    hfov_deg = 69.0,     # RealSense D435i RGB
    vfov_deg = 42.0,
    intrinsics = dict(fx=606.80, fy=606.75, cx=310.49, cy=240.52, w=640, h=480),
    # ── 출처 ──────────────────────────────────────────────────────────────
    source   = "declared (tf_static 에 base_link->camera_link 없음)",
    # 주의: 현재 지도 모델은 2D(yaw만) 이라 pitch/roll 은 점수에 반영되지 않는다.
    #   반영하려면 랜드마크 가시성의 '높이 구간'(예: 화장실 표지 ~2m, 호실번호 ~2m)을
    #   pitch·vfov 로 잘라내는 단계가 필요하다.
    # ★데이터로 검증됨: depth 전방거리 vs LiDAR 방향별 자유거리 상관을
    #   148개 시각에서 θ=0~350° 스윕 → θ=0° 에서 단봉 최대(0.636), 180° 는 0.103.
    #   즉 카메라 광축 = 점군 프레임 +x = 로봇 정면. yaw_deg=0 확인.
    verified = "depth-LiDAR 상관 (corr 0.636 @0°, 0.103 @180°, n=148)",
)

FOV = np.deg2rad(CAMERA_MOUNT['hfov_deg'] / 2)   # 시야 반각
R_BIG = 20.0                  # 계단 등 큰 구조 가시거리(m)
R_SIGN = 14.0                 # 표지/엘베
R_DOOR = 9.0                  # 문 식별 거리
SIDE_HALF = np.deg2rad(45)    # '좌/우' 로 볼 각도폭


def _mask(path):
    return np.array(Image.open(path).convert('L')) > 127


def _centroids(m, min_px=40):
    lab, n = ndi.label(m)
    out = []
    for i in range(1, n + 1):
        s = (lab == i).sum()
        if s < min_px: continue
        cy, cx = ndi.center_of_mass(lab == i)
        out.append((cx, cy, s))
    return np.array(out).reshape(-1, 3)


def _frontier(region, free, grow=6, maxpts=40):
    """시설/문 영역이 '자유공간에 면한' 부분 = 로봇이 실제로 볼 수 있는 개구부.
    영역 중심은 벽 안쪽이라 시선이 항상 막힌다 -> 프론티어를 겨냥해야 한다."""
    fr = ndi.binary_dilation(region, iterations=grow) & free
    ys, xs = np.where(fr)
    if len(xs) == 0:                      # 면한 자유공간이 없으면 영역 자체를 씀
        ys, xs = np.where(region)
        if len(xs) == 0: return np.zeros((0, 2))
    if len(xs) > maxpts:                  # 균등 서브샘플
        k = np.linspace(0, len(xs) - 1, maxpts).astype(int)
        xs, ys = xs[k], ys[k]
    return np.c_[xs, ys].astype(float)


class SemanticMap:
    def __init__(self, grid_x, grid_y, verbose=True):
        self.v = verbose
        im = np.array(Image.open(LOCMAP).convert('L'))
        self.H, self.W = im.shape
        self.wall = im < 50; self.free = im > 200
        self.PPM = self._scale()
        # --- 의미 레이어 ---
        self.corridor = _mask(f'{SEM}/4F_corridor_mask.png')
        self.room = _mask(f'{SEM}/01_fused_room_mask.png')
        self.seed = _mask(f'{MCL}/seed_allowed.png')
        self.nogo = _mask(f'{MCL}/no_go.png')
        self.footprint = _mask(f'{MCL}/building_footprint.png')   # 건물 내부
        # --- 랜드마크 점 (마스크 연결성분 중심) ---
        self.stairs   = _frontier(_mask(f'{SEM}/4F_stair_mask.png'), self.free)
        self.restroom = _frontier(_mask(f'{SEM}/4F_restroom_mask.png'), self.free)
        self.elevator = _frontier(_mask(f'{MCL}/elevator.png'), self.free)
        self.doors    = _frontier(_mask(f'{SEM}/4F_door_mask.png'), self.free, grow=4, maxpts=60)
        # --- 후보격자 ---
        GXX, GYY = np.meshgrid(grid_x, grid_y, indexing='ij')
        self.shp = GXX.shape
        self.G = np.c_[GXX.ravel(), GYY.ravel()].astype(float)
        self.N = len(self.G)
        gi = self.G.astype(int)
        self.in_free = self.free[gi[:, 1], gi[:, 0]]
        self.in_cor = self.corridor[gi[:, 1], gi[:, 0]]
        self.in_room = self.room[gi[:, 1], gi[:, 0]]
        self.in_seed = self.seed[gi[:, 1], gi[:, 0]]
        self.in_nogo = self.nogo[gi[:, 1], gi[:, 0]]
        self.in_bldg = self.footprint[gi[:, 1], gi[:, 0]]
        # ★후보 허용 영역: 자유공간 ∩ 건물내부 ∩ ~(no_go·화장실·계단).
        #   방 안은 남긴다 — 건물 밖만 확실히 배제한다.
        #   (안내도는 외곽선을 닫아 그리지 않아 복도 자유공간이 이미지 테두리까지 이어진다).
        #   진단: 배제 조건이 '자유∩건물안' 뿐이라 허용 3370곳 중 정상 통행구역은
        #     732곳(22%)뿐이었고, no_go 1294 / 화장실 69 / 계단 34 가 후보로 경쟁했다.
        #     실제로 5지점 top-20 (100개) 중 24개가 no_go(건물 아래 날개), 3개가 화장실 안.
        #     dev 8시각 A/B: t=179 가 3위 74.73m -> 1위 2.30m, 나머지 7시각 불변(악화 0건).
        #   레이어는 모두 의미지도 표준 산출물이라 다른 층에도 그대로 적용된다.
        self.restroom_area = _mask(f'{SEM}/4F_restroom_mask.png')
        self.stair_area = _mask(f'{SEM}/4F_stair_mask.png')
        self.in_restroom = self.restroom_area[gi[:, 1], gi[:, 0]]
        self.in_stair = self.stair_area[gi[:, 1], gi[:, 0]]
        self.allowed = self.in_free & self.in_bldg
        if os.environ.get('SEMLOC_LOOSE_ALLOW') != '1':      # =1 이면 옛 동작(진단용)
            self.allowed &= ~self.in_nogo & ~self.in_restroom & ~self.in_stair
        if self.v:
            print(f"[SemanticMap] PPM={self.PPM:.3f}px/m  후보 {self.N}개 "
                  f"(free {self.in_free.sum()}, corridor {self.in_cor.sum()}, "
                  f"room {self.in_room.sum()}, seed {self.in_seed.sum()}, no_go {self.in_nogo.sum()})")
            print(f"  ★허용후보(자유∩건물안∩~no_go·화장실·계단) {self.allowed.sum()} / "
                  f"자유공간에서 배제 {int(self.in_free.sum() - self.allowed.sum())}개 "
                  f"(건물밖 {int((self.in_free & ~self.in_bldg).sum())}, "
                  f"no_go {int((self.in_free & self.in_bldg & self.in_nogo).sum())}, "
                  f"화장실·계단 {int((self.in_free & self.in_bldg & ~self.in_nogo & (self.in_restroom | self.in_stair)).sum())})")
            print(f"  랜드마크: 계단{len(self.stairs)} 엘베{len(self.elevator)} "
                  f"화장실{len(self.restroom)} 문{len(self.doors)}")
        self._precompute()

    def _scale(self):
        """안내도 최장 가로복도 픽셀폭 / 94.6m (위치추정 노드와 동일 규칙)."""
        best = (0, 0, 0, 0)
        for r in range(5, self.H - 5):
            idx = np.where(self.free[r])[0]
            if len(idx) < 50: continue
            for s in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
                a, b = s[0], s[-1]
                if a > 2 and b < self.W - 3 and self.wall[r, a-1] and self.wall[r, b+1] and (b-a) > best[0]:
                    best = (b - a, a, b, r)
        return best[0] / CORRIDOR_M

    # ---- 가시성 기하 ----
    def _rays(self, nb=36, maxm=25.0):
        """후보별 nb방향 자유거리(m)."""
        ang = np.linspace(0, 2*np.pi, nb, endpoint=False)
        step = np.arange(0, int(maxm * self.PPM), 4)
        fd = np.full((self.N, nb), maxm)
        for k, a in enumerate(ang):
            xs = (self.G[:, None, 0] + step[None] * np.cos(a)).astype(int)
            ys = (self.G[:, None, 1] + step[None] * np.sin(a)).astype(int)
            ok = (xs >= 0) & (xs < self.W) & (ys >= 0) & (ys < self.H)
            fr = self.free[np.clip(ys, 0, self.H-1), np.clip(xs, 0, self.W-1)] & ok
            blocked = ~fr
            first = np.where(blocked.any(1), blocked.argmax(1), len(step)-1)
            fd[:, k] = step[first] / self.PPM
        return ang, fd

    def _los(self, pts, nstep=28):
        """후보->각 랜드마크 시선 통과 여부 [N, M] (벽 안 뚫으면 True)."""
        if len(pts) == 0: return np.zeros((self.N, 0), bool)
        t = np.linspace(0.08, 0.94, nstep)
        out = np.ones((self.N, len(pts)), bool)
        for j, (px, py) in enumerate(pts):
            xs = (self.G[:, None, 0] * (1-t) + px * t).astype(int)
            ys = (self.G[:, None, 1] * (1-t) + py * t).astype(int)
            out[:, j] = ~self.wall[np.clip(ys, 0, self.H-1), np.clip(xs, 0, self.W-1)].any(1)
        return out

    def _db(self, pts):
        """후보->랜드마크 거리(m)·방위각 [N,M]."""
        if len(pts) == 0: return np.zeros((self.N, 0)), np.zeros((self.N, 0))
        d = np.hypot(self.G[:, None, 0]-pts[None, :, 0], self.G[:, None, 1]-pts[None, :, 1]) / self.PPM
        b = np.arctan2(pts[None, :, 1]-self.G[:, None, 1], pts[None, :, 0]-self.G[:, None, 0])
        return d, b

    def _precompute(self):
        self.ang, self.fd = self._rays()
        # 개방도: 8m 넘게 뚫린 방향 비율
        self.p_open = np.clip((self.fd > 8).sum(1) / 10.0, 0, 1)
        # 자유공간 폭(로비 판정용): free 거리변환
        from scipy.ndimage import distance_transform_edt
        freeD = distance_transform_edt(self.free) / self.PPM
        gi = self.G.astype(int)
        self.width_m = freeD[gi[:, 1], gi[:, 0]] * 2.0     # 반경*2 ~= 폭
        for nm, pts in (('st', self.stairs), ('ev', self.elevator),
                        ('rs', self.restroom), ('dr', self.doors)):
            d, b = self._db(pts); l = self._los(pts)
            setattr(self, f'{nm}D', d); setattr(self, f'{nm}B', b); setattr(self, f'{nm}L', l)
        if self.v: print("[SemanticMap] 사전계산 완료")

    def _fwd(self, heading):
        """heading 방향 자유거리(m) [N]."""
        k = int(round((heading % (2*np.pi)) / (2*np.pi) * len(self.ang))) % len(self.ang)
        return self.fd[:, k]

    def _vis(self, D, B, L, heading, rng, half=FOV):
        """FOV·시선·거리 조건 만족하는 랜드마크가 하나라도 있나 -> 0/1 [N]."""
        if D.shape[1] == 0: return np.zeros(self.N)
        rel = np.abs(((B - heading) + np.pi) % (2*np.pi) - np.pi)
        return ((rel < half) & L & (D < rng)).any(1).astype(float)

    def expected(self, heading):
        """heading(지도 프레임 **로봇 전방각**, rad) 하에서 후보별 예상 관측확률 [N].
        카메라 광축 = heading + CAMERA_MOUNT['yaw_deg'] (전방 주시면 0도라 동일)."""
        heading = heading + np.deg2rad(CAMERA_MOUNT['yaw_deg'])   # 카메라 장착 yaw 반영
        fwd = self._fwd(heading)
        e = {}
        e['corridor']    = np.where(self.in_cor, 0.9, np.where(self.in_room, 0.1, 0.5))
        e['inside_room'] = np.where(self.in_room, 0.9, 0.1)
        e['open_area']   = np.clip((self.width_m - 3.0) / 6.0, 0.02, 0.95)
        e['dead_end']    = np.clip(1.0 - fwd / 6.0, 0.02, 0.95)
        e['long_view']   = np.clip(fwd / 15.0, 0.02, 0.98)
        # 측면 개방도: 그 방향 자유거리 -> 0~1
        #   이미지 좌표 y아래 -> 로봇 좌측 = heading - pi/2
        e['left_open']   = np.clip(self._fwd(heading - np.pi/2) / 6.0, 0.02, 0.98)
        e['right_open']  = np.clip(self._fwd(heading + np.pi/2) / 6.0, 0.02, 0.98)
        e['stairs']      = 0.05 + 0.9 * self._vis(self.stD, self.stB, self.stL, heading, R_BIG)
        e['elevator']    = 0.05 + 0.9 * self._vis(self.evD, self.evB, self.evL, heading, R_SIGN)
        e['restroom_sign'] = 0.05 + 0.9 * self._vis(self.rsD, self.rsB, self.rsL, heading, R_SIGN)
        e['door_ahead']  = 0.05 + 0.9 * self._vis(self.drD, self.drB, self.drL, heading, R_DOOR)
        # 이미지 좌표는 y가 아래 -> 전방 heading 기준 로봇 좌측 = heading - pi/2
        e['door_left']   = 0.05 + 0.9 * self._vis(self.drD, self.drB, self.drL,
                                                  heading - np.pi/2, R_DOOR, SIDE_HALF)
        e['door_right']  = 0.05 + 0.9 * self._vis(self.drD, self.drB, self.drL,
                                                  heading + np.pi/2, R_DOOR, SIDE_HALF)
        # 책상·의자: 문이 가까이 보일 때만 방 내부가 보일 수 있음 (문 닫혀 있으면 관측 0)
        near_door = self._vis(self.drD, self.drB, self.drL, heading, 4.0)
        e['desks_chairs'] = 0.05 + 0.45 * near_door
        return e


if __name__ == '__main__':
    GX = np.arange(950, 3300, 22); GY = np.arange(420, 1460, 22)
    sm = SemanticMap(GX, GY)
    print("\n=== heading 0도(동쪽)에서 예상 관측 요약 (free 후보만) ===")
    e = sm.expected(0.0)
    f = sm.in_free
    for k in FUSED_CUES:
        v = e[k][f]
        print(f"  {k:14s} mean={v.mean():.3f}  >0.5인 후보 {(v>0.5).sum():5d}/{f.sum()}")
