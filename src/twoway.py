#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""twoway.py — 양방향 정합. '관측→벽' 뿐 아니라 '벽→관측' 도 본다.

기존 점수는 관측 점이 벽 근처면 좋다고만 봐서, 벽이 빽빽한 지도가 무조건 이겼다
(3F 27.8%·4F 28.3% 가 판정 86% 독점). 대조점수(최고점-평균)로 상쇄하려 했더니
반대로 성긴 지도가 이겼다(1F·6F·9F). 우회로로는 안 된다.

양방향: 후보 자세에서 36개 섹터로
  d_map = 지도상 첫 벽까지 거리(m)      d_obs = 그 방향 관측 거리(m)
  둘 다 있으면        |d_map - d_obs| 벌점   (구조가 맞아야 작다)
  지도엔 벽, 관측엔 없음 → 벌점 (지도 벽이 설명 안 됨)  ★기존에 없던 항
  관측엔 벽, 지도엔 없음 → 벌점 (관측이 설명 안 됨)
비용 때문에 기존 점수 상위 N개 후보만 재채점한다.
"""
import numpy as np
K = 36                      # 섹터 수 (10도)
RMAX = 18.0                 # 최대 사거리(m)

def sector_obs(pts, hd, k=K, rmax=RMAX):
    """센서프레임 점군 → 섹터별 관측거리(m). 없으면 nan."""
    r = np.hypot(pts[:,0], pts[:,1])
    a = (np.arctan2(pts[:,1], pts[:,0]) + hd) % (2*np.pi)
    s = (a / (2*np.pi) * k).astype(int) % k
    out = np.full(k, np.nan)
    ok = (r > 0.3) & (r < rmax)
    for i in range(k):
        m = ok & (s == i)
        if m.sum() >= 2: out[i] = np.percentile(r[m], 20)
    return out

def sector_map(wall, cx, cy, ppm, k=K, rmax=RMAX, step=2.0):
    """지도에서 후보 (cx,cy) 로부터 섹터별 첫 벽까지 거리(m). 없으면 nan."""
    H, W = wall.shape
    ang = (np.arange(k) + 0.5) / k * 2*np.pi
    rr = np.arange(3.0, rmax*ppm, step)
    out = np.full(k, np.nan)
    for i, th in enumerate(ang):
        xs = np.clip((cx + rr*np.cos(th)).astype(int), 0, W-1)
        ys = np.clip((cy - rr*np.sin(th)).astype(int), 0, H-1)
        hit = wall[ys, xs]
        if hit.any(): out[i] = rr[np.argmax(hit)] / ppm
    return out

def twoway_score(wall, cx, cy, ppm, d_obs, miss=4.0, tol=0.6):
    """양방향 잔차 점수(클수록 좋음, 0 이 최대)."""
    d_map = sector_map(wall, cx, cy, ppm)
    both = np.isfinite(d_obs) & np.isfinite(d_map)
    pen = 0.0; n = 0
    if both.any():
        e = np.abs(d_map[both] - d_obs[both])
        pen += np.minimum(e, miss).sum(); n += both.sum()
    only_map = np.isfinite(d_map) & ~np.isfinite(d_obs)   # ★지도 벽이 설명 안 됨
    only_obs = np.isfinite(d_obs) & ~np.isfinite(d_map)
    pen += miss*(only_map.sum() + only_obs.sum()); n += only_map.sum() + only_obs.sum()
    return -pen/max(n,1), int(both.sum()), int(only_map.sum()), int(only_obs.sum())
