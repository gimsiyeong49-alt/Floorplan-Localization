#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wall_clean.py — 벽거리장에서 '가짜 벽'을 제거한다.

가짜 벽 두 종류
  ① 메워진 면 : 이진화·모폴로지에서 해치가 뭉개져 통째로 칠해진 영역.
                실제 벽은 그 지도 중앙두께(8~12px)인데 2F 는 최대 156px.
  ② 해치 구조 : 계단·에스컬레이터 기호. 선은 얇지만 촘촘히 모여 있어
                그 영역 전체가 '벽 근처'가 되어 정합점수를 공짜로 올린다.

둘 다 라이다가 보는 실제 벽면이 아니므로 DFpx 계산에서 뺀다.
임계는 그 지도 중앙두께의 배수 → 층·축척과 무관하게 일반적으로 적용된다.
"""
import numpy as np, scipy.ndimage as ndi

def clean_wall(wall, thick_mul=2.5, gap_mul=1.2, min_blob=600, verbose=False):
    """실제 벽면만 남긴 마스크.

    ① 해치의 틈(선 사이 5~10px)은 로봇이 들어갈 수 없고 라이다도 하나의 면으로 본다.
       중앙두께의 gap_mul 배로 닫기(closing) 하면 해치 영역이 하나의 덩어리가 된다.
    ② 그 뒤 두께가 중앙두께의 thick_mul 배를 넘는 덩어리를 '가짜 벽'으로 뺀다.
       실제 벽은 8~12px 인데 메워진 면은 2F 최대 156px.
    임계가 전부 그 지도 중앙두께의 배수라 층·축척과 무관하다.
    """
    th0 = 2*ndi.distance_transform_edt(wall)
    med = float(np.median(th0[wall])) if wall.any() else 1.0
    r = max(1, int(round(gap_mul*med/2)))
    closed = ndi.binary_closing(wall, ndi.generate_binary_structure(2,2), iterations=r)
    th = 2*ndi.distance_transform_edt(closed)
    blob = closed & (th > thick_mul*med)
    lab, n = ndi.label(blob)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        blob = (sz > min_blob)[lab]
    blob = ndi.binary_dilation(blob, iterations=r)        # 덩어리 가장자리까지
    real = wall & ~blob
    if verbose:
        w = wall.sum()
        print(f'    중앙두께 {med:.0f}px 닫기r={r} · 가짜벽 {100*(wall&blob).sum()/w:.1f}% · '
              f'남은 벽 {100*real.sum()/w:.1f}%')
    return real, wall & blob
