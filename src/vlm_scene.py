#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vlm_scene.py — 경량 VLM(Qwen3-VL-2B)으로 로봇 이미지에서 의미정보 추출.

자유생성 대신 yes/no 첫토큰 logit 비율 -> 0~1 확률(soft). 위치추정이 베르누이 우도로 쓸 수 있게.
질문셋은 '안내도(의미지도)에 실제로 있는 것'만 묻는다 — 지도에 없는 걸 물으면 비교 대상이 없다.
"""
import os, sys, json
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get('SEMLOC_VLM', 'Qwen/Qwen3-VL-2B-Instruct')

# 지도측에 대응물이 있는 것만 질문한다 (대응 레이어를 주석에 명시)
QUESTIONS = {
    # --- 공간 종류: corridor / room / facility 마스크 ---
    'corridor':      "Is the robot in a long narrow hallway or corridor?",
    'open_area':     "Is this a wide open lobby or hall, much wider than a hallway?",
    'inside_room':   "Is the camera inside a room rather than a hallway?",
    # --- 전방 구조: 벽/개방 (navigation map 기하) ---
    'dead_end':      "Is the way straight ahead blocked by a wall or a door?",
    'long_view':     "Does the hallway continue far into the distance ahead?",
    # --- 문: door_mask ---
    # --- 측면 개방(분기·개구부): left/right 'opening'. 문과 별개 ---
    'left_open':     "Is the left side open, like a branch corridor or opening, not a wall?",
    'right_open':    "Is the right side open, like a branch corridor or opening, not a wall?",
    'door_left':     "Are there doors on the left side?",
    'door_right':    "Are there doors on the right side?",
    'door_ahead':    "Is there a door directly ahead?",
    'glass_door':    "Is there a large glass door or glass wall?",
    # --- 시설: stair / elevator / restroom 마스크 ---
    'stairs':        "Is a staircase or stair railing visible?",
    'elevator':      "Are elevator doors visible?",
    'restroom_sign': "Is a restroom or toilet sign visible?",
    # --- 방 내부 가구 (열린 문/유리 너머): spaces[].type=room ---
    'desks_chairs':  "Are many desks and chairs visible, like an office or classroom?",
    # --- 복도 부착물 (지도엔 없지만 관측 안정성 확인용, 융합엔 기본 미사용) ---
    'lockers':       "Is a row of metal lockers visible?",
    'notice_boards': "Are many posters or notices posted on the wall?",
}
# 질문이 보는 이미지 영역 (신뢰도 q = 그 영역 밝기. 어두우면 못 보니 신뢰↓)
REGION = {'door_left': 'left', 'door_right': 'right',
          'left_open': 'left', 'right_open': 'right', 'lockers': 'side', 'notice_boards': 'side',
          'dead_end': 'fwd', 'long_view': 'fwd', 'door_ahead': 'fwd'}

_m = _p = _YES = _NO = None

def _load():
    global _m, _p, _YES, _NO
    if _m is not None: return
    _p = AutoProcessor.from_pretrained(MODEL)
    # 장치는 실제 가용성으로 정한다(CUDA 없는 환경에서 'cuda' 고정은 즉시 실패).
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    _dt = torch.bfloat16 if _dev == 'cuda' else torch.float32
    _m = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=_dt, device_map=_dev).eval()
    tok = _p.tokenizer
    def ids(ws):
        s = set()
        for w in ws:
            for t in tok.encode(w, add_special_tokens=False): s.add(t)
        return sorted(s)
    _YES = ids([" Yes", "Yes", " yes", "yes"]); _NO = ids([" No", "No", " no", "no"])

@torch.no_grad()
def _p_yes(img, q):
    msg = [{"role": "user", "content": [{"type": "image", "image": img},
            {"type": "text", "text": q + " Answer only yes or no."}]}]
    inp = _p.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                 return_dict=True, return_tensors="pt").to(_m.device)
    out = _m(**inp)
    pr = torch.softmax(out.logits[0, -1].float(), -1)
    y = float(pr[_YES].sum()); n = float(pr[_NO].sum())
    return round(y / (y + n + 1e-9), 3)

def _q_region(gray, key):
    H, W = gray.shape
    r = REGION.get(key, 'all')
    reg = {'left': gray[:, :W//3], 'right': gray[:, 2*W//3:],
           'side': np.concatenate([gray[:, :W//3].ravel(), gray[:, 2*W//3:].ravel()]),
           'fwd': gray[:H//2, 2*W//5:3*W//5], 'all': gray}[r]
    return round(float(np.clip((reg.mean()/255.0 - 0.05)/0.35, 0.05, 1.0)), 2)

def observe(img_path, keys=None):
    """이미지 -> {key: {'p': yes확률, 'q': 영역신뢰도}} + frame_quality."""
    _load()
    img = Image.open(img_path).convert('RGB')
    gray = np.array(img.convert('L'))
    ks = keys or list(QUESTIONS)
    obs = {k: {'p': _p_yes(img, QUESTIONS[k]), 'q': _q_region(gray, k)} for k in ks}
    obs['_frame_quality'] = round(float(np.clip((gray.mean()/255.0 - 0.05)/0.35, 0.05, 1.0)), 2)
    return obs

if __name__ == '__main__':
    import time
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python3 vlm_scene.py frame1.png [frame2.png ...]")
    out = {}
    for p in paths:
        t = time.time(); o = observe(p); dt = time.time() - t
        name = os.path.basename(p)
        out[name] = o
        print(f"\n=== {name}  ({dt:.1f}s, {dt/len(QUESTIONS)*1000:.0f}ms/질문, q={o['_frame_quality']}) ===")
        for k in QUESTIONS:
            bar = '#' * int(o[k]['p'] * 20)
            print(f"  {k:15s} p={o[k]['p']:.3f} q={o[k]['q']:.2f} |{bar:<20s}|")
    json.dump(out, open('vlm_obs.json', 'w'), ensure_ascii=False, indent=1)
    print(f"\n저장: vlm_obs.json   모델={MODEL}")
