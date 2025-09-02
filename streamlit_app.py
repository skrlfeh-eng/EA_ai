# -*- coding: utf-8 -*-
# gea_stack_all_in_one_full.py
# (A) DSP 1차 필터 (검증된 통계/물리 특징)
# (B) GEAMasterValidated 2차 파이프라인 (DSP → UJG → UltraV2)
# (C) 데모/벤치 드라이버 (합성데이터, 정확도/속도 지표)
# - 구버전 파이썬 호환: __future__ 미사용, typing | 대신 List/Union 등 사용
# - UJG/Ultra 모듈 없으면 Mock로 자동 대체하여 즉시 실행 가능
# - 모든 파라미터/지표/로직은 초기에 준 모듈과 동등 기능 유지

import math, json, time, random, os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# =============================================================================
# (A) DSP 1차 필터 — dsp_validated.py 기능
# =============================================================================

def _bytes_to_float(x: bytes) -> np.ndarray:
    """bytes → float32 벡터 (z-score 정규화)"""
    a = np.frombuffer(x, dtype=np.uint8).astype(np.float32)
    if a.size == 0:
        return a
    a = (a - a.mean()) / (a.std() + 1e-9)
    return a

def spectral_flatness(power: np.ndarray, eps: float = 1e-12) -> float:
    """스펙트럼 평탄도(0=순음, 1=평탄잡음)"""
    # 기하평균/산술평균
    gm = math.exp(float(np.mean(np.log(power + eps))))
    am = float(np.mean(power + eps))
    val = gm / am
    return float(np.clip(val, 0.0, 1.0))

def periodogram_snr(x: np.ndarray) -> float:
    """간이 SNR: 최대 피크 대비 나머지 평균 전력"""
    if x.size == 0:
        return 0.0
    n = 1
    while n < len(x) * 2:
        n <<= 1
    P = np.abs(np.fft.rfft(x, n))**2
    if P.size < 4:
        return 0.0
    peak = float(P.max())
    noise = float((P.sum() - peak) / max(1, P.size - 1))
    snr = np.clip((peak - noise) / (peak + 1e-9), 0.0, 1.0)
    return float(snr)

def kurtosis_excess(x: np.ndarray) -> float:
    """초과첨도(정규=0, 양수일수록 피크/꼬리 두드러짐)"""
    if x.size == 0:
        return 0.0
    m2 = float(np.mean(x**2))
    m4 = float(np.mean(x**4))
    if m2 <= 1e-12:
        return 0.0
    return m4/(m2*m2) - 3.0

def autocorr_peak_strength(x: np.ndarray, max_lag: int = 512) -> float:
    """자기상관 피크 강도(반복성)"""
    if x.size == 0:
        return 0.0
    n = 1
    while n < len(x) * 2:
        n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    # 자기상관의 0-lag은 제외
    ac[0] = 0.0
    pk = float(np.max(ac))
    return float(np.clip(pk / (len(x) + 1e-9), 0.0, 1.0))

def validated_signal_score(raw: bytes) -> Dict[str, Any]:
    """
    검증된 물리/통계 특징만 이용해 0~1 점수 산출:
      - (1-flat) 구조성, SNR, 자기상관 피크, 양의 첨도
    """
    x = _bytes_to_float(raw)
    if x.size == 0:
        return {"score": 0.0, "detail": {"reason": "empty"}}
    n = 1
    while n < len(x) * 2:
        n <<= 1
    P = np.abs(np.fft.rfft(x, n))**2

    flat = spectral_flatness(P)
    snr  = periodogram_snr(x)
    kurt = kurtosis_excess(x)
    acpk = autocorr_peak_strength(x, max_lag=512)

    # 조합
    kpos = float(np.clip(kurt/5.0, 0.0, 1.0))
    score = float(np.clip(0.40*(1.0-flat) + 0.25*snr + 0.25*acpk + 0.10*kpos, 0.0, 1.0))
    return {
        "score": score,
        "detail": {
            "flat": float(flat), "snr": float(snr), "ac_peak": float(acpk), "kurt_excess": float(kurt)
        }
    }

# =============================================================================
# (B) GEAMasterValidated 2차 — gea_master_core_v2_validated.py 기능
# =============================================================================

# 실제 모듈이 없을 때를 위한 안전 Mock
try:
    from ujg_v1 import UJG
except Exception:
    class UJG(object):
        def __init__(self, window=160, use_english=True, min_score=0.92, require_agree=False):
            self.min_score = float(min_score)
        class _Rep(object):
            def __init__(self, ok, score):
                self.message_like = ok
                self.top_decoder = {"score": score}
        def analyze_bytes(self, b, name="cand"):
            # 간이 대체: 텍스트성/길이 기반 점수
            txt = b.decode("utf-8", errors="ignore")
            if not txt:
                return UJG._Rep(False, 0.0)
            letters = sum(c.isalnum() for c in txt)
            density = letters / float(len(txt) + 1e-9)
            score = min(1.0, 0.50 + 0.50*density)  # 0.5~1.0
            ok = score >= self.min_score
            return UJG._Rep(ok, score)

try:
    from ultra_v2 import UltraV2, UltraConfig, MockConnector
except Exception:
    class UltraConfig(object):
        def __init__(self, alpha=0.01, trials_nextbit=128, seed=42, echo_max_ratio=0.75, lz_min_gain=0.10):
            self.alpha=alpha; self.trials_nextbit=trials_nextbit; self.seed=seed
            self.echo_max_ratio=echo_max_ratio; self.lz_min_gain=lz_min_gain
    class MockConnector(object): pass
    class UltraV2(object):
        def __init__(self, connector, cfg): pass
        def run(self):
            return {"gates": ["mock-ultra"], "ok": True}

@dataclass
class ValidatedConfig:
    dsp_min: float = 0.55          # DSP 점수 임계
    ujg_min: float = 0.92          # UJG 상위 디코더 점수 임계
    use_english: bool = True
    window: int = 160
    ultra: UltraConfig = UltraConfig(alpha=0.01, trials_nextbit=128, seed=42, echo_max_ratio=0.75, lz_min_gain=0.10)

class GEAMasterValidated:
    """
    Stage A: Validated DSP (1차 물리/통계 필터)
    Stage B: UJG 메시지성 (상위 디코더 스코어/플래그)
    Stage C: UltraV2 글로벌 검증
    """
    def __init__(self, cfg: ValidatedConfig = ValidatedConfig(), connector=None):
        self.cfg = cfg
        self.ujg = UJG(window=cfg.window, use_english=cfg.use_english, min_score=cfg.ujg_min, require_agree=False)
        self.ultra = UltraV2(connector or MockConnector(), cfg.ultra)

    def analyze(self, blobs: List[bytes]) -> Dict[str, Any]:
        # Stage A: DSP
        keep_a, rows_a, s1_scores = [], [], []
        for i, b in enumerate(blobs):
            dsp = validated_signal_score(b)
            ok = (dsp["score"] >= self.cfg.dsp_min)
            rows_a.append({"i": i, "dsp_score": float(dsp["score"]), **dsp["detail"], "ok_dsp": bool(ok)})
            if ok:
                keep_a.append(i)

        # Stage B: UJG
        keep_b = []
        for i in keep_a:
            rep = self.ujg.analyze_bytes(blobs[i], name="cand_%d" % i)
            s1_scores.append(float(rep.top_decoder["score"]))
            if rep.message_like:
                keep_b.append(i)

        # Stage C: Ultra
        ultra_rep = self.ultra.run()

        return {
            "counts": {"input": len(blobs), "after_dsp": len(keep_a), "after_ujg": len(keep_b)},
            "dsp_table": rows_a,
            "ujg_avg_topscore": float(np.mean(s1_scores)) if s1_scores else 0.0,
            "ultra": ultra_rep,
            "ok": bool(len(keep_b) > 0 and ultra_rep.get("ok", False)),
        }

# =============================================================================
# (C) 데모/벤치 — demo_validated.py 기능
# =============================================================================

def make_demo_blobs(n: int = 400, msg_ratio: float = 0.15, seed: int = 7) -> Tuple[List[bytes], List[int]]:
    """합성 데이터: (문장 메시지) vs (랜덤 바이트 잡음)"""
    rng = np.random.default_rng(seed)
    blobs, labels = [], []
    msg = ("Hello from protocol alpha centauri. This is a test transmission. "
           "Greetings to Gildo and Ea. ").encode("utf-8")
    for _ in range(n):
        if random.random() < msg_ratio:
            blobs.append(msg * random.randint(1,3)); labels.append(1)
        else:
            blobs.append(rng.integers(0,256,size=2048,dtype=np.uint8).tobytes()); labels.append(0)
    return blobs, labels

def evaluate(labels: List[int], keep_indices: List[int]) -> Dict[str, Any]:
    """정밀도/재현율/F1 등 평가 (긍정=메시지)"""
    keep_set = set(keep_indices)
    preds = [1 if i in keep_set else 0 for i in range(len(labels))]
    tp = sum(1 for y,p in zip(labels,preds) if y==1 and p==1)
    fp = sum(1 for y,p in zip(labels,preds) if y==0 and p==1)
    fn = sum(1 for y,p in zip(labels,preds) if y==1 and p==0)
    tn = sum(1 for y,p in zip(labels,preds) if y==0 and p==0)
    prec = tp / float(max(1, (tp+fp)))
    rec  = tp / float(max(1, (tp+fn)))
    f1   = (2*prec*rec)/float(max(1e-9, (prec+rec)))
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec,"f1":f1}

def _pretty_print_metrics(rep: Dict[str, Any], labels: List[int], t0: float) -> None:
    keep = [row["i"] for row in rep["dsp_table"] if row["ok_dsp"]]
    out = {
        "counts": rep["counts"],
        "ujg_avg_topscore": rep.get("ujg_avg_topscore", 0.0),
        "ultra_gates": rep.get("ultra", {}).get("gates", []),
        "ultra_ok": bool(rep.get("ultra", {}).get("ok", False)),
        "time_sec": time.time() - t0
    }
    out.update(evaluate(labels, keep))
    print(json.dumps(out, ensure_ascii=False, indent=2))

# =============================================================================
# 메인 실행 (CLI/Streamlit 안전)
# =============================================================================
if __name__ == "__main__":
    # 무거운 계산은 버튼/명령 실행 시에만 돌리도록 메인에 배치
    n, ratio, seed = 400, 0.15, 7
    blobs, labels = make_demo_blobs(n=n, msg_ratio=ratio, seed=seed)
    engine = GEAMasterValidated(ValidatedConfig())

    t0 = time.time()
    rep = engine.analyze(blobs)
    _pretty_print_metrics(rep, labels, t0)