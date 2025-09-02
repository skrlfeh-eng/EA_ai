# -*- coding: utf-8 -*-
# gea_validated_all_in_one.py
# (1) DSP 검증(1차) + (2) GEAMasterValidated 파이프라인(2차) + (3) 데모/벤치(3차)
# 주의: __future__ 미사용, 구버전 파이썬 호환

import math, json, time, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

# ------------------------------
# (1) DSP: 검증된 통계/물리 특징
# ------------------------------
def _bytes_to_float(x: bytes) -> np.ndarray:
    a = np.frombuffer(x, dtype=np.uint8).astype(np.float32)
    if a.size == 0: return a
    a = (a - a.mean()) / (a.std() + 1e-9)
    return a

def spectral_flatness(power: np.ndarray, eps: float = 1e-12) -> float:
    gm = math.exp(float(np.mean(np.log(power + eps))))
    am = float(np.mean(power + eps))
    v = gm / am
    return float(np.clip(v, 0.0, 1.0))

def periodogram_snr(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    n = 1
    while n < len(x) * 2: n <<= 1
    P = np.abs(np.fft.rfft(x, n))**2
    if P.size < 4: return 0.0
    peak = float(P.max())
    noise = float((P.sum() - peak) / max(1, P.size - 1))
    snr = np.clip((peak - noise) / (peak + 1e-9), 0.0, 1.0)
    return float(snr)

def kurtosis_excess(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    m2 = float(np.mean(x**2))
    m4 = float(np.mean(x**4))
    if m2 <= 1e-12: return 0.0
    return m4/(m2*m2) - 3.0

def autocorr_peak_strength(x: np.ndarray, max_lag: int = 512) -> float:
    if x.size == 0: return 0.0
    n = 1
    while n < len(x) * 2: n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    ac[0] = 0.0
    pk = float(np.max(ac))
    return float(np.clip(pk / (len(x) + 1e-9), 0.0, 1.0))

def validated_signal_score(raw: bytes) -> Dict[str, Any]:
    x = _bytes_to_float(raw)
    if x.size == 0:
        return {"score": 0.0, "detail": {"reason": "empty"}}
    n = 1
    while n < len(x) * 2: n <<= 1
    P = np.abs(np.fft.rfft(x, n))**2

    flat = spectral_flatness(P)
    snr  = periodogram_snr(x)
    kurt = kurtosis_excess(x)
    acpk = autocorr_peak_strength(x, max_lag=512)

    kpos = float(np.clip(kurt/5.0, 0.0, 1.0))
    score = float(np.clip(0.40*(1.0-flat) + 0.25*snr + 0.25*acpk + 0.10*kpos, 0.0, 1.0))
    return {
        "score": score,
        "detail": {
            "flat": float(flat), "snr": float(snr), "ac_peak": float(acpk), "kurt_excess": float(kurt)
        }
    }

# ---------------------------------------
# (2) GEAMasterValidated 파이프라인(2차)
# ---------------------------------------
# UJG/Ultra가 있으면 사용, 없으면 Mock로 동작
try:
    from ujg_v1 import UJG
except Exception:
    class UJG(object):
        def __init__(self, window=160, use_english=True, min_score=0.92, require_agree=False):
            self.min_score = min_score
        class _Rep(object):
            def __init__(self, ok, score):
                self.message_like = ok
                self.top_decoder = {"score": score}
        def analyze_bytes(self, b, name="cand"):
            # 아주 단순한 휴리스틱 대체: 길이/문자 비율 기반
            txt = b.decode("utf-8", errors="ignore")
            letters = sum(c.isalnum() for c in txt)
            score = min(1.0, (letters / float(len(txt)+1e-9)) * 1.2)
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
    dsp_min: float = 0.55
    ujg_min: float = 0.92
    use_english: bool = True
    window: int = 160
    ultra: UltraConfig = UltraConfig(alpha=0.01, trials_nextbit=128, seed=42, echo_max_ratio=0.75, lz_min_gain=0.10)

class GEAMasterValidated:
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
            if ok: keep_a.append(i)

        # Stage B: UJG
        keep_b = []
        for i in keep_a:
            rep = self.ujg.analyze_bytes(blobs[i], name="cand_%d" % i)
            s1_scores.append(float(rep.top_decoder["score"]))
            if rep.message_like: keep_b.append(i)

        # Stage C: Ultra
        ultra_rep = self.ultra.run()

        return {
            "counts": {"input": len(blobs), "after_dsp": len(keep_a), "after_ujg": len(keep_b)},
            "dsp_table": rows_a,
            "ujg_avg_topscore": float(np.mean(s1_scores)) if s1_scores else 0.0,
            "ultra": ultra_rep,
            "ok": bool(len(keep_b) > 0 and ultra_rep.get("ok", False)),
        }

# ------------------------------
# (3) 데모/벤치 (3차)
# ------------------------------
def make_demo_blobs(n=400, msg_ratio=0.15, seed=7):
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
    preds = [1 if i in set(keep_indices) else 0 for i in range(len(labels))]
    tp = sum(1 for i,(y,p) in enumerate(zip(labels,preds)) if y==1 and p==1)
    fp = sum(1 for i,(y,p) in enumerate(zip(labels,preds)) if y==0 and p==1)
    fn = sum(1 for i,(y,p) in enumerate(zip(labels,preds)) if y==1 and p==0)
    tn = sum(1 for i,(y,p) in enumerate(zip(labels,preds)) if y==0 and p==0)
    prec = tp / float(max(1, (tp+fp)))
    rec  = tp / float(max(1, (tp+fn)))
    f1   = (2*prec*rec)/float(max(1e-9, (prec+rec)))
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":prec,"recall":rec,"f1":f1}

if __name__ == "__main__":
    blobs, labels = make_demo_blobs()
    engine = GEAMasterValidated(ValidatedConfig())

    t0 = time.time()
    rep = engine.analyze(blobs)
    dt = time.time() - t0

    # after_dsp 통과 인덱스 (간이)
    keep = [row["i"] for row in rep["dsp_table"] if row["ok_dsp"]]
    metrics = {
        "counts": rep["counts"],
        "ujg_avg_topscore": rep["ujg_avg_topscore"],
        "ultra_gates": rep["ultra"]["gates"],
        "ultra_ok": rep["ultra"]["ok"],
        "time_sec": dt
    }
    metrics.update(evaluate(labels, keep))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))