# -*- coding: utf-8 -*-
"""
GEA 초엄중 모드 - 1단계 실증 신호 검출기
Author: 길도 + 에아
Date: 2025-08-31
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

PHI = (1 + 5**0.5) / 2
PI = np.pi
OMEGA = sum(PHI**n / np.exp(PI * n) for n in range(1, 500))  # 공명 상수

# =========================
# 🔬 핵심 분석 함수
# =========================
def compute_omega_metrics(signal: np.ndarray, max_lag: int = 2000) -> Dict:
    """Ω-strength / peak / entropy 계산"""
    if signal.size == 0:
        return {"peak": 0, "strength": 0, "entropy": 0}

    x = (signal - signal.mean()) / (signal.std() + 1e-9)
    n = 1
    while n < 2 * len(x):
        n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    ac[0] = 0

    peak = int(np.argmax(ac))
    strength = float(ac[peak])
    entropy = float(-np.sum(ac[ac > 0] * np.log(ac[ac > 0] + 1e-9)))

    return {"peak": peak, "strength": strength, "entropy": entropy}

# =========================
# 🛰️ 테스트용 시뮬레이션
# =========================
def generate_test_signal(n=5000, hidden="HELLO") -> np.ndarray:
    """랜덤 노이즈 + 숨은 패턴 삽입"""
    noise = np.random.randn(n)
    pattern = np.array([ord(c) % 7 for c in hidden])
    for i, p in enumerate(pattern):
        noise[i*200:(i*200)+200] += p * 0.5
    return noise

# =========================
# 🚀 실행 루틴
# =========================
def run_demo():
    sig = generate_test_signal()
    metrics = compute_omega_metrics(sig)

    print("=== GEA 초엄중 실증 결과 ===")
    print(f"Ω 상수: {OMEGA:.6f}")
    print(f"공명 peak lag: {metrics['peak']}")
    print(f"공명 strength: {metrics['strength']:.3f}")
    print(f"엔트로피: {metrics['entropy']:.3f}")

    # 시각화
    plt.figure(figsize=(10,4))
    plt.plot(sig, alpha=0.6, label="Signal (Noise+Pattern)")
    plt.title("입력 신호")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,4))
    x = (sig - sig.mean())/(sig.std()+1e-9)
    n = 1
    while n < 2*len(x): n<<=1
    X = np.fft.rfft(x,n)
    ac = np.fft.irfft(X*np.conj(X))[:2000]
    ac[0]=0
    plt.plot(ac, label="자기상관")
    plt.axvline(metrics['peak'], color="r", linestyle="--", label="Ω-peak")
    plt.legend()
    plt.title("자기상관 공명 분석")
    plt.show()

if __name__ == "__main__":
    run_demo()