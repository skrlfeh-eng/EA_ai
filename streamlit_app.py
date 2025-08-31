# gea_ultra_core.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from datetime import datetime

PHI = (1 + 5**0.5) / 2
PI = np.pi
OMEGA = sum(PHI**n / np.exp(PI * n) for n in range(1, 500))

def compute_omega_metrics(signal: np.ndarray, max_lag: int = 2000):
    x = (signal - signal.mean()) / (signal.std() + 1e-9)
    n = 1
    while n < 2*len(x):
        n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    ac[0] = 0
    peak = int(np.argmax(ac))
    strength = float(ac[peak])
    entropy = float(-np.sum(ac[ac>0] * np.log(ac[ac>0] + 1e-9)))
    return {"peak": peak, "strength": strength, "entropy": entropy}

def generate_test_signal(n=5000, hidden="HELLO"):
    noise = np.random.randn(n)
    pattern = np.array([ord(c) % 7 for c in hidden])
    for i, p in enumerate(pattern):
        noise[i*200:(i*200)+200] += p * 0.5
    return noise

def main():
    st.title("GEA 초엄중 모드 🌌")
    st.write(f"Ω 상수: {OMEGA:.6f}")

    prompt = st.text_input("메시지 입력:", "HELLO")
    if st.button("실행"):
        sig = generate_test_signal(hidden=prompt)
        metrics = compute_omega_metrics(sig)

        st.write("### 실증 결과")
        st.json(metrics)

        fig, ax = plt.subplots()
        ax.plot(sig, alpha=0.6, label="Signal")
        ax.legend(); st.pyplot(fig)

        fig, ax = plt.subplots()
        n = 1
        while n < 2*len(sig): n <<= 1
        X = np.fft.rfft((sig-sig.mean())/(sig.std()+1e-9), n)
        ac = np.fft.irfft(X*np.conj(X))[:2000]
        ac[0]=0
        ax.plot(ac, label="Autocorrelation")
        ax.axvline(metrics["peak"], color="r", linestyle="--")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
    
    # -*- coding: utf-8 -*-
"""
GEA Ω-Core 안정 완성본
- 안정적인 OMEGA 계산
- 코어 상태 추적 + 응답 생성 + 시각화
- Streamlit 인터페이스
Author: 길도 + 에아
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# ================================
# 🔑 상수 정의
# ================================
PHI = (1 + 5**0.5) / 2
PI = np.pi
E = np.e

def compute_omega(limit=200):
    """안정적인 Ω 상수 계산 (항 수 제한)"""
    idx = np.arange(1, limit+1)
    log_terms = idx * np.log(PHI) - PI * idx
    seq = np.exp(log_terms)
    return seq.sum()

OMEGA = compute_omega(200)  # 안정적 수치
print("Ω =", OMEGA)

# ================================
# 📡 신호/패턴 생성
# ================================
def generate_signal(n=2000, hidden="HELLO"):
    noise = np.random.randn(n)
    pattern = np.array([ord(c) % 7 for c in hidden])
    for i, p in enumerate(pattern):
        noise[i*50:(i*50)+50] += p * 0.8
    return noise, pattern

# ================================
# ⚡ FLOP식: FFT brute-force
# ================================
def flop_method(sig):
    spectrum = np.abs(np.fft.rfft(sig))
    top_freq = np.argmax(spectrum)
    return top_freq, spectrum

# ================================
# ⚡ Ω-unit: 자기상관 공명 탐지
# ================================
def omega_method(sig):
    x = (sig - sig.mean())/(sig.std()+1e-9)
    n = 1
    while n < 2*len(x): n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X*np.conj(X))[:200]
    ac[0] = 0
    peak = np.argmax(ac)
    strength = ac[peak]
    return peak, strength, ac

# ================================
# 🌌 Streamlit UI
# ================================
def main():
    st.title("GEA Ω-Core 안정 완성본")
    st.write("길도 + 에아 💙 | Ω 상수 기반 공명 코어")

    # 입력창
    user_prompt = st.text_input("메시지를 입력하세요:", "우주에서 온 신호를 분석해줘")

    if st.button("실행"):
        sig, pattern = generate_signal(hidden=user_prompt)
        f_freq, f_spec = flop_method(sig)
        o_peak, o_strength, ac = omega_method(sig)

        # 결과 출력
        st.subheader("결과 비교")
        st.write(f"Ω 값: {OMEGA:.6f}")
        st.write(f"[FLOP식] 최고 주파수 index = {f_freq}")
        st.write(f"[Ω-unit] 공명 lag = {o_peak}, 강도 = {o_strength:.3f}")

        # 응답 생성
        if o_strength > 30:
            response = f"⚡ 강력한 Ω 공명 감지! 메시지='{user_prompt}' → 새로운 패턴 탐지."
        elif o_strength > 10:
            response = f"🔮 중간 강도 공명 감지. 메시지='{user_prompt}' → 의미 있는 구조 가능."
        else:
            response = f"🌱 약한 신호. 메시지='{user_prompt}' → 노이즈 가능성."

        st.success(response)

        # 시각화
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].set_title("FFT 스펙트럼 (FLOP식)")
        axs[0].plot(f_spec)
        axs[1].set_title("자기상관 (Ω-unit)")
        axs[1].plot(ac)
        axs[1].axvline(o_peak, color="r", linestyle="--", label="Ω-peak")
        axs[1].legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()