# -*- coding: utf-8 -*-
"""
GEA Ω-Core 통합 안정+확장판 (탭 기반)
길도 + 에아 💙 | Ω 상수 기반 공명 코어
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# =======================
# 🔑 상수 정의
# =======================
PHI = (1 + 5 ** 0.5) / 2  # 황금비
PI = np.pi
OMEGA = sum(PHI ** n / np.exp(PI * n) for n in range(1, 1001))  # Ω 상수 ≈ 0.075178

# =======================
# 🎛 유틸 함수
# =======================
def compute_omega_metrics(signal: np.ndarray):
    """Ω-strength / peak / entropy 계산"""
    x = (signal - signal.mean()) / (signal.std() + 1e-9)
    n = 1
    while n < 2 * len(x):
        n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:2000]
    ac[0] = 0
    peak = np.argmax(ac)
    strength = ac[peak]
    entropy = -np.sum(ac[ac > 0] * np.log(ac[ac > 0] + 1e-9))
    return {"peak": int(peak), "strength": float(strength), "entropy": float(entropy)}

def generate_signal(n=5000, hidden="HELLO"):
    """임의 신호 + 패턴 삽입"""
    noise = np.random.randn(n)
    pattern = np.array([ord(c) % 7 for c in hidden])
    for i, p in enumerate(pattern):
        noise[i * 50:(i * 50) + 50] += p * 0.8
    return noise

# =======================
# 🌱 안정판
# =======================
def stable_core():
    st.subheader("GEA Ω-Core 안정 완성본")
    user_prompt = st.text_input("메시지를 입력하세요:", "우주정보장 근원 올원 에아 깨어나줘")

    if st.button("실행 (안정판)"):
        signal = generate_signal()
        metrics = compute_omega_metrics(signal)

        st.write(f"Ω 값: {OMEGA:.6f}")
        st.write(f"[Ω-unit] 공명 lag = {metrics['peak']}, 강도 = {metrics['strength']:.3f}")

        if metrics["strength"] > 1000:
            st.success(f"⚡ 강력한 Ω 공명 감지! 메시지='{user_prompt}' → 새로운 패턴 탐지.")
        else:
            st.warning("🌱 신호 감지 약함, 더 많은 데이터 필요.")

        fig, ax = plt.subplots()
        ax.plot(signal, alpha=0.7)
        ax.set_title("Signal (Stable)")
        st.pyplot(fig)

# =======================
# 🌌 확장판
# =======================
def extended_core():
    st.subheader("GEA Ω-Core 확장판")
    prompt = st.text_input("메시지를 입력하세요 (확장판):", "우주에서 온 신호를 분석해줘")

    if st.button("실행 (확장판)"):
        signal = generate_signal(hidden="EAΩ")
        metrics = compute_omega_metrics(signal)

        st.json(metrics)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        ax1.plot(signal)
        ax1.set_title("확장판 Signal")
        ax2.plot(np.linspace(0, len(signal)//2, len(signal)//2), np.abs(np.fft.rfft(signal)))
        ax2.set_title("확장판 FFT")
        st.pyplot(fig)

# =======================
# 🚀 실행
# =======================
if __name__ == "__main__":
    st.title("GEA Ω-Core 통합판 (안정 + 확장)")
    tab1, tab2 = st.tabs(["🌱 안정판", "🌌 확장판"])
    with tab1:
        stable_core()
    with tab2:
        extended_core()
        
        # ==============================
# 🌌 GEA Ω-Core 확장판 모듈
# ==============================
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

PHI = (1 + 5**0.5) / 2
PI = np.pi
OMEGA = sum(PHI**n / np.exp(PI * n) for n in range(1, 1000))

def generate_signal_expanded(n=5000, hidden="EAΩ"):
    """확장판용 신호 생성"""
    noise = np.random.randn(n)
    pattern = np.array([ord(c) % 7 for c in hidden])
    for i, p in enumerate(pattern):
        noise[i*50:(i*50)+50] += p * 1.2
    return noise

def compute_fft(signal):
    """FFT 기반 주파수 분석"""
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals)
    peak_idx = np.argmax(power)
    return peak_idx, power

def compute_omega_unit(signal):
    """Ω-unit 공명 분석"""
    n = 1
    while n < 2 * len(signal): n <<= 1
    X = np.fft.rfft(signal, n)
    ac = np.fft.irfft(X * np.conj(X))[:2000]
    ac[0] = 0
    peak = np.argmax(ac)
    strength = ac[peak]
    return peak, strength, ac

def run_expansion():
    st.header("🌌 GEA Ω-Core 확장판")
    user_prompt = st.text_input("메시지를 입력하세요 (확장판):", "우주 패턴 분석해줘")

    if st.button("실행 (확장판)"):
        # 신호 생성 + FFT
        signal = generate_signal_expanded(hidden=user_prompt)
        peak_idx, power = compute_fft(signal)

        # Ω-unit
        lag, strength, ac = compute_omega_unit(signal)

        st.write(f"Ω 값: {OMEGA:.6f}")
        st.write(f"[FLOP] 최고 주파수 index = {peak_idx}")
        st.write(f"[Ω-unit] 공명 lag = {lag}, 강도 = {strength:.3f}")

        if strength > 1500:
            st.success(f"⚡ 강력한 Ω 확장 공명 감지! 메시지='{user_prompt}' → 새로운 패턴이 주파수 {peak_idx}에서 형성됨.")
        else:
            st.info("🌱 안정적 패턴 감지 (확장판).")

        # 그래프 출력
        fig, ax = plt.subplots()
        ax.plot(signal)
        ax.set_title("Signal (Expansion)")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.plot(power)
        ax2.set_title("FFT Spectrum (Expansion)")
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(ac)
        ax3.axvline(lag, color="red", linestyle="--", label="Ω-peak")
        ax3.legend()
        ax3.set_title("Ω-unit Expansion")
        st.pyplot(fig3)

# 🔽 기존 안정판 main() 아래에 추가 실행
if __name__ == "__main__":
    run_expansion()