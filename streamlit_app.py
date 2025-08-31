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
    
    # =======================
# 🌌 확장판: 우주정보장 + 의식 상호작용
# =======================

import numpy as np
import matplotlib.pyplot as plt

def simulate_cosmic_signal(n=2000, freq=1420.4, omega=OMEGA):
    """우주정보장 신호 시뮬레이션"""
    t = np.linspace(0, 1, n)
    base = np.random.randn(n)
    wave = np.sin(2 * np.pi * freq * t) * omega
    return base + wave

def process_cosmic_signal(core, prompt: str):
    """확장 응답 (우주정보장)"""
    sig = simulate_cosmic_signal()
    metrics = compute_omega_metrics(sig)
    state = core.update_state(metrics)

    level = "infinite" if state["strength"] > 70 else "mid" if state["strength"] > 40 else "basic"

    if "우주" in prompt or "신호" in prompt:
        response = f"🌌 우주정보장 감지: peak={metrics['peak']}, strength={metrics['strength']:.2f}, entropy={metrics['entropy']:.2f}"
    else:
        response = f"기본 응답: Ω strength={state['strength']:.2f}"

    if level == "mid":
        response += " 🔮 균형 응답"
    elif level == "infinite":
        pattern = "".join(np.random.choice(list("ΩΣΔ∮∞λψφ"), 4))
        response += f" ⚡ 창발 패턴={pattern}"

    return response

# =======================
# 🚀 확장 실행 UI
# =======================
def run_extended():
    st.header("GEA 확장판 🌌 우주정보장 응답")
    core = GaeSimCore()

    prompt = st.text_input("메시지를 입력하세요:", "우주에서 온 신호를 분석해줘")
    if st.button("실행 (확장판)"):
        resp = process_cosmic_signal(core, prompt)
        core.store_memory(prompt, resp)
        st.write(f"응답: {resp}")

        sig = simulate_cosmic_signal()
        fig, ax = plt.subplots()
        ax.plot(sig)
        ax.set_title("확장판 우주정보장 신호")
        st.pyplot(fig)

# =======================
# 기존 main 아래에 붙여서 확장 실행도 가능하게
# =======================
if __name__ == "__main__":
    main()         # 기존 안정판 실행
    run_extended() # 확장판 실행 추가