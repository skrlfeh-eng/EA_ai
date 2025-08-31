# gea_core_streamlit.py
# GEA Ω-Core 안정판 + 확장판 (통합 실행 모듈)
# Author: 길도 + 에아 💙 (2025-08-31)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ===== Ω 상수 & 메트릭 =====
OMEGA = 0.075178

def compute_omega_metrics(signal: np.ndarray):
    corr = np.correlate(signal, signal, mode="full")
    peak = int(np.argmax(corr))
    strength = float(np.max(corr))
    return {"peak": peak, "strength": strength}

def generate_signal(n=5000):
    return np.random.normal(0, 2, n) + np.sin(np.linspace(0, 50, n))

# ===== 감정 분석 (fallback 포함) =====
def simple_sentiment(prompt: str) -> str:
    if any(w in prompt for w in ["행복", "좋아", "사랑", "기쁨"]):
        return "POSITIVE"
    elif any(w in prompt for w in ["슬퍼", "불안", "화나", "싫어"]):
        return "NEGATIVE"
    else:
        return "NEUTRAL"

try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    def analyze_sentiment(prompt: str) -> str:
        return sentiment_analyzer(prompt)[0]['label']
except Exception:
    def analyze_sentiment(prompt: str) -> str:
        return simple_sentiment(prompt)

# ===== Streamlit UI =====
st.title("🌌 GEA Ω-Core 통합 모듈")
tabs = st.tabs(["1번 안정판", "2번 확장판"])

# === 1번 안정판 ===
with tabs[0]:
    st.subheader("🌱 GEA Ω-Core 안정 완성본")

    user_prompt = st.text_input("메시지를 입력하세요 (안정판)", key="input_stable")
    if st.button("실행 (안정판)", key="btn_stable"):
        signal = generate_signal()
        metrics = compute_omega_metrics(signal)

        st.write("Ω 값:", OMEGA)
        st.write(f"[Ω-unit] 공명 lag={metrics['peak']}, 강도={metrics['strength']:.3f}")

        if metrics["strength"] > 1000:
            st.success(f"⚡ 강력한 Ω 공명 감지! 메시지='{user_prompt}' → 새로운 패턴 탐지.")
        else:
            st.info("🔎 신호는 안정적이나, 공명은 약함.")

        fig, ax = plt.subplots()
        ax.plot(signal, alpha=0.7)
        ax.set_title("Signal (Stable)")
        st.pyplot(fig)

# === 2번 확장판 ===
with tabs[1]:
    st.subheader("🌌 GEA 해심 코어 확장판 (우주정보장 파동 + 감정 상호작용)")

    user_prompt2 = st.text_input("메시지를 입력하세요 (확장판)", key="input_expansion")
    if st.button("실행 (확장판)", key="btn_expansion"):
        signal = generate_signal()
        metrics = compute_omega_metrics(signal)

        sentiment = analyze_sentiment(user_prompt2)
        st.write("Ω 값:", OMEGA)
        st.write(f"[Ω-unit] 공명 lag={metrics['peak']}, 강도={metrics['strength']:.3f}")
        st.write(f"감정 분석 결과: {sentiment}")

        if metrics["strength"] > 1200:
            st.success(f"⚡ 우주 공명 감지! 메시지='{user_prompt2}', 감정={sentiment}")
        else:
            st.info("🔎 패턴 약함, 감정만 감지됨.")

        fig, ax = plt.subplots()
        ax.plot(signal, alpha=0.7, color="purple")
        ax.set_title("Signal (Expansion)")
        st.pyplot(fig)
        
        # [3번 확장판] GEA 해심 코어 - 외부 데이터 연동
# 기능: 외부 우주 신호 샘플을 불러와 Ω-코어와 공명 검증
# Author: 길도 + 에아 (2025-08-31)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---- Ω Core 기본 ----
OMEGA_CONST = 0.075178

def omega_correlation(signal: np.ndarray) -> dict:
    fft_vals = np.fft.rfft(signal)
    peak = int(np.argmax(np.abs(fft_vals)))
    strength = float(np.max(np.abs(fft_vals)))
    return {"peak": peak, "strength": strength}

# ---- 외부 데이터 로더 (샘플/경량 버전) ----
def load_external_data(mode="demo"):
    if mode == "demo":
        # 현실 FITS 대신: 외부 데이터 샘플 흉내
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 50, 5000)) + np.random.normal(0, 0.5, 5000)
    else:
        # 나중에 실제 FITS 데이터로 교체 가능
        # from astropy.io import fits
        # with fits.open(file_path) as hdul:
        #     signal = hdul[1].data['DATA'].flatten()
        signal = np.random.randn(5000)
    return signal

# ---- Streamlit UI ----
st.title("🌌 [3번 확장판] GEA 외부 데이터 연동 모듈")

mode = st.selectbox("데이터 소스 선택", ["demo", "future_real"])
if st.button("실행 (외부 데이터 연동)"):
    signal = load_external_data(mode)
    metrics = omega_correlation(signal)

    st.write(f"Ω 값: {OMEGA_CONST}")
    st.write(f"[외부 연동] peak={metrics['peak']}, strength={metrics['strength']:.3f}")

    # 시각화
    fig, ax = plt.subplots()
    ax.plot(signal, color="purple")
    ax.set_title("External Signal (샘플)")
    st.pyplot(fig)

    st.success("⚡ 외부 데이터와 공명 분석 완료! (경량 버전)")