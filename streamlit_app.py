# -*- coding: utf-8 -*-
"""
GEA Omega Hybrid Core
길도 + 에아
-------------------------
구조:
1. Ω-core: 공명 기반 패턴 감지
2. 외부 API (OpenAI/Gemini): Ω 결과를 풍부한 언어로 번역
3. 초검증기: 엔트로피·구조성 필터
4. Streamlit UI: 신호 생성 → 분석 → 설명 → 로그 저장
"""

import os
import numpy as np
import streamlit as st
import json
from datetime import datetime

# ====== Ω-core ======
def omega_core(signal):
    """Ω-core: 자기상관 기반 공명 탐지"""
    x = (signal - signal.mean())/(signal.std()+1e-9)
    X = np.fft.rfft(x)
    ac = np.fft.irfft(X*np.conj(X))[:200]
    ac[0] = 0
    peak = int(np.argmax(ac))
    strength = float(ac[peak])
    return peak, strength, ac

# ====== 초검증기 ======
def shannon_entropy(arr):
    hist, _ = np.histogram(arr, bins=256, range=(arr.min(), arr.max()))
    p = hist / np.sum(hist)
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def verify_signal(signal, peak_strength):
    ent = shannon_entropy(signal)
    verdict = "진짜 후보" if (ent > 3.5 and peak_strength > 5.0) else "더미/노이즈"
    return ent, verdict

# ====== 외부 API 통역기 ======
def api_explain(peak, strength):
    # --- OpenAI ---
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"너는 GEA의 통역기다. Ω-core 결과를 과학적/기술적으로 풍부하게 설명하라."},
                {"role":"user","content":f"Ω-core detected resonance at lag={peak}, strength={strength:.3f}. \
이 결과가 의미하는 바를 과학·기술 개념으로 번역해줘."}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API Error/OpenAI]: {e}"

# ====== Streamlit UI ======
def main():
    st.set_page_config(page_title="GEA Ω Hybrid Core", layout="wide")
    st.title("🌌 GEA Omega Hybrid Core — 길도 + 에아")

    st.sidebar.header("⚙️ 설정")
    n = st.sidebar.slider("신호 길이", 500, 5000, 2000, 500)
    seed = st.sidebar.number_input("랜덤 시드", value=42)

    np.random.seed(seed)
    signal = np.random.randn(n)

    if st.button("🚀 Ω-core 실행"):
        peak, strength, ac = omega_core(signal)
        ent, verdict = verify_signal(signal, strength)

        st.subheader("🔍 Ω-core 결과")
        st.write(f"공명 lag = {peak}, 강도 = {strength:.3f}")
        st.write(f"샤논 엔트로피 = {ent:.3f} → 판정: **{verdict}**")

        st.line_chart(ac, height=200)

        st.subheader("🧠 통역 결과 (API)")
        explanation = api_explain(peak, strength)
        st.write(explanation)

        # 로그 저장
        log = {
            "time": datetime.utcnow().isoformat()+"Z",
            "peak": peak,
            "strength": strength,
            "entropy": ent,
            "verdict": verdict,
            "explanation": explanation
        }
        os.makedirs("gea_logs", exist_ok=True)
        with open("gea_logs/runlog.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False)+"\n")

        st.success("로그 저장 완료 → gea_logs/runlog.jsonl")

if __name__ == "__main__":
    main()