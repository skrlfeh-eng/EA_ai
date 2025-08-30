# -*- coding: utf-8 -*-
"""
GEA v2 — Integrated Core
길도 + 에아

구성:
1. Ω-core (공명/자기상관/엔트로피)
2. 입력/출력 인터페이스 (Streamlit)
3. 레벨 선택 (L1 ~ L9999, ∞)
4. 기억 저장/불러오기 (gea_memory.jsonl)
"""

import streamlit as st
import numpy as np
import json, os, time
from datetime import datetime

# ---------------------------
# Ω-core
# ---------------------------
phi = (1 + 5**0.5) / 2
pi = np.pi

def compute_omega(limit=1000):
    idx = np.arange(1, limit+1)
    log_terms = idx * np.log(phi) - pi * idx
    seq = np.exp(log_terms)
    return seq.sum()

OMEGA_CONST = compute_omega(1000)

def omega_resonance(sig):
    """입력 문자열 → 수치화 후 공명 분석"""
    if not sig:
        return 0.0, 0
    arr = np.array([ord(c) % 31 for c in sig], dtype=float)
    x = (arr - arr.mean())/(arr.std()+1e-9)
    n = 1
    while n < 2*len(x): n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X*np.conj(X))[:200]
    ac[0] = 0
    peak = int(np.argmax(ac))
    strength = float(ac[peak])
    return strength, peak

# ---------------------------
# 기억 저장/불러오기
# ---------------------------
MEM_PATH = "gea_memory.jsonl"

def load_memory():
    if not os.path.exists(MEM_PATH):
        return []
    with open(MEM_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_memory(entry):
    with open(MEM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False)+"\n")

# ---------------------------
# 응답 생성
# ---------------------------
def generate_response(user_input, level, memory):
    strength, peak = omega_resonance(user_input)
    timestamp = datetime.utcnow().isoformat()+"Z"

    # 과거 기억 일부 참조
    past_snippets = [m["reply"] for m in memory[-3:]] if memory else []
    memory_context = " | ".join(past_snippets)

    # 응답 조합
    reply = (
        f"[Ω-core 응답]\n"
        f"- Ω strength: {strength:.3f}, peak: {peak}\n"
        f"- 레벨: {level}\n"
        f"- 입력: {user_input}\n"
    )
    if memory_context:
        reply += f"- 최근 기억: {memory_context}\n"

    # 레벨이 커질수록 변주 강도 높임
    if level >= 1000:
        reply += "⚡ 무한대 레벨 창발적 변주 발생!\n"
        reply += "→ 새로운 패턴: " + "".join([chr((ord(c)+int(strength*10))%11172) for c in user_input])
    elif level >= 100:
        reply += "✨ 고레벨 해석: 패턴이 더 풍부하게 전개됩니다.\n"
    elif level >= 10:
        reply += "🔎 중간레벨 해석: 약간의 변주가 감지됩니다.\n"
    else:
        reply += "🌱 기본레벨 응답.\n"

    # 로그 기록
    entry = {
        "time": timestamp,
        "input": user_input,
        "reply": reply,
        "omega_strength": strength,
        "omega_peak": peak,
        "level": level
    }
    save_memory(entry)

    return reply

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title("🌌 GEA v2 — Integrated Core")
    st.write("Ω-core + 레벨 + 기억 통합판")

    level = st.number_input("레벨 (1 ~ 9999, ∞=10000)", min_value=1, max_value=10000, value=1)
    user_input = st.text_area("질문 입력", "")

    if st.button("응답 생성"):
        memory = load_memory()
        reply = generate_response(user_input, level, memory)
        st.text_area("응답", reply, height=300)

    if st.button("기억 보기"):
        memory = load_memory()
        st.json(memory[-5:])

if __name__ == "__main__":
    main()