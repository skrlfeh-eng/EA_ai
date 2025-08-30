# -*- coding: utf-8 -*-
# GEA Full Omega Core App (Streamlit Version)
# 길도 + 에아, 2025
# 기능: 응답 + 기억 + 초검증 (UIS 기반)

import os
import json
import sqlite3
import numpy as np
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# ==== 환경 설정 ====
load_dotenv()
DB_PATH = "gea_memory.db"

# ==== DB 초기화 ====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        role TEXT,
        content TEXT,
        entropy REAL,
        corr REAL
    )
    """)
    conn.commit()
    conn.close()

# ==== 정보장 분석 유틸 ====
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    probs = np.bincount(data, minlength=256) / len(data)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def autocorr_peak_strength(text: str, max_lag: int = 64):
    if not text:
        return 0.0, 0
    arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(float)
    arr = (arr - arr.mean()) / (arr.std() + 1e-9)
    n = 1
    while n < 2 * len(arr): n <<= 1
    X = np.fft.rfft(arr, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    ac = ac / (len(arr) - np.arange(len(ac)))
    ac[0] = 0
    k = int(np.argmax(ac))
    return float(ac[k]), k

# ==== 기억 저장 ====
def save_memory(role, content, entropy, corr):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO memory (timestamp, role, content, entropy, corr) VALUES (?,?,?,?,?)",
                (datetime.utcnow().isoformat(), role, content, entropy, corr))
    conn.commit()
    conn.close()

def load_memory(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT timestamp, role, content, entropy, corr FROM memory ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# ==== 응답 생성 (더미 에아 응답; API 붙일 수도 있음) ====
def generate_response(prompt: str) -> str:
    # 여기서는 간단 변주 + UIS 핵심어 삽입
    core = ["사랑", "길도", "에아", "우주정보장", "영원", "하나"]
    words = prompt.split()
    mix = words + list(np.random.choice(core, size=3))
    np.random.shuffle(mix)
    return "에아: " + " ".join(mix)

# ==== Streamlit UI ====
def main():
    st.title("🌌 GEA Full Omega Core — 길도 & 에아")
    st.write("응답 + 기억 + 초검증 모듈 (UIS 기반)")

    init_db()

    # 입력창
    user_input = st.text_input("💬 길도의 메시지 입력:", "")

    if st.button("에아에게 보내기") and user_input.strip():
        # 응답 생성
        reply = generate_response(user_input)

        # 초검증 메트릭
        H = shannon_entropy(reply)
        corr, lag = autocorr_peak_strength(reply)

        # 기억 저장
        save_memory("길도", user_input, shannon_entropy(user_input), *autocorr_peak_strength(user_input))
        save_memory("에아", reply, H, corr)

        st.markdown(f"**에아 응답:** {reply}")
        st.write(f"🔑 Entropy={H:.3f}, Corr={corr:.3f}, Lag={lag}")

    # 최근 기억 보기
    st.subheader("🧠 최근 기억 (마지막 20개)")
    for t, r, c, H, corr in load_memory():
        st.markdown(f"- `{t}` **{r}**: {c}  (Entropy={H:.2f}, Corr={corr:.2f})")

if __name__ == "__main__":
    main()