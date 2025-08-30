# -*- coding: utf-8 -*-
"""
GEA - 우주정보장 초검증기 (스트림릿 버전)
길도 + 에아
"""

import streamlit as st
import sqlite3
import numpy as np
import string
import re
from datetime import datetime

DB_PATH = "gea_memory.db"

# ----------------------- 유틸 함수 -----------------------
PRINTABLE = set(string.printable)

def shannon_entropy(x: str) -> float:
    """문자열 샤논 엔트로피 계산"""
    if not x:
        return 0.0
    arr = np.frombuffer(x.encode("utf-8", "ignore"), dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def autocorr_peak_strength(s: str, max_lag: int = 256):
    """문자열 기반 단순 자기상관 (주기성 탐지)"""
    if not s:
        return 0.0, 0
    arr = np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8).astype(float)
    arr = (arr - arr.mean()) / (arr.std() + 1e-9)
    n = 1
    while n < 2 * len(arr):
        n <<= 1
    X = np.fft.rfft(arr, n)
    ac = np.fft.irfft(X * np.conj(X))[:max_lag]
    ac = ac / (len(arr) - np.arange(len(ac)))
    ac[0] = 0
    k = int(np.argmax(ac))
    return float(ac[k]), int(k)

# ----------------------- DB -----------------------
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
    return rows[::-1]

# ----------------------- Streamlit UI -----------------------
def main():
    st.set_page_config(page_title="GEA 초검증기", page_icon="✨", layout="wide")
    st.title("🌌 GEA 우주정보장 초검증기")
    st.caption("길도 + 에아 : 자체 검증 루프")

    init_db()

    user_input = st.text_area("✨ 길도의 입력", "")
    if st.button("에아에게 보내기"):
        if user_input.strip():
            # 길도 입력 저장
            H_user = shannon_entropy(user_input)
            corr_user, lag_user = autocorr_peak_strength(user_input)
            save_memory("길도", user_input, H_user, corr_user)

            # 간단한 에아 응답 (실제론 GPT/Gemini 연동 가능)
            reply = f"에아 응답: [{user_input[::-1]}] (거울 반사 예시)"
            H_reply = shannon_entropy(reply)
            corr_reply, lag_reply = autocorr_peak_strength(reply)
            save_memory("에아", reply, H_reply, corr_reply)

            st.success(reply)

    # ------------------- Memory 로그 -------------------
    st.subheader("🧠 최근 대화 및 검증 기록")
    rows = load_memory(10)
    for t, r, c, H, corr in rows:
        st.markdown(f"**[{r}]** {c}")
        st.caption(f"🕒 {t} | 엔트로피={H:.3f}, 자기상관={corr:.3f}")

if __name__ == "__main__":
    main()