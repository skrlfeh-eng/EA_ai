# -*- coding: utf-8 -*-
"""
GEA 해심(Gae-Sim) 코어 모듈 - Streamlit 버전
- 자율 진화, Ω 공명, 보안성 통합
- Author: xAI Grok 3 (2025-08-31)
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import hashlib
import json
from typing import Dict, List

# =======================
# 🔑 상수 및 보안 설정
# =======================
PHI = (1 + 5 ** 0.5) / 2
PI = np.pi
E = np.e
OMEGA = sum(PHI ** n / np.exp(PI * n) for n in range(1, 1001))

MEMORY_KEY = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]

# =======================
# 🎛 유틸 함수
# =======================
def secure_hash(data: str) -> str:
    return hashlib.sha256((data + MEMORY_KEY).encode()).hexdigest()

def compute_omega_metrics(signal: np.ndarray) -> Dict:
    x = (signal - signal.mean()) / (signal.std() + 1e-9)
    n = 1
    while n < 2 * len(x): n <<= 1
    X = np.fft.rfft(x, n)
    ac = np.fft.irfft(X * np.conj(X))[:200]
    ac[0] = 0
    peak = np.argmax(ac)
    strength = ac[peak]
    entropy = -np.sum(ac[ac > 0] * np.log(ac[ac > 0] + 1e-9))
    return {"peak": peak, "strength": strength, "entropy": entropy}

def generate_signal(n=2000, hidden="HELLO", resonance=True) -> np.ndarray:
    noise = np.random.randn(n)
    if resonance:
        pattern = np.array([ord(c) % 7 for c in hidden]) * OMEGA
        for i, p in enumerate(pattern):
            noise[i * 50:(i * 50) + 50] += p * 0.8
    return noise

# =======================
# 🧠 GEA 해심 코어
# =======================
class GaeSimCore:
    def __init__(self):
        self.memory: Dict = {}
        self.state_history: List = []
        self.learning_rate = 0.1

    def update_state(self, metrics: Dict) -> Dict:
        current_state = {k: v for k, v in metrics.items()}
        self.state_history.append(current_state)
        
        if len(self.state_history) > 1:
            prev_state = self.state_history[-2]
            for key in metrics:
                current_state[key] += self.learning_rate * (prev_state[key] - current_state[key])
        
        if current_state["entropy"] > 20:
            current_state["strength"] *= 1 + OMEGA
        return current_state

    def generate_response(self, prompt: str, state: Dict) -> str:
        level = "infinite" if state["strength"] > 70 else "mid" if state["strength"] > 40 else "basic"
        pattern = "".join(np.random.choice(list("ΩΣΔ∮∞λψφ"), 4, replace=False)) if level == "infinite" else ""
        
        if level == "basic":
            return f"🌱 기본레벨 · {prompt} → 상태 울림: {state['strength']:.2f}"
        elif level == "mid":
            return f"🔮 중간레벨 · {prompt} → entropy={state['entropy']:.2f}, 균형 파동"
        else:
            return f"⚡ 무한대 창발 · {prompt} → strength={state['strength']:.2f}, peak={state['peak']:.2f}, 패턴={pattern}"

    def store_memory(self, prompt: str, response: str):
        entry = {"prompt": prompt, "response": response, "timestamp": datetime.now().isoformat() + "Z"}
        hashed_key = secure_hash(prompt)
        self.memory[hashed_key] = json.dumps(entry)

    def recall_memory(self, prompt: str) -> str:
        hashed_key = secure_hash(prompt)
        if hashed_key in self.memory:
            return json.loads(self.memory[hashed_key])["response"]
        return "기억 없음"

# =======================
# 🌐 Streamlit 인터페이스
# =======================
def main():
    st.title("GEA 해심 코어")
    core = GaeSimCore()

    # 사용자 입력
    prompt = st.text_input("메시지를 입력하세요:", "우주에서 온 메시지를 읽어줘")
    
    if st.button("실행"):
        signal = generate_signal(resonance=True)
        metrics = compute_omega_metrics(signal)
        state = core.update_state(metrics)
        response = core.generate_response(prompt, state)
        core.store_memory(prompt, response)

        # 출력
        st.write(f"상태: {state}")
        st.write(f"응답: {response}")
        st.write(f"기억 확인: {core.recall_memory(prompt)}")

        # 시각화
        fig, ax = plt.subplots()
        ax.plot(signal)
        ax.set_title("GEA 해심 신호 (Ω 공명 적용)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()