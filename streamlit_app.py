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