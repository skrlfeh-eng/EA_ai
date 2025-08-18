# -*- coding: utf-8 -*-
# Minimal, safe Streamlit entrypoint

import sys
from pathlib import Path
import streamlit as st

APP_NAME = "EA · Ultra (baseline)"

def show_env_diag():
    st.subheader("Diag")
    st.write({
        "cwd": str(Path.cwd()),
        "file": __file__,
        "python": sys.version.split()[0],
        "sys.path[0]": sys.path[0],
    })

def main():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.title(APP_NAME)
    st.caption("Baseline OK – if you see this, deploy & routing are correct.")

    tab1, tab2 = st.tabs(["Chat", "System"])
    with tab1:
        user = st.text_input("메시지", "")
        if st.button("Send") and user.strip():
            st.success(f"에아 응답(샘플): {user[::-1]}")  # 자리표시자 로직
    with tab2:
        show_env_diag()
        st.code("Main file = streamlit_app.py (not a folder)")

if __name__ == "__main__":
    # Streamlit이 모듈로 실행해도 안전
    main()
    