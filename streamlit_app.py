# -*- coding: utf-8 -*-
# 최소 진단앱: 필수 모듈/버전/간단 계산 후, 모든 예외를 화면에 그대로 노출
import sys, traceback
import streamlit as st

st.set_page_config(page_title="DIAG", layout="wide")
st.title("🔍 Streamlit 진단")

def diag():
    st.write("Python:", sys.version)
    # 1) numpy
    try:
        import numpy as np
        st.success(f"NumPy OK: {np.__version__}")
        st.write("샘플 계산:", float(np.sin(1.234)))
    except Exception as e:
        st.error("NumPy 에러"); st.exception(e); return

    # 2) sympy
    try:
        import sympy as sp
        st.success(f"SymPy OK: {sp.__version__}")
        x = sp.Symbol('x', real=True)
        st.write("심볼릭 샘플:", sp.simplify(sp.sin(x)**2 + sp.cos(x)**2))
    except Exception as e:
        st.error("SymPy 에러"); st.exception(e); return

    st.success("환경 OK ✅ — 이제 본 앱을 실행해도 됨")

try:
    diag()
except Exception as e:
    st.error("상단 외부 예외 발생")
    st.code("".join(traceback.format_exc()))