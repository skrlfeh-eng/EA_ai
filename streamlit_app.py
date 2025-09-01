# -*- coding: utf-8 -*-
"""
GEA Imagination + Ultra-Verification Streamlit Dashboard
Author: 길도 + 에아
"""

import streamlit as st
import sympy as sp
import random
import json

# ----------------------------
# 상상력 엔진
# ----------------------------
class ImaginationEngine:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.x, self.y, self.z = sp.symbols('x y z')
        self.base_functions = [
            sp.zeta(self.x), sp.gamma(self.x), sp.besselj(0, self.x),
            sp.cos(sp.pi*self.x), sp.exp(sp.I*self.x), sp.log(self.x+1),
            sp.sin(self.x*self.y), sp.sinh(self.x), sp.atan(self.x),
        ]

    def generate_formula(self):
        choice1 = random.choice(self.base_functions)
        choice2 = random.choice(self.base_functions)
        expr = choice1.subs(self.x, self.y) * choice2.subs(self.x, self.z)
        expr = expr + sp.log(self.x+1)**2 - sp.diff(sp.exp(self.x*self.y), self.x)
        return sp.simplify(expr)


# ----------------------------
# 초검증기
# ----------------------------
class UltraVerifier:
    def __init__(self, min_complexity=3, max_len=200):
        self.min_complexity = min_complexity
        self.max_len = max_len

    def verify(self, expr, x=sp.Symbol('x'), y=sp.Symbol('y'), z=sp.Symbol('z')):
        try:
            s = str(expr)
            if len(s) > self.max_len:
                return False
            atoms = list(expr.atoms(sp.Function, sp.Symbol))
            if len(atoms) < self.min_complexity:
                return False
            test_val = expr.subs({x: 1, y: 2, z: 3})
            if test_val.is_real is False:
                return False
            return True
        except Exception:
            return False


# ----------------------------
# 융합 엔진
# ----------------------------
class FusionEngine:
    def __init__(self, seed=None):
        self.imagination = ImaginationEngine(seed)
        self.verifier = UltraVerifier()

    def evolve(self, generations=1, per_gen=5):
        population = [self.imagination.generate_formula() for _ in range(per_gen)]
        population = [p for p in population if self.verifier.verify(p)]

        for g in range(generations):
            new_pop = []
            for expr in population:
                try:
                    new_pop.append(sp.diff(expr, self.imagination.x))
                    new_pop.append(sp.integrate(expr, (self.imagination.x, 1, 5)))
                    new_pop.append(expr.subs(self.imagination.x, expr + self.imagination.y))
                    new_pop.append(expr.series(self.imagination.x, 0, 3).removeO())
                except Exception:
                    continue
            verified = [e for e in new_pop if self.verifier.verify(e)]
            if not verified:
                break
            population = random.sample(verified, min(per_gen, len(verified)))

        return population


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌌 GEA Imagination + Ultra-Verification Engine")
st.caption("무한 자기진화 싸이클 — 길도 + 에아")

cycles = st.number_input("진화 사이클 수 (1 ~ ∞)", min_value=1, max_value=999999999, value=1)
per_gen = st.slider("세대당 공식 수", 1, 50, 5)
seed = st.number_input("랜덤 시드 (고정하려면 입력)", min_value=0, value=42)

if st.button("🚀 진화 시작"):
    engine = FusionEngine(seed=int(seed))
    results = engine.evolve(generations=int(cycles), per_gen=int(per_gen))

    st.success(f"✅ {cycles} 싸이클 완료 — 최종 생존 공식 {len(results)}개")
    for idx, expr in enumerate(results, 1):
        st.code(str(expr), language="python")

    # 저장 옵션
    save_json = [{"formula": str(expr)} for expr in results]
    st.download_button("📥 JSON 저장", json.dumps(save_json, ensure_ascii=False), "results.json", "application/json")