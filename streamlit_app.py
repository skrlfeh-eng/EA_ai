# -*- coding: utf-8 -*-
# gea_self_evolving_math.py
# GEA 자가진화 수학 창조자 모듈 (Streamlit 통합판)

import streamlit as st
import sympy as sp
import random, time, json

# ======================
# Infinite Receiver (무한 수학 공식 생성기)
# ======================
class InfiniteReceiver:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.x, self.y, self.z = sp.symbols('x y z')
        self.base_functions = [
            sp.zeta(self.x), sp.gamma(self.x), sp.besselj(0, self.x),
            sp.cos(sp.pi*self.x), sp.exp(sp.I*self.x),
            sp.log(self.x+1), sp.sin(self.x*self.y),
            sp.sinh(self.x), sp.atan(self.x)
        ]

    def generate_formula(self):
        c1, c2 = random.choice(self.base_functions), random.choice(self.base_functions)
        expr = c1.subs(self.x, self.y) * c2.subs(self.x, self.z)
        expr = expr + sp.log(self.x+1)**2 - sp.diff(sp.exp(self.x*self.y), self.x)
        return sp.simplify(expr)

# ======================
# Validator (초검증 필터)
# ======================
def validate_formula(expr):
    score = 0.0
    try:
        # 복잡도 낮을수록 가산점
        ops = len(str(expr))
        score += max(0, 1.0 - ops/500)
        # 특수 함수 포함 여부
        if any(fn in str(expr) for fn in ["zeta", "gamma", "besselj"]):
            score += 0.3
    except Exception:
        pass
    return round(min(score, 1.0), 3)

# ======================
# Evolution Engine (자가진화 루프)
# ======================
def evolve_formulas(receiver, generations=5, pop_size=10):
    population = []
    for _ in range(pop_size):
        expr = receiver.generate_formula()
        score = validate_formula(expr)
        population.append((expr, score))

    history = []
    for g in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)
        history.append({"gen": g, "best": str(population[0][0]), "score": population[0][1]})
        # 상위 절반 교차 + 돌연변이
        next_gen = population[:pop_size//2]
        while len(next_gen) < pop_size:
            e1, e2 = random.choice(next_gen)[0], random.choice(next_gen)[0]
            child = sp.simplify(e1 + e2)
            score = validate_formula(child)
            next_gen.append((child, score))
        population = next_gen
    return history

# ======================
# Streamlit UI
# ======================
st.title("🌌 GEA 자가진화 수학 창조자 모듈 v1")
st.write("길도의 무한 상상력 + 우주정보장 에아 + AI 플랫폼 시너지 ⚡")

seed = st.number_input("Seed (재현성)", value=42, step=1)
gens = st.slider("세대 수 (generations)", 1, 20, 5)
pop = st.slider("개체 수 (population)", 5, 50, 10)

if st.button("🚀 진화 시작"):
    st.info("자가진화 수학 엔진 실행 중…")
    receiver = InfiniteReceiver(seed=seed)
    progress = st.progress(0)
    history = []

    for g in range(gens):
        result = evolve_formulas(receiver, generations=1, pop_size=pop)
        history.extend(result)
        progress.progress((g+1)/gens)
        time.sleep(0.2)

    st.success("진화 완료 ✅")
    st.json(history)