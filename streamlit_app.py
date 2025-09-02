# -*- coding: utf-8 -*-
# GEA 자가진화 수학 — 유일식/구조진화/노벨티 v2 (NO matplotlib)
# 입력(내장함수/CSV) → 목표(MSE+복잡도-노벨티) → 구조 진화 → 실시간 표시
import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import random, time, json, math
from typing import List, Tuple, Set

st.set_page_config(page_title="GEA 자가진화 수학(유일식/노벨티)", layout="wide")
st.title("🌌 GEA 자가진화 수학 — 유일식/구조진화/노벨티 v2 (no-mpl)")

# ---------- 공통 심볼/조각 ----------
x = sp.Symbol('x', real=True)
BIN = [lambda a,b: a+b, lambda a,b: a-b, lambda a,b: a*b, lambda a,b: a/(b+1e-6)]
UNA = [lambda a:a, lambda a:sp.sin(a), lambda a:sp.cos(a), lambda a:sp.tan(a),
       lambda a:sp.exp(a), lambda a:sp.log(sp.Abs(a)+1e-9), lambda a:sp.sqrt(sp.Abs(a)+1e-12)]
TERMS = [lambda: x, lambda: sp.Integer(random.randint(-5,5)), lambda: sp.Float(random.uniform(-3,3))]

def rand_term(): return random.choice(TERMS)()
def rand_tree(depth:int)->sp.Expr:
    if depth<=0: return rand_term()
    if random.random()<0.6:
        f=random.choice(BIN); return f(rand_tree(depth-1), rand_tree(depth-1))
    else:
        g=random.choice(UNA); return g(rand_tree(depth-1))

def simplify_soft(e: sp.Expr)->sp.Expr:
    try: return sp.simplify(e, rational=True)
    except Exception: return e

def expr_hash(e: sp.Expr)->str:
    try: return sp.srepr(e)
    except Exception: return str(e)

def features(e: sp.Expr)->List[float]:
    ops=float(sp.count_ops(e))
    pres=[1.0 if e.has(fn) else 0.0 for fn in (sp.sin,sp.cos,sp.tan,sp.exp,sp.log,sp.sqrt)]
    return [ops]+pres

def l2(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

# ---------- 타깃 데이터 ----------
with st.sidebar:
    st.header("🎯 타깃")
    mode = st.radio("소스", ["내장 함수","CSV 업로드"], horizontal=True)
    seed = st.number_input("Seed", 42, step=1)
    random.seed(int(seed)); np.random.seed(int(seed))
    n_points = st.slider("표본", 50, 3000, 400, 50)
    x_min, x_max = st.slider("x 범위", -10.0, 10.0, (-3.0,3.0))
    xs = np.linspace(x_min, x_max, int(n_points))

    if mode=="내장 함수":
        fsel = st.selectbox("함수", ["sin(x)","cos(x)+x","exp(x/3)","x**3-2*x","sin(x)*exp(-x**2/5)"])
        fns = {
            "sin(x)": lambda t: np.sin(t),
            "cos(x)+x": lambda t: np.cos(t)+t,
            "exp(x/3)": lambda t: np.exp(t/3.0),
            "x**3-2*x": lambda t: t**3 - 2*t,
            "sin(x)*exp(-x**2/5)": lambda t: np.sin(t)*np.exp(-(t**2)/5.0)
        }
        noise = st.slider("노이즈", 0.0, 1.0, 0.0, 0.05)
        ys = fns[fsel](xs) + np.random.normal(0, noise, xs.shape)
        target_label = fsel
    else:
        up = st.file_uploader("CSV(x,y 2열)", type=["csv"])
        if up is None: st.stop()
        arr = np.loadtxt(up, delimiter=",")
        if arr.ndim==1 or arr.shape[1]<2:
            st.error("CSV는 2개 열(x,y)이 필요합니다."); st.stop()
        xs, ys = arr[:,0], arr[:,1]
        target_label = "uploaded.csv"

# ---------- 설정 ----------
st.header("🧬 진화 설정")
c1,c2,c3,c4,c5 = st.columns(5)
with c1: pop = st.number_input("개체수", 10, 300, 80, step=10)
with c2: depth = st.number_input("최대깊이", 1, 8, 4)
with c3: gens = st.number_input("세대수", 1, 2000, 80, step=10)
with c4: cx = st.slider("교차", 0.0, 1.0, 0.6, 0.05)
with c5: mut = st.slider("돌연변이", 0.0, 1.0, 0.35, 0.05)
c6,c7,c8 = st.columns(3)
with c6: comp_l = st.slider("복잡도 가중(λ)", 0.0, 0.1, 0.01, 0.005)
with c7: nov_w = st.slider("노벨티 가중(−)", 0.0, 1.0, 0.2, 0.05)
with c8: elite_k = st.number_input("엘리트", 1, 80, 8)
dedup = st.checkbox("유일식 강제(중복 금지)", True)

# ---------- 적합도 ----------
def fitness(e: sp.Expr, xs, ys, comp_lambda, nov_bonus)->Tuple[float,float,int,float]:
    try:
        f = sp.lambdify(x, e, modules=["numpy"])
        with np.errstate(all='ignore'):
            yhat = np.array(f(xs), dtype=float)
        if not np.all(np.isfinite(yhat)):
            return float('inf'), float('inf'), int(sp.count_ops(e)), 0.0
        mse = float(np.mean((yhat - ys)**2))
    except Exception:
        return float('inf'), float('inf'), int(sp.count_ops(e)), 0.0
    comp = int(sp.count_ops(e))
    total = mse + comp_lambda*comp - nov_bonus
    return total, mse, comp, nov_bonus

# ---------- 변이/교차(구조변화 강제) ----------
def rand_subexpr(e: sp.Expr):
    parts = list(e.atoms(sp.Symbol, sp.Number, sp.Function)) or [e]
    return random.choice(parts)

def mutate(e: sp.Expr)->sp.Expr:
    r = random.random()
    if r < 0.35:   # 서브트리 치환
        return simplify_soft(e.xreplace({rand_subexpr(e): rand_tree(2)}))
    elif r < 0.7:  # 단항 감싸기
        g = random.choice(UNA); return simplify_soft(g(e))
    else:          # 상수 섭동 / 연산자 교체
        if random.random()<0.5:
            return simplify_soft(e + sp.Float(random.uniform(-0.8,0.8)))
        else:
            try:
                a,b = list(e.as_ordered_terms())[:2]
                op = random.choice(BIN)
                return simplify_soft(op(a,b))
            except Exception:
                return simplify_soft(e + rand_term())

def crossover(a: sp.Expr, b: sp.Expr)->sp.Expr:
    sa = rand_subexpr(a); sb = rand_subexpr(b)
    return simplify_soft(a.xreplace({sa: sb}))

# ---------- 초기 개체군 ----------
def init_pop(n:int, depth:int, seen:Set[str])->List[sp.Expr]:
    out=[]; tries=0
    while len(out)<n and tries<n*50:
        e = simplify_soft(rand_tree(depth)); h = expr_hash(e)
        if (not dedup) or (h not in seen):
            out.append(e); seen.add(h)
        else:
            e2 = mutate(e); h2 = expr_hash(e2)
            if h2 not in seen:
                out.append(e2); seen.add(h2)
        tries += 1
    return out

# ---------- 노벨티 ----------
def novelty_score(e: sp.Expr, archive_feats: List[List[float]])->float:
    if not archive_feats: return 0.0
    f = features(e)
    dists = sorted(l2(f,a) for a in archive_feats)
    k = min(5, len(dists))
    return float(sum(dists[:k]) / max(1,k))

# ---------- UI 자리 ----------
run = st.button("🚀 시작")
pl_best = st.empty(); pl_curve = st.empty(); pl_plot = st.empty(); pl_stats = st.empty()

# ---------- 실행 ----------
if run:
    random.seed(int(seed)); np.random.seed(int(seed))
    seen: Set[str] = set()
    pop_exprs = init_pop(int(pop), int(depth), seen)
    archive: List[List[float]] = []
    best_hist=[]

    t0=time.time()
    for gen in range(int(gens)):
        # 평가
        scored=[]
        for e in pop_exprs:
            nov = novelty_score(e, archive)
            total, mse, comp, _ = fitness(e, xs, ys, comp_l, nov_w*nov)
            scored.append((total, mse, comp, nov, e))
        scored.sort(key=lambda t: t[0])

        elites = [t[4] for t in scored[:int(elite_k)]]
        for e in elites: archive.append(features(e))

        best = scored[0]; best_expr = best[4]
        best_hist.append((gen, best[1], best[2], best[3], expr_hash(best_expr)))

        pl_best.markdown(
            f"**세대 {gen}**  \n"
            f"- 최적식(사람용): `{sp.sstr(best_expr)}`  \n"
            f"- 손실(MSE): **{best[1]:.6f}** | 복잡도: **{best[2]}** | 노벨티: **{best[3]:.3f}**"
        )

        # Target vs Best (라인차트)
        try:
            f_best = sp.lambdify(x, best_expr, modules=["numpy"])
            with np.errstate(all='ignore'):
                yhat = np.array(f_best(xs), dtype=float)
        except Exception:
            yhat = np.full_like(xs, np.nan)
        df_tb = pd.DataFrame({"x": xs, "target": ys, "best": yhat}).set_index("x")
        pl_plot.line_chart(df_tb)

        # MSE curve
        df_curve = pd.DataFrame({"gen":[g for g,_,_,_,_ in best_hist],
                                 "mse":[m for _,m,_,_,_ in best_hist]}).set_index("gen")
        pl_curve.line_chart(df_curve)

        # 다음 세대
        next_pop = list(elites)
        tries_limit = 40
        while len(next_pop) < int(pop):
            r = random.random()
            if r < cx and len(pop_exprs)>=2:
                a = random.choice(pop_exprs); b = random.choice(pop_exprs)
                child = crossover(a,b)
            elif r < cx + mut:
                parent = random.choice(pop_exprs)
                child = mutate(parent)
            else:
                child = simplify_soft(rand_tree(int(depth)))
            h = expr_hash(child); tries=0
            while dedup and (h in seen) and tries < tries_limit:
                child = mutate(child); h = expr_hash(child); tries += 1
            if (not dedup) or (h not in seen):
                next_pop.append(child); seen.add(h)

        uniq_ratio = len({expr_hash(e) for e in next_pop}) / float(len(next_pop))
        pl_stats.info(f"세대 {gen}: 유일식 비율 {uniq_ratio*100:.1f}% | 아카이브 {len(archive)}")

        pop_exprs = next_pop
        time.sleep(0.02)

    st.success(f"완료! {int(gens)}세대 / 경과 {time.time()-t0:.2f}s")
    result = {
        "target": target_label,
        "history": [{"gen":g,"mse":float(m),"comp":int(c),"nov":float(n),"hash":h} for g,m,c,n,h in best_hist],
        "best_expr": sp.sstr(sp.simplify(pop_exprs[0]))
    }
    st.download_button("📥 결과 JSON", data=json.dumps(result, ensure_ascii=False, indent=2),
                       file_name="gea_unique_v2.json", mime="application/json")
else:
    st.info("좌측에서 타깃을 정하고, 위의 설정을 조절한 뒤 **[🚀 시작]**을 누르세요.")