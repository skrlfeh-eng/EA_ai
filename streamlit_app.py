# -*- coding: utf-8 -*-
# gea_evo_app.py
# GEA 자가진화 수학 창조자 모듈 (코어 + Streamlit 통합)

import math, random, time, json
from typing import List, Tuple, Dict, Any, Set

import numpy as np
import sympy as sp
import streamlit as st

# ====== 수식 생성 기본 구성 ======
x = sp.Symbol('x', real=True)
BIN = [lambda a,b: a+b, lambda a,b: a-b, lambda a,b: a*b, lambda a,b: a/(b+1e-6)]
UNA = [lambda a:a, sp.sin, sp.cos, sp.tan, sp.exp,
       lambda a: sp.log(sp.Abs(a)+1e-9), lambda a: sp.sqrt(sp.Abs(a)+1e-12)]
TERMS = [lambda: x, lambda: sp.Integer(random.randint(-5,5)), lambda: sp.Float(random.uniform(-3,3))]

def _rand_term(): return random.choice(TERMS)()

def _rand_tree(depth:int)->sp.Expr:
    if depth<=0: return _rand_term()
    if random.random()<0.6:
        f = random.choice(BIN); return f(_rand_tree(depth-1), _rand_tree(depth-1))
    else:
        g = random.choice(UNA); return g(_rand_tree(depth-1))

def _simplify_soft(e: sp.Expr)->sp.Expr:
    try: return sp.simplify(e, rational=True)
    except Exception: return e

def _expr_hash(e: sp.Expr)->str:
    try: return sp.srepr(e)
    except Exception: return str(e)

def _features(e: sp.Expr)->List[float]:
    ops = float(sp.count_ops(e))
    pres = [1.0 if e.has(fn) else 0.0 for fn in (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt)]
    return [ops] + pres

def _l2(a,b): return math.sqrt(sum((u-v)**2 for u,v in zip(a,b)))

def _rand_subexpr(e: sp.Expr):
    parts = list(e.atoms(sp.Symbol, sp.Number, sp.Function)) or [e]
    return random.choice(parts)

def _mutate(e: sp.Expr)->sp.Expr:
    r = random.random()
    if r < 0.35:
        return _simplify_soft(e.xreplace({_rand_subexpr(e): _rand_tree(2)}))
    elif r < 0.7:
        g = random.choice(UNA); return _simplify_soft(g(e))
    else:
        if random.random()<0.5:
            return _simplify_soft(e + sp.Float(random.uniform(-0.8,0.8)))
        else:
            try:
                a,b = list(e.as_ordered_terms())[:2]
                op = random.choice(BIN)
                return _simplify_soft(op(a,b))
            except Exception:
                return _simplify_soft(e + _rand_term())

def _crossover(a: sp.Expr, b: sp.Expr)->sp.Expr:
    sa = _rand_subexpr(a); sb = _rand_subexpr(b)
    return _simplify_soft(a.xreplace({sa: sb}))

def _init_pop(n:int, depth:int, seen:Set[str], dedup:bool)->List[sp.Expr]:
    out=[]; tries=0
    while len(out)<n and tries<n*50:
        e = _simplify_soft(_rand_tree(depth)); h = _expr_hash(e)
        if (not dedup) or (h not in seen):
            out.append(e); seen.add(h)
        else:
            e2 = _mutate(e); h2 = _expr_hash(e2)
            if h2 not in seen:
                out.append(e2); seen.add(h2)
        tries += 1
    return out

def _novelty(e: sp.Expr, archive_feats: List[List[float]])->float:
    if not archive_feats: return 0.0
    f = _features(e)
    dists = sorted(_l2(f,a) for a in archive_feats)
    k = min(5, len(dists))
    return float(sum(dists[:k]) / max(1,k))

def _fitness(e: sp.Expr, xs: np.ndarray, ys: np.ndarray,
             comp_lambda: float, nov_bonus: float)->Tuple[float,float,int,float,np.ndarray]:
    try:
        f = sp.lambdify(x, e, modules=["numpy"])
        with np.errstate(all='ignore'):
            yhat = np.array(f(xs), dtype=float)
        if not np.all(np.isfinite(yhat)):
            return float('inf'), float('inf'), int(sp.count_ops(e)), 0.0, np.full_like(xs, np.nan)
        mse = float(np.mean((yhat - ys)**2))
    except Exception:
        return float('inf'), float('inf'), int(sp.count_ops(e)), 0.0, np.full_like(xs, np.nan)
    comp = int(sp.count_ops(e))
    total = mse + comp_lambda*comp - nov_bonus
    return total, mse, comp, nov_bonus, yhat

# ====== 코어 API ======
def evo_fit_to_data(xs: np.ndarray, ys: np.ndarray,
                    seed:int=42, pop:int=80, depth:int=4, gens:int=80,
                    elite_k:int=8, cx_rate:float=0.6, mut_rate:float=0.35,
                    comp_lambda:float=0.01, novelty_w:float=0.2, dedup:bool=True) -> Dict[str,Any]:
    random.seed(int(seed)); np.random.seed(int(seed))
    seen:Set[str] = set()
    pop_exprs = _init_pop(int(pop), int(depth), seen, dedup)
    archive=[]; best_hist=[]; mse_curve=[]

    for gen in range(int(gens)):
        scored=[]
        for e in pop_exprs:
            nov = _novelty(e, archive)
            total, mse, comp, _, yhat = _fitness(e, xs, ys, comp_lambda, novelty_w*nov)
            scored.append((total,mse,comp,nov,e,yhat))
        scored.sort(key=lambda t: t[0])

        elites = [t[4] for t in scored[:int(elite_k)]]
        for e in elites: archive.append(_features(e))

        best = scored[0]
        best_expr = best[4]
        best_hist.append({"gen": gen, "mse": float(best[1]), "comp": int(best[2]),
                          "nov": float(best[3]), "expr": sp.sstr(best_expr)})
        mse_curve.append(float(best[1]))

        next_pop=list(elites)
        while len(next_pop)<int(pop):
            r=random.random()
            if r<cx_rate and len(pop_exprs)>=2:
                a=random.choice(pop_exprs); b=random.choice(pop_exprs)
                child=_crossover(a,b)
            elif r<cx_rate+mut_rate:
                parent=random.choice(pop_exprs); child=_mutate(parent)
            else:
                child=_simplify_soft(_rand_tree(int(depth)))
            h=_expr_hash(child)
            if (not dedup) or (h not in seen):
                next_pop.append(child); seen.add(h)
        pop_exprs=next_pop

    return {
        "best_expr": sp.sstr(sp.simplify(pop_exprs[0])) if pop_exprs else None,
        "mse": best_hist[-1]["mse"],
        "comp": best_hist[-1]["comp"],
        "history": best_hist,
        "mse_curve": mse_curve,
    }

def evo_random(n_points:int=400, x_range:Tuple[float,float]=(-3.0,3.0),
               target_fn:str="sin(x)", seed:int=42, **kwargs)->Dict[str,Any]:
    fns = {
        "sin(x)": lambda t: np.sin(t),
        "cos(x)+x": lambda t: np.cos(t)+t,
        "exp(x/3)": lambda t: np.exp(t/3.0),
        "x**3-2*x": lambda t: t**3 - 2*t,
        "sin(x)*exp(-x**2/5)": lambda t: np.sin(t)*np.exp(-(t**2)/5.0)
    }
    xs = np.linspace(float(x_range[0]), float(x_range[1]), int(n_points))
    ys = fns.get(target_fn, fns["sin(x)"])(xs)
    return evo_fit_to_data(xs, ys, seed=seed, **kwargs)

# ====== Streamlit UI ======
st.set_page_config(page_title="GEA 자가진화 수학 창조자", layout="wide")
st.title("🌌 GEA 자가진화 수학 창조자 모듈 v1")

mode = st.radio("입력 모드", ["내장 함수", "CSV 업로드"], horizontal=True)

if mode=="내장 함수":
    target = st.selectbox("함수", ["sin(x)", "cos(x)+x", "exp(x/3)", "x**3-2*x", "sin(x)*exp(-x**2/5)"])
    gens = st.slider("세대 수 (generations)", 10, 200, 60)
    pop = st.slider("개체 수 (population)", 20, 200, 80)
    if st.button("🚀 진화 시작"):
        out = evo_random(target_fn=target, gens=gens, pop=pop)
        st.success("완료")
        st.write("**최적식:**", out["best_expr"])
        st.write("**MSE:**", out["mse"])
        st.json(out)

else:
    up = st.file_uploader("CSV(x,y)", type=["csv"])
    gens = st.slider("세대 수 (generations)", 10, 200, 60)
    pop = st.slider("개체 수 (population)", 20, 200, 80)
    if up and st.button("🚀 진화 시작"):
        arr = np.loadtxt(up, delimiter=",", dtype=float)
        xs, ys = arr[:,0], arr[:,1]
        out = evo_fit_to_data(xs, ys, gens=gens, pop=pop)
        st.success("완료")
        st.write("**최적식:**", out["best_expr"])
        st.write("**MSE:**", out["mse"])
        st.json(out)