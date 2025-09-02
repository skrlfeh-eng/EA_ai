# -*- coding: utf-8 -*-
# gea_evo_unlimited_io.py
# GEA 자가진화 수학 창조자 — 무제한 입력(수학식/CSV) & 무제한 출력(세대별 로그/곡선) 통합본
# deps: streamlit, numpy, sympy
# run:  streamlit run gea_evo_unlimited_io.py

import json, math, random, time, traceback
from typing import List, Tuple, Dict, Any, Set, Optional

import streamlit as st

# ---- 필수 모듈 점검 ----
MISSING=[]
try:
    import numpy as np
except Exception:
    MISSING.append("numpy")
try:
    import sympy as sp
except Exception:
    MISSING.append("sympy")

st.set_page_config(page_title="GEA 무제한 I/O 자가진화 수학", layout="wide")

st.title("🌌 GEA 자가진화 수학 — 무제한 I/O 통합 모듈")
st.caption("입력(수학식/CSV/비워두면 랜덤) → 세대별 진화 로그 + MSE 곡선 → 최종 수식/JSON 다운로드")

if MISSING:
    st.error("필수 패키지 없음: " + ", ".join(MISSING))
    st.code("pip install " + " ".join(MISSING), language="bash")
    st.stop()

# ===================== 코어 유전 심볼릭 엔진 =====================
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
    if r < 0.35:   # 서브트리 치환
        return _simplify_soft(e.xreplace({_rand_subexpr(e): _rand_tree(2)}))
    elif r < 0.7:  # 단항 감싸기
        g = random.choice(UNA); return _simplify_soft(g(e))
    else:          # 상수 섭동 / 연산자 교체
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
    while len(out)<n and tries<n*60:
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
    """total_fit, mse, complexity, nov_bonus, yhat"""
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

def evo_fit_to_data(xs: np.ndarray, ys: np.ndarray,
                    seed:int=42, pop:int=80, depth:int=4, gens:int=80,
                    elite_k:int=8, cx_rate:float=0.6, mut_rate:float=0.35,
                    comp_lambda:float=0.01, novelty_w:float=0.2, dedup:bool=True,
                    live_cb=None) -> Dict[str,Any]:
    """
    입력이 있을 때: 해당 입력을 반영해 진화. live_cb(gen, best_expr, mse, comp, nov)로 실시간 콜백.
    """
    random.seed(int(seed)); np.random.seed(int(seed))
    seen:Set[str] = set()
    pop_exprs = _init_pop(int(pop), int(depth), seen, dedup)
    archive=[]; best_hist=[]; mse_curve=[]

    t0=time.time()
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

        if callable(live_cb):
            live_cb(gen, best_expr, float(best[1]), int(best[2]), float(best[3]))

        # 다음 세대
        next_pop=list(elites)
        tries=0
        while len(next_pop)<int(pop) and tries<pop*60:
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
            tries += 1
        pop_exprs=next_pop

    return {
        "best_expr": sp.sstr(sp.simplify(pop_exprs[0])) if pop_exprs else None,
        "mse": best_hist[-1]["mse"],
        "comp": best_hist[-1]["comp"],
        "history": best_hist,
        "mse_curve": mse_curve,
        "elapsed_sec": time.time()-t0
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

# ===================== 입력 UI (무제한) =====================
st.header("1) 입력")

c1,c2 = st.columns([2,1])
with c1:
    target_expr_str = st.text_area(
        "🎯 수학식 입력 (비워두면 아래 CSV/랜덤 중 선택됨) — 예:  sin(x) + 0.3*cos(3*x)",
        value="sin(x)", height=80
    )
    st.caption("SymPy 문법 사용. 허용 함수: sin, cos, tan, exp, log, sqrt 등 / 변수: x")

with c2:
    seed = st.number_input("Seed", 0, 10_000, 42, step=1)
    n_points = st.slider("표본 개수", 50, 5000, 400, 50)
    x_min, x_max = st.slider("x 범위", -20.0, 20.0, (-3.0, 3.0))

st.write("또는 CSV 업로드 (x,y 두 열)")
up = st.file_uploader("CSV 업로드", type=["csv"])

# 입력 해석
xs: Optional[np.ndarray] = None
ys: Optional[np.ndarray] = None
src_label = ""

def _sympify_target(expr_str: str) -> sp.Expr:
    # 안전한 네임스페이스로 sympify
    allowed = {
        "x": x,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
        "pi": sp.pi, "E": sp.E
    }
    return sp.sympify(expr_str, locals=allowed)

try:
    if up is not None:
        arr = np.loadtxt(up, delimiter=",", dtype=float)
        if arr.ndim==1 or arr.shape[1]<2:
            st.error("CSV는 2열(x,y)이 필요합니다."); st.stop()
        xs, ys = arr[:,0], arr[:,1]
        src_label = "CSV 업로드"
    elif target_expr_str.strip():
        expr = _sympify_target(target_expr_str.strip())
        xs = np.linspace(float(x_min), float(x_max), int(n_points))
        f = sp.lambdify(x, expr, modules=["numpy"])
        with np.errstate(all="ignore"):
            ys = np.array(f(xs), dtype=float)
        src_label = "수학식 입력"
    else:
        xs = np.linspace(float(x_min), float(x_max), int(n_points))
        ys = np.sin(xs)  # 완전 랜덤 대신 기본 sin(x)
        src_label = "랜덤 기본(sin)"
except Exception as e:
    st.error("입력 해석 중 오류")
    st.exception(e)
    st.stop()

st.success(f"입력 소스: {src_label} / 표본 {len(xs)}")

# ===================== 진화 설정 =====================
st.header("2) 진화 설정")
c3,c4,c5,c6 = st.columns(4)
with c3: gens = st.number_input("세대 수", 1, 2000, 80, step=10)
with c4: pop  = st.number_input("개체 수", 10, 500, 80, step=10)
with c5: depth = st.number_input("최대 깊이", 1, 8, 4)
with c6: elite_k = st.number_input("엘리트", 1, 80, 8)
c7,c8,c9 = st.columns(3)
with c7: cx_rate = st.slider("교차", 0.0, 1.0, 0.6, 0.05)
with c8: mut_rate = st.slider("돌연변이", 0.0, 1.0, 0.35, 0.05)
with c9: comp_lambda = st.slider("복잡도 가중(λ)", 0.0, 0.2, 0.01, 0.005)
novelty_w = st.slider("노벨티 가중(−)", 0.0, 1.0, 0.2, 0.05)
dedup = st.checkbox("유일식 강제(중복 금지)", True)

# ===================== 실행/출력 =====================
st.header("3) 실행 & 무제한 출력")

btn = st.button("🚀 진화 시작")
area_log = st.empty()
chart_placeholder = st.empty()
best_placeholder = st.empty()
stats_placeholder = st.empty()
download_placeholder = st.empty()

if btn:
    try:
        random.seed(int(seed)); np.random.seed(int(seed))
        # 실시간 로그 저장
        lines: List[str] = []
        curve: List[float] = []

        def live_cb(gen, best_expr, mse, comp, nov):
            lines.append(f"Gen {gen:4d} | MSE={mse:.6f} | C={comp:3d} | NOV={nov:.3f} | {sp.sstr(best_expr)}")
            # 최근 100줄만 출력 (너무 길어지면 잘림)
            area_log.text("\n".join(lines[-100:]))
            curve.append(mse)
            # 간단 라인차트 (pandas 없이)
            stats = {"gen": list(range(len(curve))), "mse": curve}
            import pandas as _pd  # pandas 없는 환경도 많아서 try
            try:
                df = _pd.DataFrame(stats).set_index("gen")
                chart_placeholder.line_chart(df)
            except Exception:
                # pandas 없으면 텍스트로 대체
                chart_placeholder.text(f"MSE curve (len={len(curve)}): last={curve[-1]:.6f}")

        out = evo_fit_to_data(
            xs, ys,
            seed=int(seed), pop=int(pop), depth=int(depth), gens=int(gens),
            elite_k=int(elite_k), cx_rate=float(cx_rate), mut_rate=float(mut_rate),
            comp_lambda=float(comp_lambda), novelty_w=float(novelty_w), dedup=bool(dedup),
            live_cb=live_cb
        )

        best_placeholder.markdown(f"### ✅ 최종 수식\n```\n{out['best_expr']}\n```")
        stats_placeholder.write({"MSE(final)": out["mse"], "complexity": out["comp"], "elapsed_sec": out["elapsed_sec"]})
        download_placeholder.download_button(
            "📥 결과 JSON 다운로드",
            data=json.dumps(out, ensure_ascii=False, indent=2),
            file_name="gea_unlimited_result.json",
            mime="application/json"
        )
    except Exception as e:
        st.error("실행 중 오류")
        st.code("".join(traceback.format_exc()))