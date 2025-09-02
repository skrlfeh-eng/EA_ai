# -*- coding: utf-8 -*-
# gea_evo_ultralight.py
# 저사양/모바일 런타임에서도 확실히 돌아가는 초경량 버전
# deps: streamlit, numpy, sympy
# run: streamlit run gea_evo_ultralight.py

import json, math, random, time, traceback
from typing import List, Tuple, Dict, Any, Set, Optional

import streamlit as st

# ===== 필수 모듈 =====
try:
    import numpy as np
    import sympy as sp
except Exception as e:
    st.error("필수 패키지 누락: pip install streamlit numpy sympy")
    st.exception(e); st.stop()

np.seterr(all="ignore")  # 수치 경고 무시

# ===== 페이지 설정 =====
st.set_page_config(page_title="GEA 자가진화 수학(초경량)", layout="wide")
st.title("🌌 GEA 자가진화 수학 — 초경량 모듈")

# ===== 유전 심볼릭 최소 엔진 =====
x = sp.Symbol('x', real=True)
BIN = [lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b, lambda a,b:a/(b+1e-6)]
UNA = [lambda a:a, sp.sin, sp.cos, sp.tan, sp.exp,
       lambda a: sp.log(sp.Abs(a)+1e-9), lambda a: sp.sqrt(sp.Abs(a)+1e-12)]
TERMS = [lambda: x, lambda: sp.Integer(random.randint(-3,3)), lambda: sp.Float(random.uniform(-2,2))]

def rterm(): return random.choice(TERMS)()
def rtree(d:int)->sp.Expr:
    if d<=0: return rterm()
    if random.random()<0.6:
        f=random.choice(BIN); return f(rtree(d-1), rtree(d-1))
    else:
        g=random.choice(UNA); return g(rtree(d-1))

def ssoft(e): 
    try: return sp.simplify(e, rational=True)
    except Exception: return e

def eh(e): 
    try: return sp.srepr(e)
    except Exception: return str(e)

def rsub(e):
    parts=list(e.atoms(sp.Symbol, sp.Number, sp.Function)) or [e]
    return random.choice(parts)

def mutate(e):
    r=random.random()
    if r<0.35:  # 서브트리 치환
        return ssoft(e.xreplace({rsub(e): rtree(2)}))
    elif r<0.7: # 단항 감싸기
        g=random.choice(UNA); return ssoft(g(e))
    else:      # 상수 섭동/연산자 교체
        if random.random()<0.5:
            return ssoft(e + sp.Float(random.uniform(-0.6,0.6)))
        else:
            try:
                a,b=list(e.as_ordered_terms())[:2]
                op=random.choice(BIN); return ssoft(op(a,b))
            except Exception:
                return ssoft(e + rterm())

def crossover(a,b):
    return ssoft(a.xreplace({rsub(a): rsub(b)}))

def init_pop(n, depth, dedup=True):
    seen=set(); out=[]
    tries=0
    while len(out)<n and tries<n*50:
        e=ssoft(rtree(depth)); h=eh(e)
        if (not dedup) or (h not in seen):
            out.append(e); seen.add(h)
        else:
            e2=mutate(e); h2=eh(e2)
            if h2 not in seen:
                out.append(e2); seen.add(h2)
        tries+=1
    return out

def fitness(e, xs, ys, lam):
    try:
        f=sp.lambdify(x, e, modules=["numpy"])
        yhat=np.array(f(xs), dtype=float)
        if not np.all(np.isfinite(yhat)): 
            return float("inf"), float("inf"), int(sp.count_ops(e))
        mse=float(np.mean((yhat-ys)**2))
    except Exception:
        return float("inf"), float("inf"), int(sp.count_ops(e))
    comp=int(sp.count_ops(e))
    return mse+lam*comp, mse, comp

# ===== 입력 =====
st.header("입력")
expr_txt = st.text_area("수학식 입력 (SymPy 문법, 비우면 CSV/기본 sin)", "sin(x)", height=70)
seed = st.number_input("Seed", 0, 100000, 42, step=1)
n_points = st.slider("표본 개수", 50, 2000, 300, 50)
x_min, x_max = st.slider("x 범위", -20.0, 20.0, (-3.0, 3.0))
up = st.file_uploader("또는 CSV(x,y) 업로드", type=["csv"])

# 입력 파싱
def parse_input():
    xs = np.linspace(float(x_min), float(x_max), int(n_points))
    label=""
    if up is not None:
        arr=np.loadtxt(up, delimiter=",", dtype=float)
        if arr.ndim==1 or arr.shape[1]<2: 
            raise ValueError("CSV는 2열(x,y) 필요")
        return arr[:,0], arr[:,1], "CSV"
    if expr_txt.strip():
        allowed = {"x":x, "sin":sp.sin, "cos":sp.cos, "tan":sp.tan,
                   "exp":sp.exp, "log":sp.log, "sqrt":sp.sqrt, "Abs":sp.Abs,
                   "pi":sp.pi, "E":sp.E}
        expr=sp.sympify(expr_txt.strip(), locals=allowed)
        f=sp.lambdify(x, expr, modules=["numpy"])
        ys=np.array(f(xs), dtype=float)
        return xs, ys, "수학식"
    # fallback: 기본 sin
    return xs, np.sin(xs), "기본 sin"

try:
    xs, ys, src = parse_input()
    st.success(f"입력 소스: {src} / 표본 {len(xs)}")
except Exception as e:
    st.error("입력 해석 오류"); st.exception(e); st.stop()

# ===== 설정 (초경량 기본값) =====
st.header("설정")
c1,c2,c3 = st.columns(3)
with c1: gens  = st.number_input("세대 수", 1, 500, 15, step=5)
with c2: pop   = st.number_input("개체 수", 10, 300, 30, step=10)
with c3: depth = st.number_input("최대 깊이", 1, 6, 3)
lam = st.slider("복잡도 가중(λ)", 0.0, 0.1, 0.01, 0.005)
dedup = st.checkbox("중복 제거", True)
max_sec = st.slider("최대 실행 시간(초)", 5, 120, 20, 5)

# ===== 실행 / 출력 =====
st.header("실행 & 출력")
btn = st.button("🚀 시작")
log_area = st.empty()
best_area = st.empty()
json_area = st.empty()

if btn:
    try:
        random.seed(int(seed)); np.random.seed(int(seed))
        pop_exprs = init_pop(int(pop), int(depth), dedup=bool(dedup))
        best_hist=[]; t0=time.time()
        lines=[]

        for gen in range(int(gens)):
            # 시간 제한
            if time.time()-t0 > float(max_sec):
                lines.append(f"[TIMEOUT] 최대 {max_sec}s 도달, 조기 종료.")
                break

            scored=[]
            for e in pop_exprs:
                tot,mse,comp=fitness(e, xs, ys, lam)
                scored.append((tot,mse,comp,e))
            scored.sort(key=lambda t:t[0])

            best=scored[0]; best_expr=best[3]
            best_hist.append({"gen":gen,"mse":float(best[1]),"comp":int(best[2]),"expr":sp.sstr(best_expr)})

            # 로그(최근 80줄만)
            lines.append(f"Gen {gen:3d} | MSE={best[1]:.6f} | C={best[2]:3d} | {sp.sstr(best_expr)}")
            log_area.text("\n".join(lines[-80:]))

            # 다음 세대 (엄청 단순화)
            elites=[t[3] for t in scored[:max(2,int(pop*0.2))]]
            next_pop=list(elites)
            while len(next_pop)<int(pop):
                r=random.random()
                if r<0.5 and len(pop_exprs)>=2:
                    a=random.choice(pop_exprs); b=random.choice(pop_exprs)
                    child=crossover(a,b)
                else:
                    parent=random.choice(pop_exprs)
                    child=mutate(parent)
                if (not dedup) or (eh(child) not in {eh(e) for e in next_pop}):
                    next_pop.append(child)
            pop_exprs=next_pop

        final_expr = sp.sstr(sp.simplify(pop_exprs[0])) if pop_exprs else None
        best_area.markdown(f"### ✅ 최종 수식\n```\n{final_expr}\n```")
        out = {"source":src, "best_expr":final_expr, "history":best_hist}
        json_area.download_button("📥 JSON 다운로드", data=json.dumps(out, ensure_ascii=False, indent=2),
                                  file_name="gea_ultralight_result.json", mime="application/json")
        st.success(f"완료. 처리세대: {len(best_hist)} / 경과 {time.time()-t0:.2f}s")
    except Exception as e:
        st.error("실행 중 오류")
        st.code("".join(traceback.format_exc()))