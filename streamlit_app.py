# -*- coding: utf-8 -*-
# gea_infinite_receiver_validated_v2.py
# GEA 우주정보장 수학공식 생성기 v2: 무한 생성 × 초검증 × 파이프라인 변환

from __future__ import annotations
import sympy as sp
import random, itertools, math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# =========================
# 1) 무한 생성기 (v1 기반 강화)
# =========================
class InfiniteReceiver:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # 기본 심볼
        self.x, self.y, self.z = sp.symbols('x y z', real=True)

        # 핵심 함수 풀 (조합 다양성 ↑)
        self.base_functions = [
            sp.zeta(self.x),                 # 리만 제타
            sp.gamma(self.x),                # 감마
            sp.besselj(0, self.x),           # 베셀 J0
            sp.cos(sp.pi*self.x),            # 삼각
            sp.exp(sp.I*self.x),             # 복소 지수
            sp.log(self.x+1),                # 로그
            sp.sin(self.x*self.y),           # 사인(이변수)
            sp.sinh(self.x),                 # 쌍곡 사인
            sp.atan(self.x),                 # 아크탄젠트
            sp.poly(1 + self.x + self.x**2), # 다항식 예제
        ]

    def _rand_expr(self) -> sp.Expr:
        choice1 = random.choice(self.base_functions)
        choice2 = random.choice(self.base_functions)
        expr = choice1.subs(self.x, self.y) * choice2.subs(self.x, self.z)
        # 구조 다양화: 로그 제곱, 혼합 미분항, 유리식 결합
        expr = expr + sp.log(self.x+1)**2 - sp.diff(sp.exp(self.x*self.y), self.x)
        expr = sp.simplify(expr)
        return expr

    def generate_formula(self) -> sp.Expr:
        return self._rand_expr()

class InfiniteStreamReceiver(InfiniteReceiver):
    def stream_formulas(self, limit: int = 10):
        for i in itertools.count(1):
            yield i, self.generate_formula()
            if limit and i >= limit:
                break

# =========================
# 2) 초검증기 (의미·안정성·비자명성 스코어)
# =========================
@dataclass
class FormulaValidatorConfig:
    probe_points: Tuple[Tuple[float, float, float], ...] = (
        (0.1, 0.2, 0.3), (1.0, 0.5, -0.25), (2.5, -1.2, 0.75), (0.0, 1.0, 2.0)
    )
    max_ops: int = 5000          # 과도한 복잡도 컷
    min_ops: int = 20            # 너무 단순(상수/자명) 컷
    weight_finite: float = 0.35
    weight_nontrivial: float = 0.30
    weight_specials: float = 0.20
    weight_diversity: float = 0.15

class FormulaValidator:
    SPECIAL_FUNCS = (sp.zeta, sp.gamma, sp.besselj, sp.log, sp.sin, sp.cos, sp.sinh, sp.atan, sp.exp)

    def __init__(self, cfg: FormulaValidatorConfig = FormulaValidatorConfig()):
        self.cfg = cfg

    def _safe_eval(self, expr: sp.Expr, x: float, y: float, z: float) -> complex | float | None:
        try:
            val = expr.subs({sp.Symbol('x'): x, sp.Symbol('y'): y, sp.Symbol('z'): z})
            val = sp.N(val)
            if val.has(sp.zoo) or val.has(sp.oo) or val.has(sp.nan):
                return None
            # sympy 수치 → python float/complex 변환 시도
            return complex(val) if val.is_complex else float(val)
        except Exception:
            return None

    def _finite_score(self, expr: sp.Expr) -> float:
        ok, total = 0, 0
        for (a,b,c) in self.cfg.probe_points:
            total += 1
            v = self._safe_eval(expr, a,b,c)
            if v is None:
                continue
            if isinstance(v, complex):
                finite = math.isfinite(v.real) and math.isfinite(v.imag)
            else:
                finite = math.isfinite(v)
            ok += 1 if finite else 0
        return 0.0 if total == 0 else ok/total

    def _nontrivial_score(self, expr: sp.Expr) -> float:
        # 연산자 수 기반: 너무 작으면 자명, 너무 크면 벌점
        ops = sp.count_ops(expr)
        if ops <= self.cfg.min_ops:
            return 0.0
        if ops >= self.cfg.max_ops:
            return 0.1
        # 적당한 복잡도일수록 가산 (sigmoid형)
        rng = self.cfg.max_ops - self.cfg.min_ops
        norm = (ops - self.cfg.min_ops) / max(1, rng)
        return 1.0/(1.0 + math.exp(-8*(norm-0.5)))

    def _specials_score(self, expr: sp.Expr) -> float:
        # 특수함수 등장 다양성
        present = set()
        for f in self.SPECIAL_FUNCS:
            try:
                if expr.has(f):
                    present.add(f)
            except Exception:
                pass
        # 종류 수를 0~1로 스케일
        return min(1.0, len(present)/6.0)

    def _diversity_score(self, expr: sp.Expr) -> float:
        # 자유변수, 함수 조합 다양성
        free = list(expr.free_symbols)
        n_var = len(free)
        ops = sp.count_ops(expr)
        # 변수 2개 이상 + 적당한 연산 수
        base = 0.0
        if n_var >= 2:
            base += 0.6
        if self.cfg.min_ops < ops < self.cfg.max_ops:
            base += 0.4
        return min(base, 1.0)

    def score(self, expr: sp.Expr) -> Dict[str, Any]:
        finite = self._finite_score(expr)
        nontriv = self._nontrivial_score(expr)
        specials = self._specials_score(expr)
        diversity = self._diversity_score(expr)
        s = (
            self.cfg.weight_finite*finite
            + self.cfg.weight_nontrivial*nontriv
            + self.cfg.weight_specials*specials
            + self.cfg.weight_diversity*diversity
        )
        s = max(0.0, min(1.0, s))
        return {
            "score": s,
            "detail": {
                "finite": finite,
                "nontrivial": nontriv,
                "specials": specials,
                "diversity": diversity,
                "ops": int(sp.count_ops(expr)),
                "free_symbols": sorted([str(v) for v in expr.free_symbols]),
            }
        }

# =========================
# 3) 파이프라인 변환 유틸 (수식 → bytes)
# =========================

def formula_to_bytes(expr: sp.Expr) -> bytes:
    """수식을 UTF-8 문자열로 직렬화하여 파이프라인용 바이트로 변환.
    필요 시, 수치 프로빙 값들을 꼬리표로 덧붙여 구조성을 강화할 수 있다.
    """
    s = sp.srepr(expr)  # 구조 보존 직렬화 (str(expr)보다 AST 정보가 풍부)
    # 간단한 프로빙 샘플을 붙인다(파이프라인의 UJG/Ultra에서 구조감지에 도움)
    samples = []
    for (a,b,c) in FormulaValidatorConfig().probe_points:
        try:
            val = expr.subs({sp.Symbol('x'): a, sp.Symbol('y'): b, sp.Symbol('z'): c})
            val = sp.N(val)
            samples.append(str(val))
        except Exception:
            samples.append("NaN")
    payload = s + "\n#PROBES=" + ",".join(samples)
    return payload.encode("utf-8", errors="ignore")

# =========================
# 4) 통합 엔진: 생성 → 검증 → 상위 파이프라인 투입용 패키징
# =========================
@dataclass
class InfiniteEngineConfig:
    seed: int | None = 42
    n_formulas: int = 50
    min_score: float = 0.65   # 초검증 통과 임계

class InfiniteEngine:
    def __init__(self, gen: InfiniteReceiver | None = None, val: FormulaValidator | None = None, cfg: InfiniteEngineConfig = InfiniteEngineConfig()):
        self.cfg = cfg
        self.gen = gen or InfiniteReceiver(seed=cfg.seed)
        self.val = val or FormulaValidator()

    def generate_validated(self) -> List[Dict[str, Any]]:
        out = []
        for _ in range(self.cfg.n_formulas):
            expr = self.gen.generate_formula()
            rep = self.val.score(expr)
            rep["expr"] = expr
            rep["ok"] = bool(rep["score"] >= self.cfg.min_score)
            out.append(rep)
        return out

    def package_as_blobs(self, reports: List[Dict[str, Any]]) -> Tuple[List[bytes], List[int]]:
        blobs, labels = [], []
        for r in reports:
            b = formula_to_bytes(r["expr"])
            blobs.append(b)
            labels.append(1 if r["ok"] else 0)  # 1=유의미(초검증 통과), 0=보류
        return blobs, labels
        
        # -*- coding: utf-8 -*-
# demo_fused_v3.py — 수학공식(무한 생성+초검증) → GEAMasterValidated 파이프라인 융합 데모

from __future__ import annotations
import json, time

# (A) 모듈 가져오기
from gea_infinite_receiver_validated_v2 import InfiniteEngine, InfiniteEngineConfig

# (B) 2차 파이프라인 가져오기 (실패 시 Mock로 대체)
try:
    from gea_master_core_v2_validated import GEAMasterValidated, ValidatedConfig
    HAVE_REAL = True
except Exception:
    HAVE_REAL = False

# ---- Mock 엔진 (실제 모듈이 없을 때도 데모 가능) ----
class _MockGEA:
    def analyze(self, blobs):
        # 아주 단순한 규칙: 길이가 긴 바이트/문자열일수록 구조적이라 가정
        k = [i for i,b in enumerate(blobs) if len(b) > 200]
        return {
            "counts": {"input": len(blobs), "after_dsp": len(k), "after_ujg": len(k)//2},
            "dsp_table": [{"i": i, "dsp_score": min(1.0, len(b)/1000), "flat": 0.5, "snr": 0.7, "ac_peak": 0.4, "kurt_excess": 0.0, "ok_dsp": (i in k)} for i,b in enumerate(blobs)],
            "ujg_avg_topscore": 0.93,
            "ultra": {"gates": ["len>200"], "ok": True},
            "ok": True
        }

if __name__ == "__main__":
    # 1) 수학 공식 생성 + 초검증
    engine = InfiniteEngine(cfg=InfiniteEngineConfig(seed=42, n_formulas=50, min_score=0.65))
    reports = engine.generate_validated()

    # 2) 파이프라인 입력 포맷으로 패키징
    blobs, labels = engine.package_as_blobs(reports)

    # 3) 실제 엔진 or 목 엔진 준비
    if HAVE_REAL:
        gea = GEAMasterValidated(ValidatedConfig())
    else:
        gea = _MockGEA()

    # 4) 실행
    t0 = time.time()
    rep = gea.analyze(blobs)
    dt = time.time() - t0

    # 5) 간이 메트릭 (초검증 라벨과 파이프라인 keep 비교)
    keep = [row["i"] for row in rep["dsp_table"] if row["ok_dsp"]]
    tp = sum(1 for i in keep if labels[i] == 1)
    fp = sum(1 for i in keep if labels[i] == 0)
    fn = sum(1 for i,l in enumerate(labels) if l == 1 and i not in keep)
    tn = len(labels) - tp - fp - fn
    prec = tp / max(1, (tp+fp))
    rec  = tp / max(1, (tp+fn))
    f1   = (2*prec*rec)/max(1e-9, (prec+rec))

    out = {
        "gen": {"n": len(reports), "min_score": 0.65, "passed": int(sum(r["ok"] for r in reports))},
        "pipeline": {
            "counts": rep["counts"],
            "ujg_avg_topscore": rep.get("ujg_avg_topscore", 0.0),
            "ultra": rep.get("ultra", {}),
            "ok": rep.get("ok", False)
        },
        "metrics": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1},
        "time_sec": dt
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    
    # -*- coding: utf-8 -*-
# gea_evo_self_engine_v1.py
# "공식만 뽑는" 수준을 넘어, 공식 자체가 세대를 거치며 자기 진화하도록 만드는 엔진

import json, math, random, time, os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
import sympy as sp

# (A) 내부 모듈
from gea_infinite_receiver_validated_v2 import (
    InfiniteReceiver, FormulaValidator, FormulaValidatorConfig,
    formula_to_bytes
)

# (B) 2차 파이프라인 (없으면 Mock)
try:
    from gea_master_core_v2_validated import GEAMasterValidated, ValidatedConfig
    HAVE_REAL = True
except Exception:
    HAVE_REAL = False

class _MockGEA:
    def analyze(self, blobs: List[bytes]) -> Dict[str, Any]:
        k = [i for i,b in enumerate(blobs) if len(b) > 200]
        return {
            "counts": {"input": len(blobs), "after_dsp": len(k), "after_ujg": len(k)//2},
            "dsp_table": [
                {
                    "i": i,
                    "dsp_score": min(1.0, len(b)/1000),
                    "flat": 0.5, "snr": 0.7, "ac_peak": 0.4, "kurt_excess": 0.0,
                    "ok_dsp": (i in k)
                } for i,b in enumerate(blobs)
            ],
            "ujg_avg_topscore": 0.93,
            "ultra": {"gates": ["len>200"], "ok": True},
            "ok": True
        }

# =============================
# 표현/연산 유틸
# =============================
SPECIAL_FUNCS = (sp.zeta, sp.gamma, sp.besselj, sp.log, sp.sin, sp.cos, sp.sinh, sp.atan, sp.exp)
SYMS = sp.symbols('x y z', real=True)

@dataclass
class EvoWeights:
    w_val: float = 0.45
    w_pipe: float = 0.40
    w_novel: float = 0.10
    w_simp: float = 0.05

@dataclass
class EvoConfig:
    seed: int = 77
    pop_size: int = 40
    generations: int = 20
    elite_k: int = 6
    mut_rate: float = 0.55
    cx_rate: float = 0.35
    rand_rate: float = 0.10
    validator: FormulaValidatorConfig = FormulaValidatorConfig()
    weights: EvoWeights = EvoWeights()
    log_path: str = 'gea_evo_run.jsonl'
    target_validator_min: float = 0.65

# -----------------------------
# 변이/교차 연산
# -----------------------------
_base_terms = [
    lambda x,y,z: sp.log(x+1)**2,
    lambda x,y,z: -sp.diff(sp.exp(x*y), x),
    lambda x,y,z: sp.cos(sp.pi*x)*sp.sin(y) + sp.sinh(z),
    lambda x,y,z: sp.besselj(0, x) * sp.exp(sp.I*y),
    lambda x,y,z: sp.atan(x) + sp.zeta(y+2),
]

_func_pool = [
    lambda x,y,z: sp.zeta(x),
    lambda x,y,z: sp.gamma(x),
    lambda x,y,z: sp.besselj(0, x),
    lambda x,y,z: sp.cos(sp.pi*x),
    lambda x,y,z: sp.exp(sp.I*x),
    lambda x,y,z: sp.log(x+1),
    lambda x,y,z: sp.sin(x*y),
    lambda x,y,z: sp.sinh(x),
    lambda x,y,z: sp.atan(x),
]

_rng = random.Random()

def _safe_simplify(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.simplify(expr)
    except Exception:
        return expr

def random_expr() -> sp.Expr:
    x,y,z = SYMS
    f1 = _rng.choice(_func_pool)(y,z,x)
    f2 = _rng.choice(_func_pool)(z,x,y)
    term = _rng.choice(_base_terms)(x,y,z)
    expr = f1*f2 + term
    return _safe_simplify(expr)

_def_mut_ops = [
    lambda e: _safe_simplify(e.subs({SYMS[0]: SYMS[1], SYMS[1]: SYMS[0]})),
    lambda e: _safe_simplify(e + _rng.choice(_base_terms)(*SYMS)),
    lambda e: _safe_simplify(sp.diff(e, _rng.choice(SYMS))),
    lambda e: _safe_simplify(sp.integrate(e, _rng.choice(SYMS))),
    lambda e: _safe_simplify(e * _rng.choice(_func_pool)(*SYMS)),
    lambda e: _safe_simplify(e.subs({SYMS[2]: SYMS[0]+SYMS[1]})),
]

def mutate(e: sp.Expr) -> sp.Expr:
    op = _rng.choice(_def_mut_ops)
    try:
        return op(e)
    except Exception:
        return e

_def_cx_ops = [
    lambda a,b: _safe_simplify(a + b),
    lambda a,b: _safe_simplify(a * b),
    lambda a,b: _safe_simplify(a.subs({SYMS[0]: b})),
]

def crossover(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    op = _rng.choice(_def_cx_ops)
    try:
        return op(a,b)
    except Exception:
        return a

# -----------------------------
# 특성/거리/노벨티
# -----------------------------
def expr_features(e: sp.Expr) -> List[float]:
    ops = float(sp.count_ops(e))
    nvars = float(len(e.free_symbols))
    pres = [1.0 if e.has(f) else 0.0 for f in SPECIAL_FUNCS]
    return [ops, nvars] + pres

def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

# -----------------------------
# 평가(단일 공식)
# -----------------------------
def eval_formula(e: sp.Expr, validator: FormulaValidator, gea) -> Dict[str, Any]:
    vrep = validator.score(e)
    b = formula_to_bytes(e)
    rep = gea.analyze([b])
    dsp_rows = rep.get("dsp_table", [])
    if dsp_rows:
        dsp_ok = 1.0 if dsp_rows[0].get("ok_dsp", False) else 0.0
        dsp_score = float(dsp_rows[0].get("dsp_score", 0.0))
    else:
        dsp_ok, dsp_score = (0.0, 0.0)
    ujg = float(rep.get("ujg_avg_topscore", 0.0))
    ultra_ok = 1.0 if rep.get("ultra", {}).get("ok", False) else 0.0

    return {
        "v_score": float(vrep["score"]),
        "v_detail": vrep["detail"],
        "dsp_ok": dsp_ok,
        "dsp_score": dsp_score,
        "ujg": ujg,
        "ultra_ok": ultra_ok,
        "pipe": rep,
    }

# -----------------------------
# 진화 루프
# -----------------------------
@dataclass
class EvoState:
    gen: int
    population: List[sp.Expr]
    reports: List[Dict[str, Any]]
    features: List[List[float]]

class EvoEngine:
    def __init__(self, cfg: EvoConfig = EvoConfig()):
        self.cfg = cfg
        _rng.seed(cfg.seed)
        random.seed(cfg.seed)
        self.validator = FormulaValidator(cfg.validator)
        self.gea = GEAMasterValidated(ValidatedConfig()) if HAVE_REAL else _MockGEA()
        self.receiver = InfiniteReceiver(seed=cfg.seed)

    def _fitness(self, rep: Dict[str, Any], feat: List[float]) -> float:
        W = self.cfg.weights
        ops = feat[0]
        pipe_score = 0.4*rep["dsp_ok"] + 0.2*rep["dsp_score"] + 0.3*rep["ujg"] + 0.1*rep["ultra_ok"]
        simp_bonus = 1.0/(1.0 + max(0.0, (ops-300.0))/200.0)
        return (
            W.w_val*rep["v_score"] +
            W.w_pipe*pipe_score +
            W.w_novel*rep.get("novel", 0.0) +
            W.w_simp*simp_bonus
        )

    def _log(self, line: Dict[str, Any]):
        # 안전 로그 (문자열/괄호 오류 방지 + 경로 자동 생성)
        if os.path.dirname(self.cfg.log_path):
            os.makedirs(os.path.dirname(self.cfg.log_path), exist_ok=True)
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _init_pop(self) -> List[sp.Expr]:
        return [self.receiver.generate_formula() for _ in range(self.cfg.pop_size)]

    def _evaluate(self, pop: List[sp.Expr], archive_feats: List[List[float]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        reps, feats = [], []
        for e in pop:
            feat = expr_features(e)
            if archive_feats:
                dists = sorted(l2(feat, a) for a in archive_feats)
                novel = sum(dists[:min(5, len(dists))]) / float(max(1, min(5, len(dists))))
                novel = min(1.0, novel / 500.0)
            else:
                novel = 0.0
            rep = eval_formula(e, self.validator, self.gea)
            rep["novel"] = float(novel)
            reps.append(rep)
            feats.append(feat)
        return reps, feats

    def run(self) -> Dict[str, Any]:
        pop = self._init_pop()
        archive_feats: List[List[float]] = []
        # 세대 반복
        for gen in range(1, self.cfg.generations+1):
            reps, feats = self._evaluate(pop, archive_feats)
            fits = [self._fitness(r, f) for r,f in zip(reps, feats)]
            ranked = sorted(zip(pop, reps, feats, fits), key=lambda t: t[3], reverse=True)
            elites = ranked[:self.cfg.elite_k]

            # 아카이브에 피처만 누적(노벨티 기준)
            for _, _, f, _ in elites:
                archive_feats.append(f)

            # 로깅
            best_e, best_r, best_f, best_s = elites[0]
            self._log({
                "t": time.time(), "gen": gen,
                "best_fit": best_s,
                "best_v": best_r["v_score"],
                "best_pipe": {"dsp_ok": best_r["dsp_ok"], "dsp_score": best_r["dsp_score"], "ujg": best_r["ujg"], "ultra_ok": best_r["ultra_ok"]},
                "best_ops": int(best_f[0]),
                "best_free": [str(s) for s in best_e.free_symbols],
                "expr": str(best_e)
            })

            # 선택 (토너먼트)
            def pick():
                i = _rng.randrange(len(ranked))
                j = _rng.randrange(len(ranked))
                return ranked[i if ranked[i][3] > ranked[j][3] else j][0]

            # 다음 세대
            next_pop: List[sp.Expr] = [e for e,_,_,_ in elites]  # 엘리트 보존
            while len(next_pop) < self.cfg.pop_size:
                r = _rng.random()
                if r < self.cfg.cx_rate and len(ranked) >= 2:
                    a = pick(); b = pick()
                    child = crossover(a, b)
                elif r < self.cfg.cx_rate + self.cfg.mut_rate:
                    parent = pick()
                    child = mutate(parent)
                else:
                    child = random_expr()
                next_pop.append(child)

            pop = next_pop

        # 최종 베스트 5
        final_reps, final_feats = self._evaluate(pop, archive_feats)
        final_fits = [self._fitness(r, f) for r,f in zip(final_reps, final_feats)]
        finals = sorted(zip(pop, final_reps, final_feats, final_fits), key=lambda t: t[3], reverse=True)[:5]
        return {
            "best": [{
                "expr": str(e),
                "v_score": r["v_score"],
                "dsp_ok": r["dsp_ok"],
                "dsp_score": r["dsp_score"],
                "ujg": r["ujg"],
                "ultra_ok": r["ultra_ok"],
                "novel": r["novel"],
                "ops": int(f[0]),
                "free_symbols": sorted([str(s) for s in e.free_symbols]),
                "fitness": s
            } for e,r,f,s in finals]
        }

if __name__ == "__main__":
    cfg = EvoConfig()
    engine = EvoEngine(cfg)
    out = engine.run()
    print(json.dumps(out, ensure_ascii=False, indent=2))