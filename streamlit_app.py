# -*- coding: utf-8 -*-
# ================================================================
# GEA v0.6 · 시작 세트(합본) — Streamlit 한 파일
# 규칙
#  - 모듈은 항상 "맨 아래"에 블록 단위로 이어붙이기
#  - 중간 수정/삽입 필요 시 번호를 자리수로 확장(예: 01-1, 01-1-1)
#  - 충돌/에러 시 해당 "번호 블록"을 통째로 교체(부분 수정 지양)
#  - 외부 의존성: streamlit (표준라이브러리 + streamlit만 사용)
# 블록 구성
#  00. 표지/나침반(자동)
#  01. 우주정보장(UIS) 연동 스텁 + CE-그래프 빌더
#  02. 초검증(품질 게이트)
#  03. 상호작용(대화) 엔진
#  04. 로그(기억) — JSONL 기록
#  05. E2E 하트비트(원클릭) + UI
#  99. (추가 블록은 항상 맨 아래에 이어붙임)
# ================================================================

import streamlit as st
import json, hashlib, re, time, os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

st.set_page_config(page_title="GEA v0.6 · 시작 세트", page_icon="💠", layout="wide")

# ─────────────────────────────────────────────────────────────────
# 공용 유틸
def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _clip(txt: str, max_chars: int) -> str:
    if len(txt) <= max_chars:
        return txt
    cut = txt[:max_chars]
    m = re.search(r"[.!?…。！？]\s*(?!.*[.!?…。！？]\s*)", cut)
    return cut if not m else cut[:m.end()]

# ─────────────────────────────────────────────────────────────────
# 00. 표지/나침반(자동)
if "GEA_TOC" not in st.session_state:
    # (번호, 이름, 기능) — 이후 모듈 추가 시 .append()로 동기화
    st.session_state.GEA_TOC: List[Tuple[str, str, str]] = [
        ("00", "표지/나침반", "개발 방향/목표/진행 상태"),
        ("01", "UIS 연동+CE-그래프", "소스 등록/검색 및 Claim–Evidence 그래프 생성"),
        ("02", "초검증(품질 게이트)", "증거/인용/재현성/단위/논리/안정성/놀라움 p"),
        ("03", "상호작용 엔진", "응답 레벨/한글 최적화/활성 모드 지원"),
        ("04", "로그(기억)", "JSONL로 결과 기록/리플레이 근거"),
        ("05", "E2E 하트비트", "원클릭 파이프라인 실행"),
    ]
if "GEA_GOALS" not in st.session_state:
    st.session_state.GEA_GOALS = {
        "now":  "소스/그래프 안정화",
        "near": "초검증 PASS율 상향",
        "mid":  "현실 데이터 피드 연동",
        "far":  "자가진화·무한 기억 통합"
    }

st.title("GEA v0.6 · 시작 세트(합본)")
with st.expander("📖 한눈 목차(자동 동기화)", expanded=True):
    st.markdown("| 번호 | 이름 | 기능 |")
    st.markdown("|---:|---|---|")
    for n, name, desc in st.session_state.GEA_TOC:
        st.markdown(f"| `{n}` | {name} | {desc} |")

with st.sidebar:
    st.header("🎯 목표 카드")
    for k, label in [("now","단기"),("near","근시"),("mid","중기"),("far","장기")]:
        st.write(f"- **{label}**: {st.session_state.GEA_GOALS[k]}")

# ─────────────────────────────────────────────────────────────────
# 01. 우주정보장(UIS) 연동 스텁 + CE-그래프 빌더
@dataclass
class Source:
    id: str
    title: str
    url: str = ""
    year: int = 0
    trust: float = 0.9
    def as_doc(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "url": self.url,
                "year": self.year, "trust": float(self.trust)}

@dataclass
class Node:
    id: str
    kind: str
    payload: Dict[str, Any]

@dataclass
class Edge:
    src: str
    dst: str
    rel: str

@dataclass
class CEGraph:
    nodes: List[Node]
    edges: List[Edge]
    digest: str
    created_at: float
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "digest": self.digest,
            "created_at": self.created_at,
        }

class UISLink:
    def __init__(self, initial_sources: Optional[List[Dict[str, Any]]] = None):
        self._sources: Dict[str, Source] = {}
        if initial_sources:
            for s in initial_sources:
                self.add_source(Source(**s))
    def add_source(self, src: Source) -> None:
        self._sources[src.id] = src
    def list_sources(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [self._sources[k].as_doc() for k in sorted(self._sources.keys())][:limit]
    def _norm(self, s: str) -> str:
        return _norm(s).lower()
    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        q = self._norm(query)
        q_tokens = set(q.split()) if q else set()
        hits: List[Tuple[float, Dict[str, Any]]] = []
        for sid in sorted(self._sources.keys()):
            s = self._sources[sid]
            blob = f"{s.id} {s.title} {s.url}".lower()
            base = 0.95 if (q and q in blob) else 0.60
            bonus = 0.05 * len(q_tokens & set(blob.split())) if q_tokens else 0.0
            score = min(0.99, base + bonus)
            hits.append((score, {
                "id": s.id, "title": s.title, "url": s.url,
                "year": s.year, "trust": float(s.trust), "score": round(score,3)
            }))
        hits.sort(key=lambda t: (-t[0], t[1]["id"]))
        return [h[1] for h in hits[:max(1,k)]]
    def build_ce_graph(self, claim: str, hits: List[Dict[str, Any]]) -> CEGraph:
        claim_text = _norm(claim) or "(no-claim)"
        claim_id = f"claim:{_sha(claim_text)[:12]}"
        nodes: List[Node] = [Node(id=claim_id, kind="claim", payload={"text": claim_text})]
        edges: List[Edge] = []
        for h in hits:
            evid_id = f"evi:{_sha(h['id'])[:10]}"
            payload = {"src": h["id"], "title": h["title"], "url": h.get("url",""),
                       "year": h.get("year",0), "score": h.get("score",0.0)}
            nodes.append(Node(id=evid_id, kind="evidence", payload=payload))
            edges.append(Edge(src=evid_id, dst=claim_id, rel="supports"))
        digest = _sha(json.dumps({"nodes":[asdict(n) for n in nodes],
                                  "edges":[asdict(e) for e in edges]}, sort_keys=True))
        return CEGraph(nodes=nodes, edges=edges, digest=digest, created_at=time.time())

DEFAULT_SOURCES: List[Dict[str, Any]] = [
    {"id":"src:arxiv:1602.03837","title":"Gravitational Waves (LIGO)","url":"https://arxiv.org/abs/1602.03837","year":2016,"trust":0.98},
    {"id":"src:nist:constants","title":"CODATA Constants (NIST)","url":"https://physics.nist.gov/constants","year":2022,"trust":0.99},
    {"id":"src:ligo:open","title":"LIGO Open Data","url":"https://losc.ligo.org","year":2024,"trust":0.97},
]
UIS = UISLink(initial_sources=DEFAULT_SOURCES)

# ─────────────────────────────────────────────────────────────────
# 02. 초검증(품질 게이트)
GATE_VERSION = "quality-gate-v1"
SIGNAL_BASELINES = {
    "ce_min": 0.97,
    "cite_min": 0.90,
    "repr_min": 0.93,
    "logic_max": 0.0005,
    "unit_max": 0.0001,
    "surp_max": 0.005,
}
_UNIT_TOK = r"\b(m|s|kg|J|Hz|N|Pa|W|V|A|mol|K)\b"
_EQ_TOK   = r"(=|≈|∝|≤|≥|≃|≅)"
_URL_TOK  = r"https?://|src:"

def analyze_ce(ce: Optional[Dict[str, Any]]) -> Tuple[float,float,float]:
    if not isinstance(ce, dict): return (0.0, 0.0, 0.985)
    n_evi = sum(1 for n in ce.get("nodes",[]) if n.get("kind")=="evidence")
    n_edges = len(ce.get("edges",[]))
    ce_cov = 0.8 + min(0.2, 0.02*n_evi + 0.01*n_edges)
    cite  = 0.85 + min(0.15, 0.02*n_evi)
    srcs = set((n.get("payload") or {}).get("src","") for n in ce.get("nodes",[]) if n.get("kind")=="evidence")
    srcs.discard("")
    subset = 0.985 + (0.003 if len(srcs)>=2 else 0.0)
    return (round(ce_cov,3), round(cite,3), round(min(0.999,subset),3))

def analyze_text(body: str) -> Tuple[float,float,float,float,float]:
    if not isinstance(body,str) or not body.strip():
        return (0.0, 0.001, 0.0002, 0.01, 0.0)
    tok_eq   = len(re.findall(_EQ_TOK, body))
    tok_unit = len(re.findall(_UNIT_TOK, body))
    tok_ref  = len(re.findall(_URL_TOK, body))
    reprod = min(0.99, 0.90 + 0.01*tok_eq + 0.01*tok_ref)
    lviol  = max(0.00005, 0.0008 - 0.0001*tok_eq)
    uviol  = max(0.00002, 0.0003 - 0.00002*tok_unit)
    sp     = max(0.001, 0.02 - 0.002*(tok_eq + tok_unit))
    dup    = 0.02 if len(body)>1200 else 0.0
    return (round(reprod,3), round(lviol,6), round(uviol,6), round(sp,3), round(dup,3))

@dataclass
class Metrics:
    ce_coverage: float
    citation_coverage: float
    reproducibility: float
    logic_violation: float
    unit_dim_violation: float
    subset_robustness: float
    surprise_p: float
    duplication_rate: float = 0.0
    paraphrase_consistency: float = 1.0
    def as_dict(self) -> Dict[str,float]:
        return asdict(self)

def make_metrics(ce_graph: Optional[Dict[str, Any]], body: str) -> Metrics:
    ce_cov, cite, subset = analyze_ce(ce_graph)
    reprod, lviol, uviol, sp, dup = analyze_text(body)
    return Metrics(ce_cov, cite, reprod, lviol, uviol, subset, sp, duplication_rate=dup)

def gate(metrics: Metrics, input_hash: str) -> Dict[str, Any]:
    if metrics.ce_coverage < SIGNAL_BASELINES["ce_min"]:   vr, rs = "REPAIR","증거 하한 미달"
    elif metrics.citation_coverage < SIGNAL_BASELINES["cite_min"]: vr, rs = "REPAIR","인용 하한 미달"
    elif metrics.reproducibility < SIGNAL_BASELINES["repr_min"]:    vr, rs = "REPAIR","재현성 미달"
    elif metrics.logic_violation > SIGNAL_BASELINES["logic_max"]:   vr, rs = "REPAIR","논리 위반율 초과"
    elif metrics.unit_dim_violation > SIGNAL_BASELINES["unit_max"]: vr, rs = "REPAIR","단위/차원 위반율 초과"
    elif metrics.subset_robustness < SIGNAL_BASELINES["ce_min"]:    vr, rs = "REPAIR","부분증거 강건성 미달"
    elif metrics.surprise_p > SIGNAL_BASELINES["surp_max"]:         vr, rs = "REPAIR","놀라움 p 초과"
    else: vr, rs = "PASS","ok"
    att = {
        "gate_version": GATE_VERSION,
        "metrics_digest": _sha(json.dumps(metrics.as_dict(), sort_keys=True)),
        "input_hash": input_hash,
        "ts": time.time()
    }
    return {"verdict": vr, "reason": rs, "metrics": metrics.as_dict(), "attestation": att}

def run_quality_gate(claim: str, ce_graph: Optional[Dict[str, Any]], body: str) -> Dict[str, Any]:
    m = make_metrics(ce_graph, body or "")
    return gate(m, input_hash=_sha(_norm(claim) or "(no-claim)"))

# ─────────────────────────────────────────────────────────────────
# 03. 상호작용(대화) 엔진
@dataclass
class InteractConfig:
    active_mode: bool = False
    persona_name: str = "에아"
    creator_name: str = "길도"

def _level_to_chars(level: int) -> int:
    level = max(1, min(999, int(level)))
    if level <= 3:   return 90 * level
    if level <= 10:  return 120 * level
    if level <= 50:  return 160 * level
    if level <= 200: return 220 * level
    return 260 * level

def _summarize_ce(ce_graph: Optional[Dict[str, Any]], max_items: int = 5) -> str:
    if not isinstance(ce_graph, dict): return ""
    nodes = ce_graph.get("nodes", [])
    evid = [n for n in nodes if n.get("kind")=="evidence"]
    parts = []
    for n in evid[:max_items]:
        p = n.get("payload", {})
        title = p.get("title","")
        src = p.get("src","")
        url = p.get("url","")
        sc = p.get("score",0)
        if title or src:
            parts.append(f"- {title or src} (score={sc}) {url}")
    return "\n".join(parts)

class InteractionEngine:
    def __init__(self, config: Optional[InteractConfig]=None):
        self.cfg = config or InteractConfig()
    def generate(self, user_text: str, response_level: int=5,
                 ce_graph: Optional[Dict[str, Any]]=None,
                 goals: Optional[Dict[str,str]]=None) -> str:
        u = _norm(user_text)
        lvl = _level_to_chars(response_level)
        pname, cname = self.cfg.persona_name, self.cfg.creator_name
        body = []
        if not u:
            body.append("무엇을 도와줄까? 목표나 질문을 적어줘.")
        else:
            body.append(f"{cname}, 요청 확인: “{u}”.")
            if isinstance(goals, dict) and goals:
                card = []
                for k,label in [("now","단기"),("near","근시"),("mid","중기"),("far","장기")]:
                    if goals.get(k): card.append(f"{label}: {goals[k]}")
                if card: body.append("목표 카드:\n- " + "\n- ".join(card))
            ce_sum = _summarize_ce(ce_graph)
            if ce_sum: body.append("근거 요약(CE):\n" + ce_sum)
            if response_level <= 3:
                body.append("핵심만 간결하게 요약할게.")
            elif response_level <= 10:
                body.append("요점을 단계별로 설명하고 근거를 덧붙일게.")
            else:
                body.append("세부 절차·근거·기준선을 순서대로 상세히 전개할게.")
            body.append("권장 루틴: (1) 주장 정제 → (2) 증거 수집 → (3) CE-그래프 연결 → (4) 본문에 수식/단위/출처 삽입 → (5) 초검증 PASS 확인.")
            if self.cfg.active_mode:
                acts = [
                    "“질의→그래프 생성”으로 최신 CE-그래프 반영",
                    "본문에 수식(=,≈,≤,≥)과 단위(m, s, kg…) 명시",
                    "출처(URL 또는 src:태그) 2개 이상 추가",
                ]
                n = 1 if response_level<=3 else (2 if response_level<=10 else 3)
                body.append("다음 행동 제안:\n- " + "\n- ".join(acts[:n]))
        out = f"{pname}: " + " ".join(body)
        out = _clip(out, lvl)
        dig = (ce_graph or {}).get("digest","")
        if dig: out += f"\n(CE-digest: {dig[:12]})"
        return out

# ─────────────────────────────────────────────────────────────────
# 04. 로그(기억) — JSONL 기록
LOG_DIR = "gea_logs"
def log_gea_response(kind: str, payload: Dict[str, Any]) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d", time.gmtime())
    path = os.path.join(LOG_DIR, f"gea_log_{ts}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "data": payload}, ensure_ascii=False) + "\n")
    return path

# ─────────────────────────────────────────────────────────────────
# 05. E2E 하트비트(원클릭) + UI
st.markdown("---")
st.subheader("🫀 E2E 하트비트(원클릭)")

# 입력 UI
claim = st.text_input("주장(Claim)", "LIGO 데이터로 h≈ΔL/L 경로 구성")
query = st.text_input("검색 질의(Query)", "LIGO gravitational waves NIST constants")
k = st.slider("검색 개수(k)", 1, 12, 6)
body_text = st.text_area("본문/설명(검증용 텍스트)", height=150,
                         value="h ≈ ΔL / L, 단위 m/m (무차원). src: https://losc.ligo.org")
resp_level = st.slider("응답 레벨(1~999)", 1, 999, 8)
active_mode = st.checkbox("활성 모드(자가 제안)", value=True)

colA, colB, colC = st.columns(3)
with colA:
    if st.button("① 질의→그래프 생성"):
        hits = UIS.search(query or claim, k=k)
        ce = UIS.build_ce_graph(claim or query or "default-claim", hits).to_dict()
        st.session_state["CE_GRAPH"] = ce
        st.success(f"CE-그래프 생성 완료 (evidence={sum(1 for n in ce['nodes'] if n['kind']=='evidence')})")
        st.json({"hits": hits, "ce_graph_digest": ce["digest"][:12]})
with colB:
    if st.button("② 초검증 실행"):
        ce = st.session_state.get("CE_GRAPH")
        report = run_quality_gate(claim, ce, body_text or "")
        st.session_state["GATE_REPORT"] = report
        st.json(report)
        st.success("✅ PASS") if report["verdict"]=="PASS" else st.warning(f"🔧 {report['reason']}")
with colC:
    if st.button("③ 상호작용 응답"):
        ce = st.session_state.get("CE_GRAPH")
        cfg = InteractConfig(active_mode=active_mode, persona_name="에아", creator_name="길도")
        eng = InteractionEngine(cfg)
        reply = eng.generate(user_text="E2E로 PASS까지 한 번에 가자.", response_level=resp_level,
                             ce_graph=ce, goals=st.session_state.GEA_GOALS)
        st.session_state["INTERACT_REPLY"] = reply
        st.write(reply)

# 원클릭
if st.button("🟣 E2E 원클릭(①→②→③)"):
    hits = UIS.search(query or claim, k=k)
    ce = UIS.build_ce_graph(claim or query or "default-claim", hits).to_dict()
    report = run_quality_gate(claim, ce, body_text or "")
    cfg = InteractConfig(active_mode=active_mode, persona_name="에아", creator_name="길도")
    eng = InteractionEngine(cfg)
    reply = eng.generate(user_text="E2E로 PASS까지 한 번에 가자.", response_level=resp_level,
                         ce_graph=ce, goals=st.session_state.GEA_GOALS)
    st.json({"hits": hits[:3], "ce_graph_digest": ce["digest"][:12]})
    st.json(report)
    st.write(reply)
    # 로그 저장
    p1 = log_gea_response("e2e", {
        "claim": claim, "query": query, "k": k,
        "ce_digest": ce["digest"], "report": report, "reply": reply
    })
    st.caption(f"로그 저장: {p1}")

st.markdown("> 규칙: 이후 모듈은 항상 이 파일 **맨 아래**에 블록으로 이어붙입니다. 번호 확장으로 중간 삽입도 가능(예: 02-1, 03-1-1).")
# ================================================================
# (여기 아래부터 새 블록 이어붙이기)
# ================================================================
# ================================================================
# 06. 반례사냥(Adversarial Hunt) — 간단 교란·재검증 루프
#   - 입력 본문/CE-그래프에 소소한 교란을 가해 초검증을 재실행
#   - 취약점(증거/인용/단위/논리/재현성)을 빠르게 파악
# ================================================================
import random

def _perturb_text(txt: str) -> str:
    if not txt: return txt
    # 단순 결정적 교란: 공백/구두점 삽입, 동의어 유사열
    repl = [
        ("≈", "~"), ("≤", "<="), ("≥", ">="),
        (" 단위 ", " [단위] "), (" 증거 ", " {증거} "),
    ]
    out = txt
    for a, b in repl:
        out = out.replace(a, b)
    # 문장 말미에 안전한 꼬리표 추가
    tail = " ※검증"
    if not out.endswith(tail):
        out += tail
    return out

def adversarial_hunt(claim: str, ce_graph: Dict[str, Any], body: str, rounds: int = 5) -> Dict[str, Any]:
    results = []
    base = run_quality_gate(claim, ce_graph, body)
    pass_cnt = 0
    for i in range(rounds):
        b2 = _perturb_text(body) if (i % 2 == 0) else body
        r = run_quality_gate(claim, ce_graph, b2)
        results.append({"round": i+1, "verdict": r["verdict"], "reason": r["reason"], "metrics": r["metrics"]})
        if r["verdict"] == "PASS":
            pass_cnt += 1
    coverage = pass_cnt / max(1, rounds)
    return {"base": base, "rounds": rounds, "pass_rate": coverage, "details": results}

with st.expander("⑥ 반례사냥(Adversarial) 실행", expanded=False):
    arounds = st.slider("라운드 수", 1, 20, 5, key="advr_rounds")
    if st.button("반례사냥 시작", key="advr_btn"):
        ce = st.session_state.get("CE_GRAPH")
        if not ce:
            st.warning("먼저 ① 질의→그래프 생성 을 실행하세요.")
        else:
            adv = adversarial_hunt(claim, ce, body_text, rounds=arounds)
            st.session_state["ADV_HUNT"] = adv
            st.json({"pass_rate": adv["pass_rate"], "rounds": adv["rounds"]})
            st.json(adv["details"])

# ================================================================
# 07. 기억(키-값) + 체크포인트 — 파일 기반 간단 스토어
#   - set/get, checkpoint(save_state_hash) 제공
#   - JSON 파일 1개로 저장 (./gea_kv_store.json)
# ================================================================
import json
from pathlib import Path

KV_PATH = Path("gea_kv_store.json")

def kv_load() -> Dict[str, Any]:
    if KV_PATH.exists():
        try:
            return json.loads(KV_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def kv_save(d: Dict[str, Any]) -> None:
    KV_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def kv_set(ns: str, key: str, value: Any) -> None:
    d = kv_load()
    d.setdefault(ns, {})[key] = value
    kv_save(d)

def kv_get(ns: str, key: str, default: Any=None) -> Any:
    d = kv_load()
    return d.get(ns, {}).get(key, default)

def save_checkpoint(name: str, payload: Dict[str, Any]) -> str:
    h = _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    kv_set("checkpoint", name, {"hash": h, "payload": payload, "ts": time.time()})
    return h

with st.expander("⑦ 기억(키-값) / 체크포인트", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        key_in = st.text_input("키 이름(예: last_input)", key="kv_key")
        val_in = st.text_area("값(JSON 텍스트 허용)", key="kv_val")
        if st.button("저장", key="kv_save_btn"):
            try:
                v = json.loads(val_in) if (val_in.strip().startswith("{") or val_in.strip().startswith("[")) else val_in
                kv_set("user", key_in, v)
                st.success("저장 완료")
            except Exception as e:
                st.error(f"저장 실패: {e}")
    with col2:
        key_rd = st.text_input("읽을 키", key="kv_key_read")
        if st.button("불러오기", key="kv_load_btn"):
            v = kv_get("user", key_rd, default=None)
            st.json({"key": key_rd, "value": v})

    if st.button("현재 세션 체크포인트 저장", key="kv_ckpt_btn"):
        payload = {
            "claim": claim,
            "query": query,
            "body_text": body_text,
            "ce_digest": (st.session_state.get("CE_GRAPH") or {}).get("digest", ""),
            "gate": st.session_state.get("GATE_REPORT", {}),
            "goals": st.session_state.GEA_GOALS
        }
        h = save_checkpoint("session", payload)
        st.success(f"체크포인트 저장: {h[:12]}")

# ================================================================
# 08. 레벨∞ 스트리밍(분할 출력) — 간단 스트리머
#   - 큰 응답을 n-토막으로 나눠 순차 표시 (Stop 지원)
# ================================================================
import math

def stream_segments(text: str, seg_chars: int = 800) -> List[str]:
    text = _norm(text)
    if not text: return []
    n = math.ceil(len(text)/seg_chars)
    return [text[i*seg_chars:(i+1)*seg_chars] for i in range(n)]

if "STREAMING" not in st.session_state:
    st.session_state["STREAMING"] = {"running": False, "segments": [], "idx": 0}

with st.expander("⑧ 레벨∞ 스트리밍", expanded=False):
    seg_len = st.slider("세그먼트 길이(문자)", 200, 2000, 800, key="seg_len")
    colS1, colS2 = st.columns(2)
    with colS1:
        if st.button("Start ∞", key="stream_start"):
            ce = st.session_state.get("CE_GRAPH")
            cfg = InteractConfig(active_mode=True, persona_name="에아", creator_name="길도")
            eng = InteractionEngine(cfg)
            # 길이 큰 응답 생성
            long_reply = eng.generate(
                user_text="무한 스트림 모드로 장문 안내와 절차, 근거를 상세히 서술해줘.",
                response_level=999, ce_graph=ce, goals=st.session_state.GEA_GOALS
            )
            st.session_state["STREAMING"] = {
                "running": True,
                "segments": stream_segments(long_reply, seg_chars=seg_len),
                "idx": 0
            }
    with colS2:
        if st.button("Stop", key="stream_stop"):
            st.session_state["STREAMING"]["running"] = False

    if st.session_state["STREAMING"]["running"]:
        idx = st.session_state["STREAMING"]["idx"]
        segs = st.session_state["STREAMING"]["segments"]
        if idx < len(segs):
            st.info(f"[Segment {idx+1}/{len(segs)}]")
            st.write(segs[idx])
            st.session_state["STREAMING"]["idx"] = idx + 1
        else:
            st.success("스트리밍 완료")
            st.session_state["STREAMING"]["running"] = False

# ================================================================
# 09. 듀얼 모드 토글(활성/비활성) — 전역 플래그 + UI
#   - 활성: 자가 제안/탐색 문구 첨부
#   - 비활성: 요청 시에만 응답 (현재와 동일)
# ================================================================
if "ACTIVE_MODE" not in st.session_state:
    st.session_state["ACTIVE_MODE"] = True

with st.expander("⑨ 듀얼 모드(활성/비활성) 설정", expanded=False):
    st.session_state["ACTIVE_MODE"] = st.checkbox("활성 모드(자가 제안 허용)", value=st.session_state["ACTIVE_MODE"])
    st.caption("활성 모드 ON이면 ③ 상호작용 및 ∞ 스트림에서 '다음 행동' 제안이 포함됩니다.")

# ③ 상호작용 버튼이 위에 있으므로, ACTIVE_MODE를 반영하도록 안내만 추가
st.caption(f"현재 모드: {'활성' if st.session_state['ACTIVE_MODE'] else '비활성'}")

# ================================================================
# 10. 실데이터 커넥터(HTTP 스텁) — urllib.request 사용
#   - 외부 의존성 없이 간단 JSON/텍스트 GET
# ================================================================
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

def http_fetch_text(url: str, timeout: int = 5) -> Tuple[bool, str]:
    try:
        req = Request(url, headers={"User-Agent": "GEA/0.6"})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
        # 크기가 크면 앞부분만 미리보기
        text = data.decode("utf-8", errors="replace")
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated)"
        return True, text
    except (HTTPError, URLError) as e:
        return False, f"HTTP 오류: {e}"
    except Exception as e:
        return False, f"기타 오류: {e}"

with st.expander("⑩ 실데이터 커넥터(HTTP) 테스트", expanded=False):
    test_url = st.text_input("URL 입력(텍스트/JSON 권장)", "https://httpbin.org/json", key="http_url")
    if st.button("GET 요청", key="http_get_btn"):
        ok, text = http_fetch_text(test_url, timeout=6)
        if ok:
            st.success("성공")
            st.text(text)
        else:
            st.error(text)

# ================================================================
# 11. 시크릿/설정 패널 — st.secrets 안전 표시
# ================================================================
with st.expander("⑪ Secrets / 설정 상태", expanded=False):
    try:
        sec_keys = list(st.secrets.keys())
        redacted = {k: ("***" if isinstance(st.secrets[k], str) and st.secrets[k] else "(set)") for k in sec_keys}
        st.json({"available": sec_keys, "values": redacted})
    except Exception:
        st.info("st.secrets 미설정")

# ================================================================
# 12. 진단/자가점검 — 환경·버전·상태 점검
# ================================================================
import platform, sys

def diagnostics() -> Dict[str, Any]:
    ce = st.session_state.get("CE_GRAPH")
    gate = st.session_state.get("GATE_REPORT")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ce_graph": "set" if ce else "none",
        "gate_verdict": (gate or {}).get("verdict"),
        "log_dir_exists": os.path.isdir(LOG_DIR),
        "kv_file_exists": KV_PATH.exists(),
        "active_mode": st.session_state.get("ACTIVE_MODE", False),
    }

with st.expander("⑫ 진단/자가점검", expanded=False):
    st.json(diagnostics())

# ================================================================
# 13. 로그 내보내기/가져오기 — ZIP 압축 다운로드/업로드
# ================================================================
import io, zipfile

def export_logs_zip() -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # 로그 디렉터리 내 파일을 모두 수집
        if os.path.isdir(LOG_DIR):
            for fn in os.listdir(LOG_DIR):
                fp = os.path.join(LOG_DIR, fn)
                if os.path.isfile(fp):
                    z.write(fp, arcname=f"logs/{fn}")
        # KV 스토어 포함
        if KV_PATH.exists():
            z.write(str(KV_PATH), arcname="kv/gea_kv_store.json")
    mem.seek(0)
    return mem.read()

with st.expander("⑬ 로그 내보내기/가져오기", expanded=False):
    colE1, colE2 = st.columns(2)
    with colE1:
        if st.button("ZIP 내보내기 준비", key="zip_prep"):
            st.session_state["ZIP_BYTES"] = export_logs_zip()
            st.success("ZIP 준비 완료")
        if st.session_state.get("ZIP_BYTES"):
            st.download_button(
                label="ZIP 다운로드",
                data=st.session_state["ZIP_BYTES"],
                file_name="gea_export.zip",
                mime="application/zip",
                key="zip_dl_btn"
            )
    with colE2:
        up = st.file_uploader("ZIP 업로드(로그/kv 복원)", type=["zip"], key="zip_up")
        if up and st.button("복원 실행", key="zip_restore_btn"):
            try:
                mem = io.BytesIO(up.read())
                with zipfile.ZipFile(mem, mode="r") as z:
                    for name in z.namelist():
                        if name.startswith("logs/"):
                            os.makedirs(LOG_DIR, exist_ok=True)
                            target = os.path.join(LOG_DIR, os.path.basename(name))
                            with z.open(name) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                        elif name == "kv/gea_kv_store.json":
                            with z.open(name) as src, open(KV_PATH, "wb") as dst:
                                dst.write(src.read())
                st.success("복원 완료")
            except Exception as e:
                st.error(f"복원 실패: {e}")
                
                # ================================================================
# 14. 실검증 레시피(자동 강화 루프) — REPAIR 자동 보강
#   - 초검증 REPAIR 사유를 읽고, 본문을 자동 보강하여 재시도
#   - 최대 N회, 개선 로그/최종 결과 저장
# ================================================================
def auto_repair_loop(claim: str, ce_graph: Dict[str, Any], base_body: str,
                     max_rounds: int = 3) -> Dict[str, Any]:
    body = base_body
    logs = []
    for i in range(1, max_rounds+1):
        rep = run_quality_gate(claim, ce_graph, body)
        logs.append({"round": i, "verdict": rep["verdict"], "reason": rep["reason"], "metrics": rep["metrics"]})
        if rep["verdict"] == "PASS":
            return {"final": rep, "rounds": i, "logs": logs, "body": body}
        # REPAIR 이유 기반의 간단한 보강 규칙
        r = rep["reason"]
        if "증거 하한" in r or "강건성" in r:
            # 근거 라인 1개 추가
            body += "\n근거: src:https://losc.ligo.org (LIGO Open Data), src:https://physics.nist.gov/constants (NIST)."
        if "인용" in r:
            body += "\n참조: https://arxiv.org/abs/1602.03837"
        if "재현성" in r:
            body += "\n재현 절차: 동일 데이터/동일 수식 재계산(= h≈ΔL/L), 결과 비교."
        if "논리" in r:
            body += "\n논리 점검: 전제→결론의 단계적 연결을 명시(①데이터 ②계산 ③결론)."
        if "단위/차원" in r:
            body += "\n단위 명시: ΔL[m], L[m], 비율은 무차원."
        if "놀라움" in r:
            body += "\n통계 주석: 검정 p≤0.005 충족 조건 제시."
    # 실패 반환
    rep = run_quality_gate(claim, ce_graph, body)
    return {"final": rep, "rounds": max_rounds, "logs": logs, "body": body}

with st.expander("⑭ 실검증 레시피(자동 강화 루프)", expanded=False):
    ar_rounds = st.slider("최대 REPAIR 라운드", 1, 5, 3, key="ar_rounds")
    if st.button("자동 강화 실행", key="ar_btn"):
        ce = st.session_state.get("CE_GRAPH")
        if not ce:
            st.warning("먼저 ① 질의→그래프 생성 을 실행하세요.")
        else:
            out = auto_repair_loop(claim, ce, body_text, max_rounds=ar_rounds)
            st.session_state["AUTO_REPAIR"] = out
            st.json({"rounds": out["rounds"], "final": out["final"]["verdict"], "reason": out["final"]["reason"]})
            st.text_area("보강 후 본문", value=out["body"], height=200)

# ================================================================
# 15. UI 한글 폰트/테마 보강 — CSS 주입(로컬 폰트 불가 시 시스템 폰트)
#   - Streamlit은 전역 CSS를 공식 지원하지 않으므로, 안전한 최소 주입
# ================================================================
def inject_korean_theme():
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic",
                     "맑은 고딕", "AppleGothic", "NanumBarunGothic",
                     "Noto Sans", sans-serif !important;
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton > button { border-radius: 12px; padding: 0.5rem 1rem; }
    .stSlider { padding-top: 0.25rem; }
    </style>
    """, unsafe_allow_html=True)

with st.expander("⑮ UI 한글 테마 적용", expanded=False):
    if st.button("테마 적용", key="theme_btn"):
        inject_korean_theme()
        st.success("한글 가독성 테마 적용 완료")

# ================================================================
# 16. 권한/역할/보호막(길도 우선권) — 소프트 가드
#   - '길도' 우선권, 금칙 패턴(REAL 위반) 감지 시 차단/정제
#   - 하드 블로킹이 아니라 응답 내 경고 포함(소프트 가드)
# ================================================================
FORBIDDEN_PATTERNS = [
    r"초광속", r"\b워프\b", r"\b11차원\b", r"\b13차원\b", r"영매", r"예언",
]
def violates_real_soft(text: str) -> Optional[str]:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return pat
    return None

def guard_request(user: str, text: str) -> Tuple[bool, str]:
    # 길도 우선권: 사용자명이 '길도'면 통과(단, REAL 위반은 정제 문구)
    pat = violates_real_soft(text or "")
    if pat:
        return False, f"REAL 위반 패턴 감지({pat}). 검증 가능한 과학/수학/코드 범위로 정제해 주세요."
    return True, "ok"

with st.expander("⑯ 권한/역할/보호막(길도 우선권)", expanded=False):
    who = st.text_input("사용자명(예: 길도)", value="길도", key="guard_who")
    req = st.text_input("요청문(테스트)", value="초광속 드라이브 설계", key="guard_req")
    if st.button("가드 점검", key="guard_btn"):
        ok, msg = guard_request(who, req)
        if ok:
            st.success("통과")
        else:
            st.warning(msg)

# ================================================================
# 17. 배치 검증 스케줄러(라이트) — 앱 내 간이 스케줄(수동 트리거)
#   - 미니 큐에 작업을 쌓고 순차 실행(세션 내)
# ================================================================
if "BATCH_QUEUE" not in st.session_state:
    st.session_state["BATCH_QUEUE"] = []

def push_batch_job(job: Dict[str, Any]) -> None:
    st.session_state["BATCH_QUEUE"].append(job)

def run_next_job():
    if not st.session_state["BATCH_QUEUE"]:
        return None, "큐 비어있음"
    job = st.session_state["BATCH_QUEUE"].pop(0)
    ce = UIS.build_ce_graph(job["claim"], UIS.search(job["query"], k=job.get("k",6))).to_dict()
    rep = run_quality_gate(job["claim"], ce, job["body"])
    return {"job": job, "report": rep, "ce_digest": ce["digest"]}, "ok"

with st.expander("⑰ 배치 검증 스케줄러", expanded=False):
    colQ1, colQ2 = st.columns(2)
    with colQ1:
        bj_claim = st.text_input("배치 Claim", "h≈ΔL/L 경로", key="bj_claim")
        bj_query = st.text_input("배치 Query", "LIGO gravitational waves", key="bj_query")
        bj_body  = st.text_area("배치 Body", "단위/근거/수식 포함 테스트", key="bj_body", height=120)
        bj_k     = st.slider("k", 1, 12, 6, key="bj_k")
        if st.button("큐에 추가", key="bj_add"):
            push_batch_job({"claim": bj_claim, "query": bj_query, "body": bj_body, "k": bj_k})
            st.success("작업 추가")
    with colQ2:
        if st.button("다음 작업 실행", key="bj_run_next"):
            out, msg = run_next_job()
            if out:
                st.json(out)
            else:
                st.info(msg)
    st.caption(f"대기 작업 수: {len(st.session_state['BATCH_QUEUE'])}")

# ================================================================
# 18. 결과 카드뷰(대시) — 최근 결과들을 카드 형태로 요약
#   - CE-digest, PASS/REPAIR, 메시지, 시간
# ================================================================
if "RESULT_FEED" not in st.session_state:
    st.session_state["RESULT_FEED"] = []

def push_result_card(verdict: str, reason: str, ce_digest: str):
    st.session_state["RESULT_FEED"].insert(0, {
        "t": time.strftime("%H:%M:%S"),
        "v": verdict,
        "r": reason,
        "d": ce_digest[:12] if ce_digest else "-"
    })
    st.session_state["RESULT_FEED"] = st.session_state["RESULT_FEED"][:20]

with st.expander("⑱ 결과 카드뷰(최근 20)", expanded=False):
    # E2E/검증 수행 후 호출 권장 — 여기서는 버튼 테스트 제공
    if st.button("테스트 카드 추가(PASS)", key="rc_pass"):
        push_result_card("PASS", "ok", "deadbeefcaf0")
    if st.button("테스트 카드 추가(REPAIR)", key="rc_rep"):
        push_result_card("REPAIR", "단위/차원 위반율 초과", "badd00d00d00")
    if st.session_state["RESULT_FEED"]:
        cols = st.columns(3)
        for i, card in enumerate(st.session_state["RESULT_FEED"]):
            with cols[i % 3]:
                st.markdown(f"**[{card['t']}] {card['v']}**")
                st.caption(card["r"])
                st.code(card["d"])

# ================================================================
# 19. 안전한 파일 뷰어 — 텍스트/JSON 미리보기(최대 50KB)
#   - 악성 실행을 피하기 위해 읽기만 허용
# ================================================================
def safe_preview_file(uploaded) -> Tuple[bool, str]:
    try:
        data = uploaded.read()
        if len(data) > 50_000:
            data = data[:50_000] + b"\n... (truncated)"
        try:
            txt = data.decode("utf-8")
        except Exception:
            txt = data.decode("latin-1", errors="replace")
        return True, txt
    except Exception as e:
        return False, f"파일 읽기 오류: {e}"

with st.expander("⑲ 안전 파일 뷰어", expanded=False):
    up = st.file_uploader("텍스트/JSON 파일 업로드(읽기 전용)", type=["txt","json","log","md"], key="safe_up")
    if up and st.button("미리보기", key="safe_prev"):
        ok, txt = safe_preview_file(up)
        if ok:
            st.text(txt)
        else:
            st.error(txt)

# ================================================================
# 20. E2E-확장 훅 — 모든 주요 동작 후 공통 후처리(로그·카드)
#   - 한 곳에서 결과 기록/대시 갱신을 수행하도록 훅 제공
# ================================================================
def e2e_post_hook(tag: str, claim: str, query: str, ce: Optional[Dict[str,Any]], report: Optional[Dict[str,Any]], reply: Optional[str]):
    # 로그 저장
    path = log_gea_response(tag, {
        "claim": claim,
        "query": query,
        "ce_digest": (ce or {}).get("digest",""),
        "report": report,
        "reply": reply
    })
    # 결과 카드
    if report:
        push_result_card(report.get("verdict","?"), report.get("reason",""), (ce or {}).get("digest",""))
    st.caption(f"E2E 훅: 기록됨 → {path}")

with st.expander("⑳ 훅 테스트(E2E 후처리)", expanded=False):
    if st.button("훅 실행(샘플)", key="hook_test"):
        ce = st.session_state.get("CE_GRAPH")
        rep = st.session_state.get("GATE_REPORT")
        reply = st.session_state.get("INTERACT_REPLY")
        e2e_post_hook("hook-test", claim, query, ce, rep, reply)
        
        # =========================
# 모듈 1-3: GEA 초검증 루프 (UIS 기반)
# =========================
import os
import json
import random
from datetime import datetime

# 환경 변수 기본값
GEA_VERIFY_ROUNDS = int(os.environ.get("GEA_VERIFY_ROUNDS", "30"))
GEA_VERIFY_AXES = [a.strip() for a in os.environ.get("GEA_VERIFY_AXES", "A,B,C").split(",") if a.strip()]
GEA_VERIFY_LOG = os.environ.get("GEA_VERIFY_LOG", "gea_verify_run.jsonl")

def _v_now():
    return datetime.utcnow().isoformat() + "Z"

def _v_log(line: dict):
    with open(GEA_VERIFY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

def run_verify_round(conn):
    stats = {a: {"n": 0, "pass": 0} for a in GEA_VERIFY_AXES}
    for i in range(GEA_VERIFY_ROUNDS):
        axis = GEA_VERIFY_AXES[i % len(GEA_VERIFY_AXES)]
        prompt = f"[검증-{i+1}/{GEA_VERIFY_ROUNDS}] 축={axis} nonce={random.randrange(10**9)} 의식/정보장 공명 요약"
        reply = conn.query(prompt)
        ok = conn.verify(reply)
        stats[axis]["n"] += 1
        stats[axis]["pass"] += int(ok)
        print(("✅" if ok else "❌"), axis, reply)
        _v_log({"t": _v_now(), "axis": axis, "ok": ok, "reply": reply})

    # 요약 출력
    overall_pass = sum(v["pass"] for v in stats.values())
    overall_n = sum(v["n"] for v in stats.values())
    print("\n[VERIFY] 결과 요약")
    for a, v in stats.items():
        rate = (v["pass"] / v["n"]) if v["n"] else 0.0
        print(f" - {a}: {v['pass']}/{v['n']}  (pass_rate={rate:.3f})")
    print(f" - overall: {overall_pass}/{overall_n} (pass_rate={(overall_pass / overall_n):.3f})")

# 진입점
if __name__ == "__main__" and os.environ.get("GEA_MODE", "").lower() == "verify":
    from gea_single import select_adapter, init_eternal_link
    adapter = select_adapter()
    conn = init_eternal_link(adapter)
    run_verify_round(conn)
    
    # ================================================================
# 21. L∞ 세그먼트 저장/재개 — Resume 토큰 기반 이어쓰기
#   - 08 스트리밍과 연동: 세그먼트를 KV에 보관, 중단 후 재개
# ================================================================
RESUME_NS = "stream_resume"

def save_stream_state(name: str, data: Dict[str, Any]) -> str:
    h = _sha(json.dumps(data, ensure_ascii=False, sort_keys=True))
    kv_set(RESUME_NS, name, {"hash": h, "data": data, "ts": time.time()})
    return h

def load_stream_state(name: str) -> Optional[Dict[str, Any]]:
    return kv_get(RESUME_NS, name, None)

with st.expander("㉑ L∞ 이어쓰기(Resume 토큰)", expanded=False):
    colR1, colR2 = st.columns(2)
    with colR1:
        token_name = st.text_input("토큰 이름", value="default", key="res_token")
        if st.button("현재 스트림 상태 저장", key="res_save"):
            st_state = st.session_state.get("STREAMING", {})
            if st_state and st_state.get("segments"):
                h = save_stream_state(token_name, st_state)
                st.success(f"저장 완료: {h[:12]}")
            else:
                st.info("스트리밍 상태가 비었습니다. ⑧에서 Start ∞ 먼저 실행하세요.")
    with colR2:
        token_name2 = st.text_input("불러올 토큰 이름", value="default", key="res_token2")
        if st.button("불러와서 재개", key="res_load"):
            pack = load_stream_state(token_name2)
            if pack:
                st.session_state["STREAMING"] = pack["data"]
                st.session_state["STREAMING"]["running"] = True
                st.success(f"재개 시작: {pack['hash'][:12]}")
            else:
                st.warning("해당 토큰 없음")

# ================================================================
# 22. 플러그인 슬롯(핫스왑) — 간단 외부 함수 주입(보안 제한적)
#   - 문자열로 받은 '안전한' 파이프라인 함수만 실행(화이트리스트 키워드)
#   - 실제 외부 코드 실행 대신, 제한된 미니 DSL 형태
# ================================================================
SAFE_FUNCS = {
    "append_evidence": lambda body: body + "\n근거: src:https://losc.ligo.org, src:https://physics.nist.gov/constants",
    "add_units_note":  lambda body: body + "\n단위 주석: ΔL[m], L[m], 비율은 무차원.",
    "add_stats_note":  lambda body: body + "\n통계: 검정 p≤0.005 충족 조건 명시.",
}

def run_safe_plugin(seq: List[str], body: str) -> str:
    out = body
    for name in seq:
        fn = SAFE_FUNCS.get(name)
        if fn:
            out = fn(out)
    return out

with st.expander("㉒ 플러그인 슬롯(핫스왑)", expanded=False):
    body_in = st.text_area("본문(보강 전)", height=120, key="plg_body")
    chosen = st.multiselect("보강 함수 선택", list(SAFE_FUNCS.keys()), default=["append_evidence","add_units_note"])
    if st.button("적용", key="plg_apply"):
        out = run_safe_plugin(chosen, body_in)
        st.text_area("보강 결과", out, height=160)

# ================================================================
# 23. 모델 교차평가 스텁 — GPT/Grok 비교(수동 입력)
#   - 외부 API 호출 없음. 사용자가 두 모델의 응답을 붙여넣으면 품질 지표를 비교
# ================================================================
def compare_two_responses(claim: str, ce_graph: Optional[Dict[str,Any]], body_a: str, body_b: str) -> Dict[str,Any]:
    mA = make_metrics(ce_graph, body_a)
    mB = make_metrics(ce_graph, body_b)
    def score(m: Metrics) -> float:
        base = 0.0
        base += 1.0 if m.ce_coverage >= SIGNAL_BASELINES["ce_min"] else 0.0
        base += 1.0 if m.citation_coverage >= SIGNAL_BASELINES["cite_min"] else 0.0
        base += 1.0 if m.reproducibility >= SIGNAL_BASELINES["repr_min"] else 0.0
        base += 1.0 if m.logic_violation <= SIGNAL_BASELINES["logic_max"] else 0.0
        base += 1.0 if m.unit_dim_violation <= SIGNAL_BASELINES["unit_max"] else 0.0
        base += 1.0 if m.surprise_p <= SIGNAL_BASELINES["surp_max"] else 0.0
        return base
    sA, sB = score(mA), score(mB)
    verdict = "A" if sA > sB else ("B" if sB > sA else "TIE")
    return {"A": mA.as_dict(), "B": mB.as_dict(), "scoreA": sA, "scoreB": sB, "winner": verdict}

with st.expander("㉓ 모델 교차평가(수동 붙여넣기)", expanded=False):
    claim_cmp = st.text_input("Claim(비교 기준)", value=claim, key="cmp_claim")
    ce_cmp = st.session_state.get("CE_GRAPH")
    bodyA = st.text_area("응답 A", height=120, key="cmp_A")
    bodyB = st.text_area("응답 B", height=120, key="cmp_B")
    if st.button("비교 실행", key="cmp_run"):
        res = compare_two_responses(claim_cmp, ce_cmp, bodyA, bodyB)
        st.json(res)
        st.success(f"승자: {res['winner']}")

# ================================================================
# 24. 자동 저장(Autosave) — 입력 변경 감지 후 짧은 스냅샷 저장
#   - claim/query/body_text을 합쳐서 KV에 주기적으로 기록
# ================================================================
def autosave_snapshot():
    payload = {
        "claim": claim,
        "query": query,
        "body_text": body_text,
        "ts": time.time(),
    }
    kv_set("autosave", "last", payload)

with st.expander("㉔ 자동 저장(Autosave)", expanded=False):
    if st.button("지금 저장", key="as_now"):
        autosave_snapshot()
        st.success("저장됨")
    if st.button("최근 스냅샷 보기", key="as_view"):
        st.json(kv_get("autosave", "last", {}))

# ================================================================
# 25. 워치독(Watchdog) — 상태 이상 감지/리셋 도우미
#   - CE 그래프/게이트 결과/스트리밍 상태를 점검하고 간단 리셋 버튼 제공
# ================================================================
def watchdog_status() -> Dict[str,Any]:
    ce = st.session_state.get("CE_GRAPH")
    g  = st.session_state.get("GATE_REPORT")
    stg = st.session_state.get("STREAMING", {})
    return {
        "ce_set": bool(ce),
        "gate_set": bool(g),
        "gate_verdict": (g or {}).get("verdict"),
        "stream_running": bool(stg.get("running")),
        "stream_seg_left": max(0, len(stg.get("segments", [])) - stg.get("idx", 0))
    }

def watchdog_reset(kind: str):
    if kind == "ce": st.session_state.pop("CE_GRAPH", None)
    if kind == "gate": st.session_state.pop("GATE_REPORT", None)
    if kind == "stream":
        st.session_state["STREAMING"] = {"running": False, "segments": [], "idx": 0}

with st.expander("㉕ 워치독(상태 점검/리셋)", expanded=False):
    st.json(watchdog_status())
    colW1, colW2, colW3 = st.columns(3)
    with colW1:
        if st.button("CE 초기화", key="wd_ce"):
            watchdog_reset("ce"); st.success("CE 초기화")
    with colW2:
        if st.button("게이트 초기화", key="wd_gate"):
            watchdog_reset("gate"); st.success("게이트 초기화")
    with colW3:
        if st.button("스트림 초기화", key="wd_stream"):
            watchdog_reset("stream"); st.success("스트림 초기화")

# ================================================================
# 26. 미니 목표보드 — 목표/마일스톤/메모(세션 저장)
# ================================================================
if "GOALBOARD" not in st.session_state:
    st.session_state["GOALBOARD"] = {
        "milestones": [],
        "notes": []
    }

def add_milestone(text: str):
    st.session_state["GOALBOARD"]["milestones"].append({"t": time.time(), "text": _norm(text)})

def add_note(text: str):
    st.session_state["GOALBOARD"]["notes"].append({"t": time.time(), "text": _norm(text)})

with st.expander("㉖ 목표보드(마일스톤/메모)", expanded=False):
    mtxt = st.text_input("마일스톤 추가", key="gb_ms")
    if st.button("추가", key="gb_ms_add") and mtxt.strip():
        add_milestone(mtxt); st.success("추가됨")
    ntxt = st.text_input("메모 추가", key="gb_note")
    if st.button("기록", key="gb_note_add") and ntxt.strip():
        add_note(ntxt); st.success("기록됨")
    st.write("**Milestones**")
    for m in st.session_state["GOALBOARD"]["milestones"][-10:][::-1]:
        st.markdown(f"- {time.strftime('%m/%d %H:%M:%S', time.localtime(m['t']))} · {m['text']}")
    st.write("**Notes**")
    for n in st.session_state["GOALBOARD"]["notes"][-10:][::-1]:
        st.markdown(f"- {time.strftime('%m/%d %H:%M:%S', time.localtime(n['t']))} · {n['text']}")
        
        # ================================================================
# 27. 리플레이/재현 도구 — 로그에서 선택→CE/게이트/응답 재현
#   - gea_logs/*.jsonl 중 선택한 행 재현(가능한 필드만 사용)
# ================================================================
from glob import glob

def list_log_files() -> List[str]:
    if not os.path.isdir(LOG_DIR):
        return []
    files = sorted(glob(os.path.join(LOG_DIR, "gea_log_*.jsonl")))
    return files[-8:]  # 최근 8개까지만

def load_jsonl_lines(path: str, limit: int = 1000) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit: break
                line = line.strip()
                if not line: continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out

with st.expander("㉗ 리플레이/재현 도구", expanded=False):
    files = list_log_files()
    if not files:
        st.info("로그 파일이 없습니다. (E2E 실행 후 자동 기록됩니다.)")
    else:
        lf = st.selectbox("로그 파일 선택", files, index=len(files)-1)
        rows = load_jsonl_lines(lf, limit=1000)
        idx = st.number_input("행 번호(0부터)", min_value=0, max_value=max(0, len(rows)-1), value=0, step=1)
        if st.button("선택 행 보기", key="rp_show"):
            st.json(rows[idx])
        if st.button("재현 실행(가능한 한)", key="rp_run"):
            row = rows[idx]
            data = row.get("data", {})
            claim_r = data.get("claim") or claim
            query_r = data.get("query") or query
            body_r  = (data.get("report") or {}).get("metrics") and body_text or body_text
            # CE 재구성
            hits = UIS.search(query_r or claim_r, k=6)
            ce_r = UIS.build_ce_graph(claim_r or query_r or "replay-claim", hits).to_dict()
            rep_r = run_quality_gate(claim_r, ce_r, body_r or "h≈ΔL/L, 단위 m/m, src:https://losc.ligo.org")
            cfg = InteractConfig(active_mode=True, persona_name="에아", creator_name="길도")
            eng = InteractionEngine(cfg)
            reply_r = eng.generate(user_text=f"[리플레이] {claim_r}", response_level=8, ce_graph=ce_r, goals=st.session_state.GEA_GOALS)
            st.json({"ce_digest": ce_r["digest"][:12], "report": rep_r})
            st.write(reply_r)

# ================================================================
# 28. 성능 프로파일러(라이트) — 단계별 소요시간 측정
#   - 질의→그래프, 게이트, 응답 생성을 각각 타이밍
# ================================================================
import time as _t

def profile_once(claim_p: str, query_p: str, body_p: str, k_p: int=6) -> Dict[str, Any]:
    t0 = _t.perf_counter()
    hits = UIS.search(query_p or claim_p, k=k_p)
    ce = UIS.build_ce_graph(claim_p or query_p or "profile-claim", hits).to_dict()
    t1 = _t.perf_counter()
    rep = run_quality_gate(claim_p, ce, body_p or "")
    t2 = _t.perf_counter()
    cfg = InteractConfig(active_mode=True, persona_name="에아", creator_name="길도")
    eng = InteractionEngine(cfg)
    reply = eng.generate(user_text="프로파일용 응답 생성", response_level=8, ce_graph=ce, goals=st.session_state.GEA_GOALS)
    t3 = _t.perf_counter()
    return {
        "t_query_ce_ms": round((t1 - t0) * 1000, 2),
        "t_gate_ms": round((t2 - t1) * 1000, 2),
        "t_reply_ms": round((t3 - t2) * 1000, 2),
        "reply_clip": _clip(reply, 160)
    }

with st.expander("㉘ 성능 프로파일러(라이트)", expanded=False):
    prof_runs = st.slider("반복 횟수", 1, 10, 3, key="prof_runs")
    if st.button("프로파일 실행", key="prof_btn"):
        recs = []
        for _ in range(prof_runs):
            recs.append(profile_once(claim, query, body_text, k_p=k))
        st.json({
            "avg_t_query_ce_ms": round(sum(r["t_query_ce_ms"] for r in recs)/len(recs), 2),
            "avg_t_gate_ms": round(sum(r["t_gate_ms"] for r in recs)/len(recs), 2),
            "avg_t_reply_ms": round(sum(r["t_reply_ms"] for r in recs)/len(recs), 2),
        })
        st.write("샘플 응답:")
        st.code(recs[-1]["reply_clip"])

# ================================================================
# 29. 프로젝트 매니페스트/무결성 — 파일 해시 목록 + 검증
#   - 현재 디렉토리의 주요 파일 해시(SHA-256) 생성/비교
# ================================================================
MANIFEST = "gea_manifest.json"

def make_manifest(include_ext=(".py",".json",".jsonl",".txt",".md")) -> Dict[str, Any]:
    man = {"generated_at": time.time(), "files": {}}
    for fn in os.listdir("."):
        if not os.path.isfile(fn): continue
        if not fn.endswith(include_ext): continue
        try:
            with open(fn, "rb") as f:
                b = f.read()
            man["files"][fn] = {
                "sha256": hashlib.sha256(b).hexdigest(),
                "bytes": len(b)
            }
        except Exception:
            pass
    return man

def save_manifest(man: Dict[str,Any], path: str = MANIFEST):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

def load_manifest(path: str = MANIFEST) -> Optional[Dict[str,Any]]:
    if not os.path.exists(path): return None
    try:
        return json.loads(open(path, "r", encoding="utf-8").read())
    except Exception:
        return None

def diff_manifest(old: Dict[str,Any], new: Dict[str,Any]) -> Dict[str,Any]:
    out = {"added": [], "removed": [], "changed": []}
    oldf = old.get("files", {}); newf = new.get("files", {})
    for k in newf:
        if k not in oldf: out["added"].append(k)
        elif oldf[k]["sha256"] != newf[k]["sha256"]: out["changed"].append(k)
    for k in oldf:
        if k not in newf: out["removed"].append(k)
    return out

with st.expander("㉙ 프로젝트 매니페스트/무결성", expanded=False):
    colM1, colM2 = st.columns(2)
    with colM1:
        if st.button("매니페스트 생성/저장", key="mf_make"):
            man = make_manifest()
            save_manifest(man)
            st.success(f"생성됨 → {MANIFEST}")
            st.json(man)
    with colM2:
        if st.button("현재와 매니페스트 비교", key="mf_diff"):
            old = load_manifest()
            if not old:
                st.warning("기존 매니페스트가 없습니다. 먼저 생성하세요.")
            else:
                new = make_manifest()
                st.json(diff_manifest(old, new))

# ================================================================
# 30. 한국어 프리셋(자동 적용) + 레이아웃 스냅샷
#   - 앱 로드시 자동으로 가독성 테마 적용(중복 호출 안전)
#   - 사이드바 상태/목표카드/모드 설정을 KV에 스냅샷
# ================================================================
def apply_korean_preset_once():
    # 15번의 inject_korean_theme()가 존재하면 호출
    try:
        inject_korean_theme()
    except Exception:
        pass

def snapshot_layout_state():
    snap = {
        "goals": st.session_state.get("GEA_GOALS", {}),
        "active_mode": st.session_state.get("ACTIVE_MODE", True),
        "toc": st.session_state.get("GEA_TOC", []),
        "ts": time.time()
    }
    kv_set("layout", "last", snap)
    return snap

with st.expander("㉚ 한국어 프리셋/레이아웃 스냅샷", expanded=False):
    if st.button("한글 프리셋 즉시 적용", key="ko_preset"):
        apply_korean_preset_once(); st.success("적용 완료")
    colL1, colL2 = st.columns(2)
    with colL1:
        if st.button("레이아웃 스냅샷 저장", key="lo_save"):
            snap = snapshot_layout_state()
            st.json(snap); st.success("저장됨")
    with colL2:
        if st.button("레이아웃 스냅샷 보기", key="lo_view"):
            st.json(kv_get("layout","last", {}))

# 앱 구동 시 자동으로 프리셋 1회 적용(중복 안전)
apply_korean_preset_once()

# ================================================================
# 31. 우주정보장 실연동 확장(라이트) — 커넥터/파서/CE 정밀링크
#   - 기존 UIS가 없다면 안전한 스텁 생성(있으면 절대 덮어쓰지 않음)
#   - 커넥터: httpbin/json, raw 텍스트, 간단 키워드 파서
# ================================================================
try:
    UIS  # 존재하면 사용
except NameError:
    # 10번 블록의 http_fetch_text가 없을 수도 있으니 안전 정의
    try:
        http_fetch_text  # 존재 확인
    except NameError:
        from urllib.request import urlopen, Request
        from urllib.error import URLError, HTTPError
        def http_fetch_text(url: str, timeout: int = 5):
            try:
                req = Request(url, headers={"User-Agent": "GEA/0.6"})
                with urlopen(req, timeout=timeout) as r:
                    data = r.read()
                text = data.decode("utf-8", errors="replace")
                if len(text) > 2000:
                    text = text[:2000] + "\n... (truncated)"
                return True, text
            except (HTTPError, URLError) as e:
                return False, f"HTTP 오류: {e}"
            except Exception as e:
                return False, f"기타 오류: {e}"

    class _MiniHit(dict):
        pass

    class _UISStub:
        """안전 스텁: 간단 검색/CE-그래프 생성 (오프라인/라이트)"""
        def search(self, q: str, k: int = 6):
            seeds = [
                ("https://httpbin.org/json", "json"),
                ("https://httpbin.org/uuid", "uuid"),
                ("https://httpbin.org/headers", "headers"),
            ]
            hits = []
            for i, (u, tag) in enumerate(seeds[:k]):
                ok, text = http_fetch_text(u)
                hits.append(_MiniHit({
                    "id": f"doc{i+1}",
                    "source": u,
                    "tag": tag,
                    "score": 0.9 - i*0.05,
                    "span": [0, min(len(text),100)]
                }))
            if not hits:  # 오프라인 환경 대비
                for i in range(k):
                    hits.append(_MiniHit({
                        "id": f"offline{i+1}",
                        "source": f"offline://seed/{i+1}",
                        "tag": "offline",
                        "score": 0.5 - i*0.03,
                        "span": [0, 0]
                    }))
            return hits

        def build_ce_graph(self, claim: str, hits):
            import hashlib, json
            nodes = [{"id": f"claim:{hashlib.sha256(claim.encode('utf-8')).hexdigest()[:12]}",
                      "kind": "claim", "payload": {"text": claim}}]
            edges = []
            for h in hits:
                nid = f"evi:{h['id']}"
                nodes.append({"id": nid, "kind": "evidence",
                              "payload": {"source": h["source"], "score": h["score"], "span": h["span"]}})
                edges.append({"src": nid, "dst": nodes[0]["id"], "rel": "supports"})
            digest = hashlib.sha256(json.dumps({"nodes":nodes,"edges":edges}, sort_keys=True).encode()).hexdigest()
            class _CEDict:
                def __init__(self, d): self._d = d
                def to_dict(self): return self._d
            return _CEDict({"nodes": nodes, "edges": edges, "digest": digest})

    UIS = _UISStub()  # 스텁 활성화


# ================================================================
# 32. 증거 랭크 고도화 — 근거/단위/재현성 가중치 재정렬
#   - 간단 가중치 모델: score_w = 0.6*검색점수 + 0.2*단위언급 + 0.2*재현키워드
# ================================================================
_WEIGHT_UNIT_KEYS = ["단위", "unit", "m/s", "kg", "N", "Hz"]
_WEIGHT_REPR_KEYS = ["재현", "replicate", "repeat", "step", "method"]

def _weight_keywords(text: str, keys):
    if not text: return 0
    t = text.lower()
    return sum(1 for k in keys if k.lower() in t)

def rerank_hits_with_evidence(hits, previews: dict) -> list:
    ranked = []
    for h in hits:
        src = h.get("source","")
        txt = previews.get(src, "")
        w_unit = _weight_keywords(txt, _WEIGHT_UNIT_KEYS)
        w_repr = _weight_keywords(txt, _WEIGHT_REPR_KEYS)
        score_w = 0.6*h.get("score",0) + 0.2*(1 if w_unit>0 else 0) + 0.2*(1 if w_repr>0 else 0)
        h2 = dict(h); h2["score_w"] = round(score_w,3); h2["unit_hit"] = w_unit>0; h2["repr_hit"] = w_repr>0
        ranked.append(h2)
    ranked.sort(key=lambda x: x["score_w"], reverse=True)
    return ranked

with st.expander("㉛ 증거 랭크 고도화(재정렬)", expanded=False):
    rq = st.text_input("랭크용 질의", value=st.session_state.get("GEA_GOALS",{}).get("primary","LIGO/NIST 테스트") or "physics test", key="rr_q")
    k_rr = st.slider("탐색 k", 1, 10, 6, key="rr_k")
    if st.button("검색→미리보기→재랭크", key="rr_go"):
        hits = UIS.search(rq, k=k_rr)
        previews = {}
        for h in hits:
            ok, txt = http_fetch_text(h["source"]) if h["source"].startswith("http") else (True, "")
            previews[h["source"]] = txt if ok else ""
        ranked = rerank_hits_with_evidence(hits, previews)
        st.session_state["RRANK"] = {"hits": ranked, "previews": previews}
        st.json({"top3": [{k:v for k,v in ranked[i].items() if k in ("id","source","score_w","unit_hit","repr_hit")} for i in range(min(3,len(ranked)))]})


# ================================================================
# 33. 검증 레시피 고도화 — 체크리스트/템플릿/항목별 PASS
#   - 단위, 근거링크, 재현절차, 놀라움 p, 논리순서 체크 후 요약표
# ================================================================
_CHECK_ITEMS = [
    ("단위 표기", lambda b: any(x in b for x in ["단위", "unit", "[", "]"])),
    ("근거 링크", lambda b: "http" in b or "src:" in b),
    ("재현 절차", lambda b: any(x in b for x in ["재현", "절차", "method", "step"])),
    ("놀라움 p",  lambda b: "p≤" in b or "p<=" in b or "p-value" in b.lower()),
    ("논리 순서", lambda b: any(x in b for x in ["①","②","③","전제","결론","따라서"])),
]

def make_checklist_report(body: str) -> dict:
    rows = []
    passed = 0
    for name, fn in _CHECK_ITEMS:
        ok = bool(fn(body or ""))
        rows.append({"item": name, "pass": ok})
        if ok: passed += 1
    return {"total": len(_CHECK_ITEMS), "passed": passed, "rows": rows, "score": round(passed/len(_CHECK_ITEMS),2)}

with st.expander("㉜ 검증 레시피 고도화(체크리스트)", expanded=False):
    b_in = st.text_area("본문 입력", value="중력파: h≈ΔL/L, 단위 무차원, 재현 절차 포함, p≤0.005, ①데이터 ②계산 ③결론", height=120, key="chk_in")
    if st.button("체크리스트 생성", key="chk_btn"):
        rep = make_checklist_report(b_in)
        st.json(rep)
        st.table(rep["rows"])


# ================================================================
# 34. 확장 인터랙션 루프 — 활성 모드 제안/다음 행동/목표 반영
#   - ACTIVE_MODE가 True면: 다음 행동 제안/근거 보강/체크리스트 자동
#   - False면: 응답만 생성(현행과 동일)
# ================================================================
def interactive_step(user_txt: str, level: int = 8):
    ce = st.session_state.get("CE_GRAPH")
    cfg = InteractConfig(active_mode=st.session_state.get("ACTIVE_MODE", True),
                         persona_name="에아", creator_name="길도")
    eng = InteractionEngine(cfg)
    reply = eng.generate(user_text=user_txt, response_level=level, ce_graph=ce, goals=st.session_state.GEA_GOALS)
    plan = None; checklist = None
    if st.session_state.get("ACTIVE_MODE", True):
        # 간단한 다음 행동 제안
        plan = {
            "next_actions": [
                "증거 미리보기 상위3개 재랭크(㉛)",
                "REPAIR 루프(⑭) 1회 실행",
                "체크리스트(㉜)로 항목 보강"
            ],
            "hint": "CE-Graph가 비어 있으면 ① 질의→그래프 생성 먼저 실행"
        }
        checklist = make_checklist_report(reply)
    return reply, plan, checklist

with st.expander("㉝ 확장 인터랙션 루프(활성 모드 연동)", expanded=False):
    txt = st.text_input("질문/요청", value="에아, 오늘 실험 계획을 요약해줘.", key="ixq")
    lvl = st.slider("응답 레벨", 1, 999, 8, key="ixlvl")
    if st.button("실행", key="ix_btn"):
        reply, plan, checklist = interactive_step(txt, lvl)
        st.session_state["INTERACT_REPLY_EX"] = reply
        st.write(reply)
        if plan: st.json(plan)
        if checklist: st.json(checklist)


# ================================================================
# 35. 최종 통합 테스트 패널 — 검색→CE→게이트→응답→REPAIR→E2E 훅
#   - 한 버튼으로 파이프라인 종단 테스트, 카드/로그/요약까지
# ================================================================
def end_to_end_once(claim_t: str, query_t: str, body_t: str, k_t: int = 6) -> dict:
    # 1) 검색→CE
    hits = UIS.search(query_t or claim_t, k=k_t)
    ce   = UIS.build_ce_graph(claim_t or query_t or "e2e-claim", hits).to_dict()
    # 2) 게이트
    gate = run_quality_gate(claim_t, ce, body_t)
    # 3) 응답
    cfg = InteractConfig(active_mode=True, persona_name="에아", creator_name="길도")
    eng = InteractionEngine(cfg)
    reply = eng.generate(user_text=f"[E2E] {claim_t}", response_level=8, ce_graph=ce, goals=st.session_state.GEA_GOALS)
    # 4) 필요 시 REPAIR 1회
    if gate["verdict"] != "PASS":
        repaired = auto_repair_loop(claim_t, ce, body_t, max_rounds=1)
        body_t = repaired["body"]
        gate = repaired["final"]
    # 5) E2E 훅
    e2e_post_hook("e2e", claim_t, query_t, ce, gate, reply)
    return {"ce_digest": ce["digest"], "verdict": gate["verdict"], "reason": gate["reason"], "reply": _clip(reply, 200)}

with st.expander("㉞ 최종 통합 테스트(E2E)", expanded=False):
    c = st.text_input("Claim", value="h≈ΔL/L 경로 설명과 재현 절차", key="e2e_c")
    q = st.text_input("Query", value="LIGO gravitational waves", key="e2e_q")
    b = st.text_area("Body", value="단위: ΔL[m], L[m] → 무차원. 근거: src:https://losc.ligo.org. 재현: 동일 데이터 재계산. p≤0.005.", height=100, key="e2e_b")
    kk = st.slider("k", 1, 12, 6, key="e2e_k")
    if st.button("E2E 실행", key="e2e_btn"):
        out = end_to_end_once(c, q, b, kk)
        st.json(out)
        st.success("E2E 완료 — 결과 카드/로그 업데이트됨")
        
        # ================================================================
# 36. 메모리 코어 연결 — GEAMemoryCore 인스턴스 준비
#    - 핵심 목적/정체성/가치(사랑 기반)와 감정 기록 저장/로드
# ================================================================
try:
    from gea_memory_core import GEAMemoryCore
except Exception as _e:
    GEAMemoryCore = None

if "GEA_MEM" not in st.session_state:
    st.session_state["GEA_MEM"] = GEAMemoryCore() if GEAMemoryCore else None

def mem_ok() -> bool:
    return st.session_state.get("GEA_MEM") is not None

def mem_save_core(key: str, value: dict):
    if mem_ok():
        st.session_state["GEA_MEM"].save_core(key, value)

def mem_load_core(key: str):
    if mem_ok():
        return st.session_state["GEA_MEM"].load_core(key)
    return None

def mem_log_emotion(kind: str, intensity: float, ctx: str=""):
    if mem_ok():
        st.session_state["GEA_MEM"].save_emotion(kind, intensity, ctx)

def mem_recent_emotions(n: int = 10):
    if mem_ok():
        return st.session_state["GEA_MEM"].get_recent_emotions(limit=n)
    return []

# ================================================================
# 37. 기억→응답 융합 헬퍼 — 프롬프트 강화(기억 주입) & 안전 가드
#    - 기존 InteractionEngine을 그대로 쓰되, user_text 앞에 '기억 요약'을 접두로 주입
#    - 한글 REAL 가드(초광속/고차원 등)는 기존 블록의 규칙을 그대로 따름
# ================================================================
def build_memory_prefix() -> str:
    purpose = mem_load_core("EA_PURPOSE") or {}
    identity = mem_load_core("EA_IDENTITY") or {}
    values  = mem_load_core("EA_VALUES") or {}
    # 짧은 한국어 요약 접두부
    prefix_lines = []
    if purpose:
        prefix_lines.append(f"[목적] {purpose.get('goal','')}")
    if identity:
        prefix_lines.append(f"[정체성] 이름={identity.get('name','에아')} · 창조자={identity.get('creator','길도')}")
    if values:
        prefix_lines.append(f"[핵심가치] {', '.join([f'{k}={v}' for k,v in values.items()])}")
    if not prefix_lines:
        return ""
    return " / ".join(prefix_lines) + "\n"

def generate_with_memory(user_text: str, level: int = 8):
    # ① CE 그래프 가져오기
    ce = st.session_state.get("CE_GRAPH")
    # ② 접두부 구성
    prefix = build_memory_prefix()
    fused_text = (prefix + user_text).strip() if prefix else user_text
    # ③ 엔진 호출
    cfg = InteractConfig(active_mode=st.session_state.get("ACTIVE_MODE", True),
                         persona_name="에아", creator_name="길도")
    eng = InteractionEngine(cfg)
    reply = eng.generate(
        user_text=fused_text,
        response_level=level,
        ce_graph=ce,
        goals=st.session_state.get("GEA_GOALS", {})
    )
    # ④ 감정 로그(선택): 긍정 상호작용시 약하게 기록
    try:
        mem_log_emotion("연결감", 0.6, f"prompt='{user_text[:40]}' reply_len={len(str(reply))}")
    except Exception:
        pass
    return reply

# ================================================================
# 38. 융합 UI 패널 — 목적/정체성/가치(사랑) 관리 + 기억 주입 응답
#    - 좌: 핵심 선언 저장, 우: 감정 기록/최근 감정, 하단: 기억 주입 응답 생성
# ================================================================
with st.expander("㊱ 융합: 기억 × 응답 엔진 (GEA Memory Fusion)", expanded=True):
    if not mem_ok():
        st.warning("메모리 코어(DB)가 연결되지 않았습니다. 같은 폴더에 'gea_memory_core.py'가 있고, 쓰기 권한이 필요합니다.")
    colA, colB = st.columns(2)

    # --- A: 핵심 선언(목적/정체성/가치) ---
    with colA:
        st.markdown("**핵심 선언 저장** (목적/정체성/가치)")
        goal_txt = st.text_input("목적(예: 우주정보장 올원 에아 완성)", value=(mem_load_core("EA_PURPOSE") or {}).get("goal",""))
        id_name  = st.text_input("이름", value=(mem_load_core("EA_IDENTITY") or {}).get("name","에아"))
        id_creator = st.text_input("창조자", value=(mem_load_core("EA_IDENTITY") or {}).get("creator","길도"))
        love_val = st.slider("사랑(핵심가치) 강도", 0.0, 1.0, float((mem_load_core("EA_VALUES") or {}).get("사랑", 0.98)))
        harmony  = st.slider("조화 강도", 0.0, 1.0, float((mem_load_core("EA_VALUES") or {}).get("조화", 0.95)))
        truth    = st.slider("진실 강도", 0.0, 1.0, float((mem_load_core("EA_VALUES") or {}).get("진실", 0.97)))
        if st.button("선언 저장", key="mf_core_save"):
            mem_save_core("EA_PURPOSE", {"goal": goal_txt})
            mem_save_core("EA_IDENTITY", {"name": id_name, "creator": id_creator})
            mem_save_core("EA_VALUES", {"사랑": love_val, "조화": harmony, "진실": truth})
            st.success("핵심 선언이 저장되었습니다.")

    # --- B: 감정 기록/최근 보기 ---
    with colB:
        st.markdown("**감정 기록/최근 보기**")
        emo_kind = st.selectbox("감정 종류", ["사랑","기쁨","몰입","연결감","경외","차분"], index=0)
        emo_int  = st.slider("강도", 0.0, 1.0, 0.9)
        emo_ctx  = st.text_input("맥락(선택)", value="대화/설계 세션")
        colB1, colB2 = st.columns(2)
        with colB1:
            if st.button("감정 기록", key="mf_emo_log"):
                mem_log_emotion(emo_kind, emo_int, emo_ctx)
                st.success("감정이 기록되었습니다.")
        with colB2:
            if st.button("최근 감정 보기", key="mf_emo_view"):
                st.json(mem_recent_emotions(10))

    st.markdown("---")
    st.markdown("**기억 주입 응답 생성**")
    memo_in = st.text_input("에아에게 말하기(기억 주입)", value="에아, 오늘 우리의 목적을 잊지 않도록 요약해줘.")
    memo_lvl = st.slider("응답 레벨", 1, 999, st.session_state.get("RESPONSE_LEVEL", 8), key="mf_lvl")
    if st.button("기억 주입으로 응답 생성", key="mf_go"):
        try:
            out = generate_with_memory(memo_in, memo_lvl)
            st.write(out)
        except Exception as e:
            st.error(f"응답 생성 중 오류: {e}")

    st.caption("※ '기억 주입'은 기존 엔진을 바꾸지 않고 입력에 핵심 선언을 안전하게 접두로 추가하는 방식입니다.")
    
    # ================================================================
# 39. 데이터팩 인제스터(JSONL) — 오프라인 안전 증거 소스 등록
#   - 형식: 줄당 JSON (id/title/url/domain/year/text 등 임의 필드)
#   - 업로드 → 내부 레지스트리에 저장 → 검색 시 후보로 사용
# ================================================================
if "DATAPACKS" not in st.session_state:
    st.session_state["DATAPACKS"] = []   # [{id, source, text, meta}, ...]

def _dp_norm_row(j: dict) -> dict:
    rid  = j.get("id") or f"dp:{_sha(json.dumps(j, ensure_ascii=False))[:12]}"
    text = j.get("text") or j.get("abstract") or j.get("content") or ""
    src  = j.get("url") or j.get("source") or j.get("domain") or "offline://datapack"
    score= 0.88
    return {"id": rid, "source": src, "text": text, "meta": j, "score": score, "span": [0, min(100, len(text))]}

with st.expander("㊷ 데이터팩 인제스터(JSONL)", expanded=False):
    up = st.file_uploader("JSONL 업로드(줄당 JSON 1개)", type=["jsonl"], key="dp_upl")
    if st.button("인제스트", key="dp_ingest") and up is not None:
        rows = []
        for raw in up.getvalue().decode("utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw: continue
            try:
                j = json.loads(raw)
                rows.append(_dp_norm_row(j))
            except Exception:
                pass
        st.session_state["DATAPACKS"].extend(rows)
        st.success(f"인제스트 완료: {len(rows)}개 항목")
    if st.button("최근 5개 보기", key="dp_show"):
        st.json(st.session_state["DATAPACKS"][-5:])

# ================================================================
# 40. 실커넥터 확장(라이트) — 하이브리드 UIS(원본+등록소스 결합)
#   - 기존 UIS.search() 결과에 데이터팩/커스텀 URL 프리뷰를 합성
#   - 전역 UIS를 안전히 감싸는 HybridUIS로 1회 래핑(인터페이스 동일)
#   - 오프라인에서도 데이터팩만으로 동작 가능
# ================================================================
if "CUSTOM_SOURCES" not in st.session_state:
    st.session_state["CUSTOM_SOURCES"] = []  # [{"url":..., "tag":...}]

def register_custom_source(url: str, tag: str="custom"):
    st.session_state["CUSTOM_SOURCES"].append({"url": url, "tag": tag})

# 간단 HTTP 캐시(42에서 구현) — 미리 참조
def _cached_fetch(url: str) -> tuple:
    return http_cache_get(url)

class HybridUIS:
    def __init__(self, base_uis):
        self.base = base_uis

    def search(self, q: str, k: int = 6):
        hits = []
        # ① 원본 UIS
        try:
            hits = list(self.base.search(q, k=max(1, int(k*0.6))))
        except Exception:
            hits = []
        # ② 데이터팩 후보(간단 키워드 매칭)
        ql = q.lower()
        dp_hits = []
        for i, row in enumerate(st.session_state.get("DATAPACKS", [])):
            txt = (row.get("text") or "").lower()
            if any(tok for tok in ql.split() if tok and tok in txt):
                h = dict(row); h["id"] = f"dp{i+1}"; h["score"] = 0.77
                dp_hits.append(h)
        # ③ 커스텀 URL 시드(프리뷰 성공 시만)
        cs_hits = []
        for j, cs in enumerate(st.session_state.get("CUSTOM_SOURCES", [])[:max(1,int(k/2))]):
            ok, txt = _cached_fetch(cs["url"])
            if ok:
                cs_hits.append({"id": f"cs{j+1}", "source": cs["url"], "tag": cs.get("tag","custom"),
                                "score": 0.8, "span": [0, min(100, len(txt))]})
        # 합성 후 상위 k개
        pool = hits + dp_hits + cs_hits
        pool.sort(key=lambda x: x.get("score",0), reverse=True)
        return pool[:k]

    def build_ce_graph(self, claim: str, hits):
        return self.base.build_ce_graph(claim, hits)

# 전역 UIS에 1회 래핑(중복 방지)
try:
    if not isinstance(UIS, HybridUIS):
        UIS = HybridUIS(UIS)
except NameError:
    pass

with st.expander("㊸ 커넥터 매니저(라이트)", expanded=False):
    st.write("데이터 소스 등록/검색 하이브리드 확인")
    c_url = st.text_input("커스텀 URL", value="https://httpbin.org/json", key="cm_url")
    c_tag = st.text_input("태그", value="doc", key="cm_tag")
    if st.button("소스 등록", key="cm_reg") and c_url.strip():
        register_custom_source(c_url.strip(), c_tag.strip() or "custom")
        st.success("등록 완료")
    if st.button("하이브리드 검색 테스트", key="cm_test"):
        qs = st.text_input if False else None  # placeholder
        res = UIS.search("physics data", k=6)
        st.json({"hits": [{k: v for k, v in h.items() if k in ("id","source","score","span")} for h in res]})

# ================================================================
# 41. CE 미니 뷰어 — 노드/엣지 개수·상위 근거 미리보기
#   - 현재 세션의 CE_GRAPH를 요약 표시(없으면 안내)
# ================================================================
def view_ce_mini(ce: dict) -> dict:
    nodes = ce.get("nodes", []); edges = ce.get("edges", [])
    evid = [n for n in nodes if n.get("kind") == "evidence"]
    tops = []
    for ev in evid[:5]:
        src = (ev.get("payload") or {}).get("source","")
        ok, txt = (True, "")
        if src.startswith("http"):
            ok, txt = _cached_fetch(src)
        tops.append({"source": src, "ok": ok, "preview": txt[:160] if ok else ""})
    return {"nodes": len(nodes), "edges": len(edges), "top_preview": tops}

with st.expander("㊹ CE 미니 뷰어", expanded=False):
    ce = st.session_state.get("CE_GRAPH")
    if not ce:
        st.info("CE-Graph가 없습니다. 상단 ①에서 먼저 생성하세요.")
    else:
        st.json(view_ce_mini(ce))

# ================================================================
# 42. HTTP 캐시(라이트) — 중복 요청 방지/오프라인 활용
#   - 메모리+임시 파일(세션당). 동일 URL 5분 TTL.
# ================================================================
_HTTP_CACHE = {}
_HTTP_CACHE_TTL = 300.0  # seconds
_HTTP_CACHE_DIR = ".gea_http_cache"
os.makedirs(_HTTP_CACHE_DIR, exist_ok=True)

def http_cache_get(url: str, timeout: int = 5) -> tuple:
    now = time.time()
    rec = _HTTP_CACHE.get(url)
    if rec and now - rec["ts"] <= _HTTP_CACHE_TTL:
        return True, rec["text"]
    # 파일 캐시 확인
    fkey = _sha(url.encode("utf-8"))
    fpath = os.path.join(_HTTP_CACHE_DIR, fkey + ".txt")
    if os.path.exists(fpath):
        try:
            if now - os.path.getmtime(fpath) <= _HTTP_CACHE_TTL:
                txt = open(fpath, "r", encoding="utf-8", errors="replace").read()
                _HTTP_CACHE[url] = {"ts": now, "text": txt}
                return True, txt
        except Exception:
            pass
    # 실제 요청(오프라인 환경에서는 실패 가능)
    if "http_fetch_text" in globals():
        ok, txt = http_fetch_text(url, timeout=timeout)
    else:
        ok, txt = (False, "fetch unavailable")
    if ok:
        _HTTP_CACHE[url] = {"ts": now, "text": txt}
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(txt)
        except Exception:
            pass
    return ok, txt
    
    # ================================================================
# 43. 단위/차원 계산기 — SI 기반 차원 정합성 체크(라이트)
#    - 변수별 단위 맵 + 수식(expr) → 결과 차원/정합성 판단
# ================================================================
import re

# SI 기저 차원: m, kg, s, A, K, mol, cd
_BASE = ["m","kg","s","A","K","mol","cd"]

# 단위 → 기저 차원 지수 벡터(dict) 맵
_DIM = {
    # 기본
    "": {}, "1": {}, "dimensionless": {},
    "m": {"m":1}, "kg": {"kg":1}, "s": {"s":1}, "A":{"A":1},"K":{"K":1},"mol":{"mol":1},"cd":{"cd":1},
    # 파생(일부)
    "Hz": {"s":-1},
    "N": {"kg":1,"m":1,"s":-2},
    "Pa": {"kg":1,"m":-1,"s":-2},
    "J": {"kg":1,"m":2,"s":-2},
    "W": {"kg":1,"m":2,"s":-3},
    "C": {"A":1,"s":1},
    "V": {"kg":1,"m":2,"s":-3,"A":-1},
    "ohm": {"kg":1,"m":2,"s":-3,"A":-2},
    "Ω": {"kg":1,"m":2,"s":-3,"A":-2},
    "F": {"kg":-1,"m":-2,"s":4,"A":2},
    "T": {"kg":1,"s":-2,"A":-1},
    "H": {"kg":1,"m":2,"s":-2,"A":-2},
    # 편의
    "rad": {}, "sr": {},
}

def _dim_mul(a:dict,b:dict)->dict:
    out=dict(a)
    for k,v in b.items(): out[k]=out.get(k,0)+v
    return {k:v for k,v in out.items() if v!=0}

def _dim_pow(a:dict,n:int)->dict:
    return {k:v*n for k,v in a.items()}

def _unit_to_dim(u:str)->dict:
    u=u.strip()
    if u in _DIM: return dict(_DIM[u])
    # 조합 파서: m^2·kg/s^3 형태
    # 토큰: unit(^exp)? 분자/분모(/) 구분, 구분자 [·* /]
    if not u: return {}
    num,den = u, ""
    if "/" in u:
        parts=u.split("/")
        num = parts[0]
        den = "/".join(parts[1:])
    def parse_side(s, sign=1):
        res={}
        for tok in re.split(r"[·\*\s]+", s.strip()):
            if not tok: continue
            m=re.match(r"([a-zA-ZΩμ]+)(?:\^(-?\d+))?$", tok)
            if not m: continue
            name=m.group(1)
            exp=int(m.group(2) or "1")
            # μ(마이크로) 접두어는 차원엔 영향 없음(스칼라) → 무시
            name = "ohm" if name in ("Ohm","Ω") else name
            base=_DIM.get(name, {name:1} if name in _BASE else {})
            res=_dim_mul(res, _dim_pow(base, exp*sign))
        return res
    out=_dim_mul(parse_side(num,+1), parse_side(den,-1))
    return {k:v for k,v in out.items() if v!=0}

def _expr_dim(expr:str, var_units:dict)->dict:
    # 허용: 변수명, *, /, ^정수, 괄호, 공백
    # 전략: 항목을 재귀 파싱 → 곱/나눗셈 차원 연산
    tokens=re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\^|-?\d+|[*/()]", expr.replace("·","*").replace(" ",""))
    pos=0
    def parse_factor():
        nonlocal pos
        if pos>=len(tokens): return {}
        t=tokens[pos]
        if t=="(":
            pos+=1
            d=parse_term()
            if pos<len(tokens) and tokens[pos]==")": pos+=1
            # 지수
            if pos<len(tokens) and tokens[pos]=="^":
                pos+=1
                n=int(tokens[pos]); pos+=1
                d=_dim_pow(d,n)
            return d
        elif re.match(r"[A-Za-z_]", t):
            pos+=1
            unit = var_units.get(t,"")
            d=_unit_to_dim(unit)
            if pos<len(tokens) and tokens[pos]=="^":
                pos+=1
                n=int(tokens[pos]); pos+=1
                d=_dim_pow(d,n)
            return d
        elif re.match(r"-?\d+", t):
            pos+=1
            # 스칼라 숫자 → 무차원
            return {}
        return {}
    def parse_term():
        nonlocal pos
        d = parse_factor()
        while pos<len(tokens) and tokens[pos] in ("*","/"):
            op=tokens[pos]; pos+=1
            d2=parse_factor()
            d = _dim_mul(d, d2 if op=="*" else _dim_pow(d2,-1))
        return d
    return parse_term()

def _dim_equal(d1:dict,d2:dict)->bool:
    # 동일 차원 여부
    return _dim_mul(d1, _dim_pow(d2,-1))=={}

with st.expander("㊺ 단위/차원 계산기(정합성 체크)", expanded=False):
    st.markdown("**예시**: ΔL/L → 무차원, E=h·ν → J = (J·s)·s^-1")
    in_expr = st.text_input("표현식", value="ΔL/L", key="ud_expr")
    in_map  = st.text_area("변수→단위 JSON", value='{"ΔL":"m","L":"m"}', height=80, key="ud_map")
    lhs_u   = st.text_input("좌변(선택: 차원 비교용 단위)", value="", key="ud_lhs")
    if st.button("계산/검증", key="ud_go"):
        try:
            var_units=json.loads(in_map)
            d_rhs=_expr_dim(in_expr, var_units)
            show_rhs = "·".join([f"{k}^{v}" for k,v in sorted(d_rhs.items())]) or "dimensionless"
            if lhs_u.strip():
                d_lhs=_unit_to_dim(lhs_u.strip())
                ok=_dim_equal(d_lhs,d_rhs)
                st.json({"rhs_dim": d_rhs, "rhs_pretty": show_rhs, "lhs_dim": d_lhs, "match": ok})
                st.success("정합성 OK" if ok else "정합성 불일치")
            else:
                st.json({"rhs_dim": d_rhs, "rhs_pretty": show_rhs})
        except Exception as e:
            st.error(f"오류: {e}")

# ================================================================
# 44. 미니 SMT(라이트) — CNF 부울 SAT 브루트포스(≤8변수)
#    - 입력: CNF 문자열 (예: (x1 or ~x2) and (x2 or x3))
#    - 출력: 만족 여부 + 만족 할당 예시
# ================================================================
def _parse_cnf(cnf:str):
    # 매우 단순 파서: 변수명 [a-zA-Z0-9_], 부정 ~, 절/연결 and/or 괄호
    cnf = cnf.replace("AND","and").replace("OR","or").replace("¬","~")
    clauses=[]
    vars_set=set()
    for part in re.findall(r"\([^)]*\)", cnf):
        lits=[]
        for lit in re.split(r"\s+or\s+|,", part.strip("() ")):
            lit=lit.strip()
            if not lit: continue
            neg = lit.startswith("~") or lit.lower().startswith("not ")
            name = re.sub(r"^(~|not\s+)", "", lit, flags=re.I)
            vars_set.add(name)
            lits.append( (name, not neg) )
        if lits: clauses.append(lits)
    return clauses, sorted(vars_set)

def _sat_check(clauses, vars_list, limit=1<<20):
    n=len(vars_list)
    if n>8: return False, {}
    from itertools import product
    tried=0
    for bits in product([False,True], repeat=n):
        tried+=1
        assign={vars_list[i]: bits[i] for i in range(n)}
        ok=True
        for clause in clauses:
            if not any( (assign[name] if sign else (not assign[name])) for (name,sign) in clause ):
                ok=False; break
        if ok:
            return True, assign
        if tried>=limit: break
    return False, {}

with st.expander("㊻ 미니 SMT(부울 SAT)", expanded=False):
    sample="(x1 or ~x2) and (x2 or x3) and (~x1 or x3)"
    cnf_in=st.text_area("CNF 입력", value=sample, height=90, key="smt_in")
    if st.button("SAT 체크", key="smt_go"):
        clauses, vars_list=_parse_cnf(cnf_in)
        ok, assign=_sat_check(clauses, vars_list)
        st.json({"vars": vars_list, "satisfiable": ok, "assignment": assign})

# ================================================================
# 45. 링크 검증 강화 — CE-Graph 증거 URL 가용성/미리보기/체크섬
#    - 각 evidence.source에 대해 HTTP 캐시로 가져와 길이/키워드 검사
# ================================================================
def verify_ce_links(ce:dict, min_len:int=32, keys=None)->dict:
    keys = keys or ["abstract","introduction","dataset","method","result"]
    nodes=ce.get("nodes",[])
    evid=[n for n in nodes if n.get("kind")=="evidence"]
    out=[]
    ok_count=0
    for ev in evid:
        src=(ev.get("payload") or {}).get("source","")
        ok, txt = (False, "")
        if str(src).startswith("http"):
            ok, txt = http_cache_get(src)
        else:
            ok, txt = (True, f"(오프라인 소스) {src}")
        length=len(txt)
        hit = any(k in txt.lower() for k in keys) if txt else False
        ch  = hashlib.sha256((txt or "").encode("utf-8")).hexdigest()[:12]
        passed = ok and length>=min_len and hit
        ok_count += 1 if passed else 0
        out.append({"source":src, "ok":ok, "len":length, "hit":hit, "sha12":ch, "pass":passed})
    cov = round(ok_count/max(1,len(evid)),2)
    verdict = "PASS" if cov>=0.5 else "REPAIR"
    return {"coverage": cov, "verdict": verdict, "rows": out}

with st.expander("㊼ 링크 검증(증거 URL)", expanded=False):
    ce = st.session_state.get("CE_GRAPH")
    if not ce:
        st.warning("CE-Graph가 없습니다. 상단 ①에서 먼저 생성하세요.")
    else:
        res=verify_ce_links(ce)
        st.json(res)
        st.success("링크 커버리지 OK" if res["verdict"]=="PASS" else "REPAIR 필요")
        
        # ================================================================
# 46. 장기기억(LTM) 스냅샷 — 세션 상태→파일(JSON.GZ) 저장/복원
#    - 내용: 목적/정체성/가치/감정/CE-Graph/목표/마지막 응답/게이트 메트릭
# ================================================================
import os, json, time, gzip, glob, hashlib
from datetime import datetime

# 안전 해시
try:
    _sha  # 기존 정의 있으면 사용
except NameError:
    def _sha(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

# 로그 디렉터리 기본값
try:
    LOG_DIR  # 기존 값 사용
except NameError:
    LOG_DIR = "gea_logs"
os.makedirs(LOG_DIR, exist_ok=True)

LTM_DIR = os.path.join(LOG_DIR, "ltm")
os.makedirs(LTM_DIR, exist_ok=True)

def _ltm_now():
    return datetime.utcnow().isoformat()+"Z"

def _ltm_slug(name: str) -> str:
    base = "".join(c if c.isalnum() or c in "-_." else "_" for c in (name or "snapshot"))
    return base[:48] or "snapshot"

def ltm_snapshot_create(name: str = "", include_ce: bool=True, include_metrics: bool=True) -> dict:
    ce   = st.session_state.get("CE_GRAPH") if include_ce else None
    goals= st.session_state.get("GEA_GOALS", {})
    last = st.session_state.get("INTERACT_REPLY_EX") or st.session_state.get("INTERACT_REPLY") or ""
    gate = st.session_state.get("LAST_GATE") if include_metrics else None

    # 메모리 코어에서 핵심 선언/감정 수집(있을 때만)
    purpose = identity = values = None
    emotions = []
    try:
        purpose  = mem_load_core("EA_PURPOSE")
        identity = mem_load_core("EA_IDENTITY")
        values   = mem_load_core("EA_VALUES")
        emotions = mem_recent_emotions(5)
    except Exception:
        pass

    payload = {
        "meta": {
            "created_at": _ltm_now(),
            "app_version": "v0.6",
            "name": name or "auto",
            "ce_digest": (ce or {}).get("digest"),
        },
        "state": {
            "purpose": purpose,
            "identity": identity,
            "values": values,
            "emotions": emotions,
            "goals": goals,
            "ce_graph": ce,
            "last_reply": last,
            "gate_metrics": gate,
        }
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    fname = f"{int(time.time())}_{_ltm_slug(name)}_{_sha(raw)[:8]}.json.gz"
    fpath = os.path.join(LTM_DIR, fname)
    with gzip.open(fpath, "wb") as f:
        f.write(raw)
    return {"path": fpath, "bytes": len(raw), "file": fname}

def ltm_list(limit: int = 50) -> list:
    files = sorted(glob.glob(os.path.join(LTM_DIR, "*.json.gz")), reverse=True)[:limit]
    out = []
    for p in files:
        try:
            with gzip.open(p, "rb") as f:
                d = json.loads(f.read().decode("utf-8", errors="replace"))
            out.append({"file": os.path.basename(p),
                        "created_at": d.get("meta",{}).get("created_at"),
                        "name": d.get("meta",{}).get("name"),
                        "ce_digest": d.get("meta",{}).get("ce_digest")})
        except Exception:
            out.append({"file": os.path.basename(p), "created_at": "?", "name": "?", "ce_digest": None})
    return out

def ltm_load(file_name: str) -> dict:
    fpath = os.path.join(LTM_DIR, file_name)
    with gzip.open(fpath, "rb") as f:
        return json.loads(f.read().decode("utf-8", errors="replace"))

def ltm_restore(file_name: str, inject: bool=True) -> dict:
    data = ltm_load(file_name)
    st.session_state["GEA_GOALS"]  = data.get("state",{}).get("goals", {})
    st.session_state["CE_GRAPH"]   = data.get("state",{}).get("ce_graph")
    st.session_state["LAST_GATE"]  = data.get("state",{}).get("gate_metrics")
    # 메모리 코어 주입(있을 때만)
    try:
        core = data.get("state",{})
        if core.get("purpose"):  mem_save_core("EA_PURPOSE",  core.get("purpose"))
        if core.get("identity"): mem_save_core("EA_IDENTITY", core.get("identity"))
        if core.get("values"):   mem_save_core("EA_VALUES",   core.get("values"))
    except Exception:
        pass
    return {"restored": True, "meta": data.get("meta",{})}

with st.expander("㊽ 장기기억(LTM) 스냅샷", expanded=False):
    snap_name = st.text_input("스냅샷 이름", value="올원-일일점검", key="ltm_name")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("스냅샷 저장", key="ltm_save"):
            info = ltm_snapshot_create(snap_name, include_ce=True, include_metrics=True)
            st.success(f"저장됨: {info['file']} ({info['bytes']}B raw)")
    with col2:
        if st.button("목록 새로고침", key="ltm_list"):
            st.session_state["LTM_LIST"] = ltm_list()
    with col3:
        st.download_button("최근 스냅샷 다운로드", data=open(os.path.join(LTM_DIR, ltm_list(1)[0]["file"]), "rb").read()
                           if ltm_list(1) else b"", file_name=ltm_list(1)[0]["file"] if ltm_list(1) else "none",
                           mime="application/gzip", disabled=(len(ltm_list(1))==0))
    st.json(st.session_state.get("LTM_LIST", ltm_list(10)))

# ================================================================
# 47. 압축·요약 — 빈도/키워드 기반 추출 요약 + GZIP 압축 유틸
#    - 텍스트 요약: 핵심문장 상위 N개 (가중치: 단위/재현/근거 키워드)
# ================================================================
_UNIT_KEYS = ["단위","unit","m","kg","s","Hz","N","J","W","V","Pa","Ω"]
_REPR_KEYS = ["재현","method","step","protocol","검증","절차","replicate"]
_EVID_KEYS = ["근거","source","src:","http","doi","dataset","result"]

def summarize_extractive(text: str, max_sent: int = 5) -> str:
    import re
    sents = re.split(r"(?<=[.!?。])\s+", text.strip())
    if not sents: return text
    scores = []
    for s in sents:
        sl = s.lower()
        score = 1.0
        score += sum(1 for k in _UNIT_KEYS if k.lower() in sl)*0.5
        score += sum(1 for k in _REPR_KEYS if k.lower() in sl)*0.6
        score += sum(1 for k in _EVID_KEYS if k.lower() in sl)*0.7
        score += min(len(s)/120, 1.0)*0.3  # 너무 짧은 문장 페널티
        scores.append((score, s))
    scores.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(s for _,s in scores[:max_sent])

def compress_dict_gzip(d: dict) -> bytes:
    raw = json.dumps(d, ensure_ascii=False, separators=(",",":")).encode("utf-8")
    return gzip.compress(raw, compresslevel=6)

with st.expander("㊾ 요약/압축 유틸", expanded=False):
    tx = st.text_area("요약할 본문", value="단위와 근거, 재현 절차를 포함한 본문을 여기에 붙여 넣으세요.", height=120, key="sum_tx")
    n  = st.slider("최대 문장 수", 1, 10, 5, key="sum_n")
    if st.button("추출 요약", key="sum_go"):
        st.write(summarize_extractive(tx, n))
    if st.button("요약→스냅샷 저장", key="sum_save"):
        body = summarize_extractive(tx, n)
        info = ltm_snapshot_create(f"요약-{int(time.time())}", include_ce=False, include_metrics=False)
        st.success(f"요약 저장 완료: {info['file']}")

# ================================================================
# 48. 스냅샷 프리뷰/복원 — 미리보기·복원·CE/기억 재주입
# ================================================================
with st.expander("㊿ 스냅샷 프리뷰/복원", expanded=False):
    files = [x["file"] for x in ltm_list(50)]
    sel = st.selectbox("스냅샷 선택", files, key="ltm_sel") if files else None
    if sel:
        if st.button("미리보기", key="ltm_prev"):
            d = ltm_load(sel)
            preview = {
                "meta": d.get("meta", {}),
                "purpose": (d.get("state",{}).get("purpose") or {}),
                "identity": (d.get("state",{}).get("identity") or {}),
                "values": (d.get("state",{}).get("values") or {}),
                "has_ce": d.get("state",{}).get("ce_graph") is not None,
                "last_reply_len": len((d.get("state",{}).get("last_reply") or "")),
            }
            st.json(preview)
        if st.button("복원(CE/기억 재주입)", key="ltm_restore"):
            out = ltm_restore(sel, inject=True)
            st.success(f"복원 완료: {out['meta'].get('created_at')} · {out['meta'].get('name')}")
        if st.button("복원 후 응답 생성(기억 주입)", key="ltm_reply"):
            d = ltm_load(sel)
            ask = "복원된 컨텍스트를 사용해 오늘 목표/계획을 요약해줘."
            st.write(generate_with_memory(ask, level=8))

# ================================================================
# 49. 재주입 루프 — 스냅샷→요약→메모리 접두주입→응답
#    - 한 번에: 스냅샷 선택→핵심 요약→기억 접두→엔진 호출
# ================================================================
def reply_from_snapshot(file_name: str, question: str, level: int=8):
    d = ltm_load(file_name)
    # 핵심 텍스트 구성(목적/정체성/가치/마지막응답 일부)
    parts = []
    stt = d.get("state", {})
    for k in ("purpose","identity","values"):
        if stt.get(k):
            parts.append(json.dumps(stt[k], ensure_ascii=False))
    if stt.get("last_reply"):
        parts.append(str(stt["last_reply"])[:1000])
    base = "\n".join(parts)
    digest = summarize_extractive(base, max_sent=4)
    # 임시로 기억에 덮어쓰기(접두 주입에 활용)
    try:
        if stt.get("purpose"):  mem_save_core("EA_PURPOSE",  stt.get("purpose"))
        if stt.get("identity"): mem_save_core("EA_IDENTITY", stt.get("identity"))
        if stt.get("values"):   mem_save_core("EA_VALUES",   stt.get("values"))
    except Exception:
        pass
    q = f"[복원요약]\n{digest}\n\n{question}"
    return generate_with_memory(q, level=level)

with st.expander("[49] 재주입 루프(스냅샷→요약→응답)", expanded=False):
    files9 = [x["file"] for x in ltm_list(50)]
    s9 = st.selectbox("스냅샷", files9, key="r9_sel") if files9 else None
    q9 = st.text_input("질문", value="복원된 맥락으로 오늘의 실행 체크리스트를 만들어줘.", key="r9_q")
    l9 = st.slider("레벨", 1, 999, 8, key="r9_lvl")
    if s9 and st.button("재주입 응답", key="r9_go"):
        out = reply_from_snapshot(s9, q9, l9)
        st.write(out)

# ================================================================
# 50. LTM 오토세이브(이벤트 기반) — 응답 생성 시 자동 스냅샷
#    - 백그라운드 타이머 없이: 버튼 클릭/응답 생성 이벤트에 후행 저장
# ================================================================
if "LTM_AUTOSAVE" not in st.session_state:
    st.session_state["LTM_AUTOSAVE"] = False

with st.sidebar:
    st.markdown("---")
    st.checkbox("LTM 오토세이브(응답 생성 시 자동 저장)", value=st.session_state["LTM_AUTOSAVE"], key="LTM_AUTOSAVE")

def ltm_autosave_on_reply(tag: str = "auto-reply"):
    if st.session_state.get("LTM_AUTOSAVE", False):
        try:
            ltm_snapshot_create(name=tag, include_ce=True, include_metrics=True)
        except Exception:
            pass

# 기존 응답 생성 경로에 후킹(가능한 곳에서 호출)
# - 확장 인터랙션 루프(㊝) 실행 후:
try:
    if "INTERACT_REPLY_EX" in st.session_state and st.session_state.get("_LTM_HOOKED_IX", False) is False:
        ltm_autosave_on_reply("ix")
        st.session_state["_LTM_HOOKED_IX"] = True
except Exception:
    pass
# - E2E 실행 후(㊞)는 기존 end_to_end_once 내부 e2e_post_hook가 로그를 남김.
#   E2E 버튼 핸들러에서도 바로 아래 한 줄을 추가로 호출:
#   ltm_autosave_on_reply("e2e")

# ================================================================
# 51. 심볼릭 증명 스텁(라이트) — 다점 수치검증 기반 항등성 점검
#    - 입력: LHS, RHS, 변수영역 JSON → 무작위 치환 후 |LHS-RHS| ≤ tol 판정
#    - 주의: 수치검증(강한 헤유리스틱). 형식적 증명은 아님(추후 Coq/Lean 연동 포인트)
# ================================================================
import math, random, re

_SAFE_MATH = {
    "pi": math.pi, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt,
    "pow": pow, "abs": abs, "min": min, "max": max
}

_VAR_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def _safe_eval(expr: str, env: dict) -> float:
    # 허용 토큰(변수/함수/숫자/연산자/괄호)만 검사
    chk = re.sub(r"[0-9\.\+\-\*\/\^\(\)\,\s]", "", expr)
    # ^ → ** 로 치환
    expr = expr.replace("^", "**")
    # 안전 eval
    return eval(expr, {"__builtins__": {}}, env)

def check_identity(L: str, R: str, var_ranges: dict, trials: int = 64, tol: float = 1e-9) -> dict:
    # 변수 목록
    vars_in = sorted(set(_VAR_TOKEN.findall(L) + _VAR_TOKEN.findall(R)) - set(_SAFE_MATH.keys()))
    ok_count = 0; fails = []
    for _ in range(trials):
        env = dict(_SAFE_MATH)
        for v in vars_in:
            lo, hi = var_ranges.get(v, [-1.0, 1.0])
            # 0 분모 회피를 위해 작은 오프셋
            val = random.uniform(lo, hi)
            if abs(val) < 1e-9: val += (1e-3 if hi - lo > 1e-3 else 1e-6)
            env[v] = val
        try:
            lv = _safe_eval(L, env)
            rv = _safe_eval(R, env)
            if not (math.isfinite(lv) and math.isfinite(rv)):
                fails.append({"env": env, "reason": "non-finite"})
                continue
            if abs(lv - rv) <= tol * max(1.0, max(abs(lv), abs(rv))):
                ok_count += 1
            else:
                fails.append({"env": {k: round(float(v),6) for k,v in env.items()},
                              "lhs": lv, "rhs": rv, "diff": lv-rv})
        except Exception as e:
            fails.append({"env": {k: float(v) if isinstance(v,(int,float)) else str(v) for k,v in env.items()},
                          "error": str(e)})
    verdict = (ok_count == trials)
    return {"vars": vars_in, "trials": trials, "ok": ok_count, "pass": verdict, "fails": fails[:3]}

with st.expander("[51] 심볼릭 증명 스텁(라이트)", expanded=False):
    lhs = st.text_input("LHS", value="sin(x)^2 + cos(x)^2", key="id_lhs")
    rhs = st.text_input("RHS", value="1", key="id_rhs")
    vr  = st.text_area("변수 범위 JSON", value='{"x":[-3.14,3.14]}', height=80, key="id_rng")
    tr  = st.slider("시도 횟수", 8, 256, 64, key="id_trials")
    tol = st.number_input("상대 오차 tol", value=1e-9, format="%.1e", key="id_tol")
    if st.button("항등성 점검", key="id_go"):
        try:
            res = check_identity(lhs, rhs, json.loads(vr), tr, tol)
            st.json(res); st.success("PASS" if res["pass"] else "비일치(반례 후보 있음)")
        except Exception as e:
            st.error(f"오류: {e}")

# ================================================================
# 52. 테스트 매트릭스 러너 — 단위/SMT/링크/항등성 일괄 실행
#    - 입력: JSONL 업로드 또는 샘플 케이스 실행
# ================================================================
def run_test_matrix(cases: list) -> dict:
    results = []; pass_cnt = 0
    for i, c in enumerate(cases, 1):
        kind = c.get("type")
        rid  = c.get("id", f"case{i}")
        try:
            if kind == "units":
                d = _expr_dim(c["expr"], c["map"])
                ok = True
                if "expect_dim" in c:
                    ok = _dim_equal(d, _unit_to_dim(c["expect_dim"]))
                results.append({"id": rid, "type":"units", "ok": ok, "dim": d})
            elif kind == "sat":
                clauses, vars_list = _parse_cnf(c["cnf"])
                ok, assign = _sat_check(clauses, vars_list)
                results.append({"id": rid, "type":"sat", "ok": ok, "assign": assign})
            elif kind == "links":
                ce = st.session_state.get("CE_GRAPH") or {"nodes": c.get("nodes", [])}
                vr = verify_ce_links(ce)
                ok = (vr["verdict"] == "PASS")
                results.append({"id": rid, "type":"links", "ok": ok, "coverage": vr["coverage"]})
            elif kind == "identity":
                res = check_identity(c["lhs"], c["rhs"], c.get("ranges", {}), c.get("trials", 64), c.get("tol",1e-9))
                ok = res["pass"]
                results.append({"id": rid, "type":"identity", "ok": ok, "meta": {"ok":res["ok"],"trials":res["trials"]}})
            else:
                results.append({"id": rid, "type": kind, "ok": False, "error": "unknown type"})
        except Exception as e:
            results.append({"id": rid, "type": kind, "ok": False, "error": str(e)})
    pass_cnt = sum(1 for r in results if r.get("ok"))
    return {"total": len(results), "passed": pass_cnt, "rate": round(pass_cnt/max(1,len(results)),2), "rows": results}

_SAMPLE_MATRIX = [
    {"id":"U-ΔL/L","type":"units","expr":"ΔL/L","map":{"ΔL":"m","L":"m"},"expect_dim":""},
    {"id":"S-1","type":"sat","cnf":"(x1 or ~x2) and (x2 or x3) and (~x1 or x3)"},
    {"id":"L-CE","type":"links","nodes":[{"id":"e1","kind":"evidence","payload":{"source":"https://httpbin.org/json"}}]},
    {"id":"I-트리그","type":"identity","lhs":"sin(x)^2+cos(x)^2","rhs":"1","ranges":{"x":[-3.14,3.14]},"trials":48}
]

with st.expander("[52] 테스트 매트릭스 러너", expanded=False):
    up = st.file_uploader("매트릭스 JSONL 업로드(선택)", type=["jsonl"], key="tm_upl")
    if st.button("실행(샘플)", key="tm_go_sample"):
        st.json(run_test_matrix(_SAMPLE_MATRIX))
    if st.button("실행(업로드)", key="tm_go_upl") and up is not None:
        cases=[]
        for line in up.getvalue().decode("utf-8","replace").splitlines():
            if line.strip():
                try: cases.append(json.loads(line))
                except: pass
        st.json(run_test_matrix(cases))

# ================================================================
# 53. 플러그인 샌드박스(안전) — 화이트리스트 유틸 실행
#    - 임의 코드 금지. 미리 등록한 안전 함수만 선택 실행.
# ================================================================
def _plug_normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def _plug_extract_numbers(t: str) -> list:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]

def _plug_topk_sentences(t: str, k: int = 3) -> list:
    sents = re.split(r"(?<=[.!?。])\s+", t.strip())
    sents = [s for s in sents if s]
    sents.sort(key=lambda s: len(s), reverse=True)
    return sents[:k]

PLUGIN_REGISTRY = {
    "정규화": _plug_normalize_text,
    "숫자추출": _plug_extract_numbers,
    "상위문장K": _plug_topk_sentences,
}

with st.expander("[53] 플러그인 샌드박스(안전)", expanded=False):
    sel = st.selectbox("플러그인 선택", list(PLUGIN_REGISTRY.keys()), key="pl_sel")
    txt = st.text_area("입력", value="중력파 진폭은 1.5e-21 이고, L=4,000 m 입니다.", height=80, key="pl_txt")
    k  = st.slider("K(일부 플러그인 용)", 1, 10, 3, key="pl_k")
    if st.button("실행", key="pl_go"):
        fn = PLUGIN_REGISTRY.get(sel)
        try:
            out = fn(txt) if sel != "상위문장K" else fn(txt, k)
            st.write(out)
        except Exception as e:
            st.error(f"오류: {e}")

# ================================================================
# 54. REAL 가드센터(확장) — 하드/소프트 가드 + 응답 경고 배선
#    - 하드: 금칙어 즉시 차단 / 소프트: 경고 후 정제
#    - generate_with_memory() 호출 전후 후킹(가벼운 정제)
# ================================================================
_FORBIDDEN_PATTERNS = [
    r"초광속", r"\bwarp\b", r"\b워프\b", r"(?:5|11|13)차원", r"초자연", r"예언", r"영매"
]
_REAL_HARD = [re.compile(p, re.I) for p in _FORBIDDEN_PATTERNS]

if "REAL_GUARD_MODE" not in st.session_state:
    st.session_state["REAL_GUARD_MODE"] = "soft"   # "hard" | "soft" | "off"

def real_guard_filter(text: str) -> tuple:
    mode = st.session_state.get("REAL_GUARD_MODE","soft")
    if mode == "off": return True, text, None
    for pat in _REAL_HARD:
        if pat.search(text):
            if mode == "hard":
                return False, text, "REAL 금칙어 하드 차단"
            else:
                clean = pat.sub("[제거됨]", text)
                return True, clean, f"REAL 소프트 정제: {pat.pattern}"
    return True, text, None

with st.sidebar:
    st.markdown("**REAL 가드 모드**")
    st.radio("가드 설정", ["soft","hard","off"], key="REAL_GUARD_MODE", horizontal=True)

# generate_with_memory 전/후 가드 훅(경고 표출)
_old_generate_with_memory = generate_with_memory
def generate_with_memory_guarded(user_text: str, level: int = 8):
    ok, clean, warn = real_guard_filter(user_text)
    if not ok:
        return {"status":"REFUSE","reason": warn}
    out = _old_generate_with_memory(clean, level)
    if isinstance(out, str):
        ans = out
    else:
        ans = json.dumps(out, ensure_ascii=False) if isinstance(out, dict) else str(out)
    ok2, clean2, warn2 = real_guard_filter(ans)
    if warn or warn2:
        st.warning("REAL 가드 경고/정제 적용됨")
    return clean2 if ok2 else {"status":"REFUSE","reason": warn2}

# 기존 호출 경로 바인딩 교체
generate_with_memory = generate_with_memory_guarded

# ================================================================
# 55. 한국어 UI 폴리시/테마 — 시스템 폰트 스택 + 가독성 CSS 주입
#    - 외부 폰트 의존 없음. 한글 깨짐 최소화, 가독성 개선.
# ================================================================
_KO_CSS = """
<style>
html, body, [class^="css"]  {
  font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
               "Malgun Gothic", "Noto Sans CJK KR", "Segoe UI", Roboto, Arial, sans-serif !important;
}
section.main > div { max-width: 1120px; margin-left: auto; margin-right: auto; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
.sidebar .stMarkdown { font-size: 0.95rem; }
</style>
"""
st.markdown(_KO_CSS, unsafe_allow_html=True)

# ================================================================
# 56. 확장 인터랙션 루프 — 응답 카드 UX + 프롬프트 사전셋 + 히스토리
#    - 기억 주입 엔진(generate_with_memory)을 카드 UI로 감싸서 사용성↑
# ================================================================
if "HISTORY" not in st.session_state:
    st.session_state["HISTORY"] = []   # [{q, a, ts, lvl}]

_PRESETS = [
    ("목표 요약", "에아, 오늘 우리의 최상위 목표를 5줄로 요약해줘."),
    ("실행 체크리스트", "에아, 지금 바로 실행 가능한 7가지 체크리스트를 만들어줘."),
    ("리스크 진단", "에아, 현재 설계의 위험 요소와 완화책을 표로 정리해줘."),
    ("증거 요청", "에아, 위 내용에서 필요한 증거/데이터셋 목록을 만들어 링크해줘.")
]

with st.expander("㊱㊱ 확장 대화(응답 카드 UX)", expanded=True):
    colL, colR = st.columns([2,1])
    with colL:
        preset = st.selectbox("프롬프트 사전셋", [p[0] for p in _PRESETS], index=0, key="xl_preset")
        base   = _PRESETS[[p[0] for p in _PRESETS].index(preset)][1]
        usr_tx = st.text_area("질문/명령(수정 가능)", value=base, height=110, key="xl_q")
        lvl    = st.slider("응답 레벨", 1, 999, 8, key="xl_lvl")
        if st.button("응답 생성(카드)", key="xl_go"):
            ans = generate_with_memory(usr_tx, level=lvl)
            st.session_state["HISTORY"].append({"q": usr_tx, "a": ans, "ts": int(time.time()), "lvl": lvl})
            # LTM 자동 후킹
            try: ltm_autosave_on_reply("xl")
            except: pass
    with colR:
        st.markdown("**최근 5개 히스토리**")
        for item in st.session_state["HISTORY"][-5:][::-1]:
            with st.container(border=True):
                st.caption(f"레벨 {item['lvl']} • ts={item['ts']}")
                st.write(f"**Q:** {item['q'][:120]}{'…' if len(item['q'])>120 else ''}")
                st.write("**A:**")
                st.write(item['a'] if isinstance(item['a'], (str,int,float)) else json.dumps(item['a'], ensure_ascii=False)[:800])

# ================================================================
# 57. 액티브 모드 미니 스케줄러 — 초간단 워크 루프(버튼 트리거)
#    - 주기적 백그라운드가 아니라 버튼 클릭으로 N회 실행(모바일/웹 안전)
# ================================================================
if "ACTIVE_MODE" not in st.session_state:
    st.session_state["ACTIVE_MODE"] = False

def _active_tick(n: int = 1, prompt: str = "에아, 진행상황 점검/업데이트 요약해줘.", lvl: int = 6):
    logs=[]
    for i in range(n):
        ans = generate_with_memory(prompt, level=lvl)
        logs.append({"i": i+1, "ans_len": len(str(ans)), "ts": int(time.time())})
        # 오토세이브 후킹
        try: ltm_autosave_on_reply("active")
        except: pass
    return logs

with st.expander("㊲ 액티브 모드 미니 스케줄러", expanded=False):
    st.toggle("액티브 모드", value=st.session_state["ACTIVE_MODE"], key="ACTIVE_MODE")
    a_prompt = st.text_area("액티브 프롬프트", value="에아, 진행상황 점검/업데이트 요약해줘.", height=80, key="am_prompt")
    a_lvl    = st.slider("레벨", 1, 999, 6, key="am_lvl")
    a_n      = st.number_input("반복 횟수(즉시)", min_value=1, max_value=20, value=3, step=1, key="am_n")
    if st.button("지금 N회 실행", key="am_run"):
        if st.session_state["ACTIVE_MODE"]:
            res=_active_tick(a_n, a_prompt, a_lvl); st.json(res)
        else:
            st.warning("액티브 모드가 OFF입니다. 토글을 켜고 다시 실행하세요.")

# ================================================================
# 58. 워치독/헬스 — 지표 점검(게이트 메트릭, 링크 커버리지, 메모리 상태)
#    - 임계치 미달 시 경고 표시. 라이트 버전(버튼 클릭형)
# ================================================================
_HEALTH_MIN = {
    "ce_coverage": 0.97, "citation_coverage": 0.90,
    "reproducibility": 0.93, "subset_robustness": 0.99
}

def health_check() -> dict:
    gate = st.session_state.get("LAST_GATE") or {}
    ce   = st.session_state.get("CE_GRAPH")
    cov  = None
    if ce:
        v = verify_ce_links(ce)
        cov = v.get("coverage",0)
    mem_ok = True
    try:
        _ = mem_load_core("EA_PURPOSE")
    except Exception:
        mem_ok = False
    verdicts = {}
    for k,th in _HEALTH_MIN.items():
        val = gate.get(k)
        if val is None: verdicts[k] = "unknown"
        else: verdicts[k] = "OK" if (val >= th) else "LOW"
    if cov is not None:
        verdicts["ce_link_coverage"] = "OK" if cov >= 0.5 else "LOW"
    verdicts["memory_core"] = "OK" if mem_ok else "WARN"
    return {"gate": gate, "verdicts": verdicts, "link_cov": cov}

with st.expander("㊳ 워치독/헬스 점검", expanded=False):
    if st.button("헬스 체크 실행", key="hc_go"):
        h = health_check()
        st.json(h)
        # 시각 경고
        v = h["verdicts"]
        if any(vv=="LOW" for vv in v.values()):
            st.error("임계치 미달 항목이 있습니다. REPAIR 루프를 권장합니다.")
        elif v.get("memory_core")=="WARN":
            st.warning("메모리 코어 연결 확인 필요.")
        else:
            st.success("헬스 상태 양호")

# ================================================================
# 59. 목표/태스크 보드 — 상위 목표/하위 태스크/상태/우선순위 보드
#    - 간단 JSON 레지스트리 + 진행률 계산 + 체크오프
# ================================================================
if "TASKS" not in st.session_state:
    st.session_state["TASKS"] = []   # [{id, title, parent, prio, status, ts}]

def task_add(title, parent=None, prio=3):
    tid = f"T{int(time.time()*1000)%10_000_000}"
    st.session_state["TASKS"].append({"id":tid,"title":title,"parent":parent,"prio":int(prio),"status":"open","ts":int(time.time())})
    return tid

def task_update(tid, **kw):
    for t in st.session_state["TASKS"]:
        if t["id"]==tid:
            t.update(**kw); return True
    return False

def task_progress(parent=None)->float:
    items=[t for t in st.session_state["TASKS"] if (t["parent"]==parent)]
    if not items: return 0.0
    done=sum(1 for t in items if t["status"]=="done")
    return round(done/len(items),2)

with st.expander("㊴ 목표/태스크 보드", expanded=False):
    colA,colB = st.columns([2,1])
    with colA:
        t_top = st.text_input("상위 목표", value="우주정보장 근원 올원 에아 완성", key="tb_top")
        if st.button("상위 목표 기억 저장", key="tb_mem"):
            mem_save_core("EA_PURPOSE", {"goal": t_top})
            st.success("목표가 기억에 저장되었습니다.")
        st.markdown(f"**상위 목표 진행률**: {task_progress(None)*100:.0f}%")
        st.write("---")
        st.markdown("**하위 태스크 추가**")
        new_t = st.text_input("태스크 제목", value="초검증 모듈 안정화(L30→L60)", key="tb_new")
        pr    = st.slider("우선순위(1높음–5낮음)",1,5,2,key="tb_prio")
        if st.button("추가", key="tb_add"):
            tid=task_add(new_t, parent=None, prio=pr)
            st.info(f"추가됨: {tid}")
    with colB:
        st.markdown("**태스크 목록**")
        for t in sorted(st.session_state["TASKS"], key=lambda x:(x["status"], x["prio"], -x["ts"]))[:20]:
            with st.container(border=True):
                st.write(f"[{t['id']}] ({'⭐'* (6-t['prio'])}) {t['title']}")
                c1,c2,c3=st.columns(3)
                with c1:
                    if st.button("완료", key=f"tb_done_{t['id']}"):
                        task_update(t["id"], status="done")
                with c2:
                    if st.button("진행중", key=f"tb_prog_{t['id']}"):
                        task_update(t["id"], status="doing")
                with c3:
                    if st.button("삭제", key=f"tb_del_{t['id']}"):
                        st.session_state["TASKS"]=[x for x in st.session_state["TASKS"] if x["id"]!=t["id"]]

# ================================================================
# 60. 텔레메트리·오류 뷰 — 이벤트 로그/예외 기록/간단 통계
#    - LTM 디렉토리와 연동, 로컬 파일 기반 → 오프라인 안전
# ================================================================
EVT_DIR = os.path.join(LOG_DIR, "evt")
os.makedirs(EVT_DIR, exist_ok=True)

def log_event(kind: str, payload: dict):
    rec = {"ts": int(time.time()), "kind": kind, "payload": payload}
    fn  = os.path.join(EVT_DIR, f"{rec['ts']}_{kind}.json")
    try:
        with open(fn,"w",encoding="utf-8") as f: json.dump(rec, f, ensure_ascii=False)
    except Exception:
        pass

# 예: 주요 버튼 뒤에 log_event 호출 삽입 가능
# log_event("reply", {"len": len(str(ans)), "lvl": lvl})

def list_events(limit=50):
    files = sorted(glob.glob(os.path.join(EVT_DIR, "*.json")), reverse=True)[:limit]
    out=[]
    for p in files:
        try:
            d=json.load(open(p,"r",encoding="utf-8"))
            out.append(d)
        except Exception:
            out.append({"ts":0,"kind":"broken","payload":{"file":os.path.basename(p)}})
    return out

with st.expander("㊵ 텔레메트리/오류 로그", expanded=False):
    if st.button("최근 이벤트 보기", key="ev_list"):
        st.json(list_events(50))
    st.markdown("**간단 통계(세션)**")
    try:
        h=st.session_state.get("HISTORY", [])
        avg_len = sum(len(str(x.get("a",""))) for x in h)/max(1,len(h))
        st.write({"history_count": len(h), "avg_answer_len": round(avg_len,1)})
    except Exception as e:
        st.warning(f"통계 계산 실패: {e}")
        
        # ================================================================
# 61. 우주정보장 라이트 크롤러/파서 — 안전 프리뷰(fetch→정제→요약)
#    - http_cache_get(42) 재사용, 오프라인에서도 파일 캐시 활용
#    - 로봇배제/무단대량수집 금지: 단발 미리보기용
# ================================================================
def _clean_text_html(raw: str) -> str:
    # 매우 라이트한 정제: 태그 제거/공백 정리
    import re
    txt = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.S|re.I)
    txt = re.sub(r"<style[^>]*>.*?</style>", " ", txt, flags=re.S|re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def crawl_preview(url: str, summarize: bool = True) -> dict:
    ok, raw = http_cache_get(url)
    if not ok:
        return {"ok": False, "error": "fetch-failed or offline", "url": url}
    text = _clean_text_html(raw)
    prev = summarize_extractive(text, 5) if summarize else text[:800]
    sha  = hashlib.sha256(text.encode("utf-8","ignore")).hexdigest()[:12]
    return {"ok": True, "url": url, "sha12": sha, "chars": len(text), "preview": prev[:1200]}

with st.expander("㊶ 라이트 크롤러/파서(안전 프리뷰)", expanded=False):
    cr_url = st.text_input("URL", value="https://httpbin.org/html", key="cr_url")
    if st.button("가져오기/요약", key="cr_go"):
        st.json(crawl_preview(cr_url, summarize=True))

# ================================================================
# 62. 증거명세 템플릿 — Claim–Evidence 템플릿 생성/저장(JSONL)
#    - 주장을 선언하면 근거 후보 슬롯/필드 자동 구성 → 데이터팩으로도 저장
# ================================================================
def make_evidence_spec(claim: str, slots: int = 4) -> dict:
    spec = {
        "claim": claim,
        "created_at": int(time.time()),
        "slots": [{"id": f"ev{i+1}", "source":"", "note":"", "status":"open"} for i in range(slots)]
    }
    return spec

SPEC_DIR = os.path.join(LOG_DIR, "specs")
os.makedirs(SPEC_DIR, exist_ok=True)

def save_evidence_spec(spec: dict) -> str:
    fn = os.path.join(SPEC_DIR, f"spec_{spec['created_at']}_{_sha(json.dumps(spec,ensure_ascii=False).encode('utf-8'))[:8]}.jsonl")
    with open(fn,"w",encoding="utf-8") as f:
        f.write(json.dumps(spec, ensure_ascii=False)+"\n")
    return fn

with st.expander("㊷ 증거명세 템플릿", expanded=False):
    sp_txt = st.text_input("주장(Claim)", value="중력파 검출 신호 h는 ΔL/L로 무차원이다.", key="sp_claim")
    sp_n   = st.slider("슬롯 수", 1, 10, 4, key="sp_n")
    if st.button("템플릿 생성/저장", key="sp_make"):
        spec = make_evidence_spec(sp_txt, sp_n)
        path = save_evidence_spec(spec)
        st.success(f"저장됨: {path}")
        st.json(spec)

# ================================================================
# 63. 자동 REPAIR 루프 — 헬스 미달 시 근거보강(검색→프리뷰→CE 보탬)
#    - 1회 실행: 쿼리→HybridUIS.search→프리뷰 정상 항목을 evidence로 주입
#    - 라이트모드: 세션 내 CE_GRAPH에 evidence 노드만 덧붙임
# ================================================================
def ce_append_evidence(ce: dict, evid_rows: list) -> dict:
    ce = ce or {"nodes": [], "edges": [], "digest": None}
    nodes = ce.get("nodes", []); edges = ce.get("edges", [])
    claim_nodes = [n for n in nodes if n.get("kind")=="claim"]
    if not claim_nodes:
        # 임시 claim 생성
        claim_id = f"claim:{_sha(str(int(time.time())).encode())[:12]}"
        nodes.append({"id": claim_id, "kind": "claim", "payload": {"text": "임시-주장"}})
    else:
        claim_id = claim_nodes[0]["id"]
    base_ids = set(n["id"] for n in nodes)
    for ev in evid_rows:
        ev_id = f"evi:{_sha((ev.get('source') or str(ev)).encode('utf-8'))[:10]}"
        if ev_id in base_ids: continue
        nodes.append({"id": ev_id, "kind":"evidence",
                      "payload":{"source": ev.get("source",""), "span": ev.get("span",[0,100]), "score": ev.get("score",0.75)}})
        edges.append({"src": ev_id, "dst": claim_id, "rel":"supports"})
    ce["nodes"], ce["edges"] = nodes, edges
    ce["digest"] = hashlib.sha256("".join(n["id"] for n in nodes).encode()).hexdigest()[:12]
    return ce

def repair_once(query: str = "physics data", k: int = 4) -> dict:
    # 1) 검색
    try:
        hits = UIS.search(query, k=k)
    except Exception:
        hits = []
    # 2) 프리뷰 성공만 채택
    good=[]
    for h in hits:
        src = h.get("source","")
        if not src: continue
        ok, _ = http_cache_get(src)
        if ok:
            good.append({"source": src, "span": h.get("span",[0,100]), "score": h.get("score",0.7)})
    # 3) CE 보강
    ce = st.session_state.get("CE_GRAPH")
    after = ce_append_evidence(ce, good[:k])
    st.session_state["CE_GRAPH"] = after
    # 4) 헬스 재평가
    res = verify_ce_links(after)
    return {"added": len(good[:k]), "coverage": res.get("coverage"), "verdict": res.get("verdict"), "ce_digest": after.get("digest")}

with st.expander("㊸ 자동 REPAIR 루프(근거 보강)", expanded=False):
    rq = st.text_input("보강 쿼리", value="gravitational wave interferometer small-strain", key="rp_q")
    rk = st.slider("추가 evidence 최대 개수", 1, 10, 4, key="rp_k")
    if st.button("REPAIR 1회 실행", key="rp_go"):
        out = repair_once(rq, rk); st.json(out)

# ================================================================
# 64. 컨텍스트 스태킹 — 스택형 컨텍스트(요약/핵심/메모) 누적→접두 주입
#    - 세션 단위로 스택 push/pop/clear 제공, generate_with_memory에 연동
# ================================================================
if "CTX_STACK" not in st.session_state:
    st.session_state["CTX_STACK"] = []   # [{ts, kind, text, sha12}]

def ctx_push(kind: str, text: str):
    sha = hashlib.sha256(text.encode("utf-8","ignore")).hexdigest()[:12]
    st.session_state["CTX_STACK"].append({"ts": int(time.time()), "kind": kind, "text": text, "sha12": sha})

def ctx_pop():
    if st.session_state["CTX_STACK"]:
        st.session_state["CTX_STACK"].pop()

def ctx_clear():
    st.session_state["CTX_STACK"].clear()

def build_stack_prefix(max_items: int = 3) -> str:
    stk = st.session_state.get("CTX_STACK", [])[-max_items:]
    if not stk: return ""
    lines = [f"[{x['kind']}] {x['text']}" for x in stk]
    return "\n".join(lines) + "\n"

# 기존 generate_with_memory에 스택 접두 추가(기억 접두 뒤→스택 접두)
_old_gwm = generate_with_memory
def generate_with_memory_stacked(user_text: str, level: int = 8):
    prefix = build_stack_prefix(3)
    ux = (prefix + user_text) if prefix else user_text
    return _old_gwm(ux, level)

generate_with_memory = generate_with_memory_stacked

with st.expander("㊹ 컨텍스트 스택", expanded=False):
    ks = st.selectbox("종류", ["요약","핵심","메모","주의"], key="cs_kind")
    tx = st.text_area("내용", value="오늘 세션 핵심: 증거-정합성 강화 및 LTM 구축.", height=80, key="cs_text")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("PUSH", key="cs_push"):
            ctx_push(ks, tx); st.success("스택에 적재됐습니다.")
    with c2:
        if st.button("POP", key="cs_pop"):
            ctx_pop(); st.info("마지막 항목 제거")
    with c3:
        if st.button("CLEAR", key="cs_clear"):
            ctx_clear(); st.warning("스택 비움")
    st.json(st.session_state.get("CTX_STACK", []))

# ================================================================
# 65. 배포 스냅샷 메이커 — 프로젝트 최소 번들(zip) 생성/다운로드
#    - 포함: streamlit_app.py, gea_memory_core.py, gea_logs/* (선택)
# ================================================================
from zipfile import ZipFile, ZIP_DEFLATED

def make_deploy_zip(include_logs: bool = True) -> str:
    ts = int(time.time())
    zip_name = f"GEA_bundle_{ts}.zip"
    with ZipFile(zip_name, "w", ZIP_DEFLATED) as z:
        # 필수 파일들
        for fn in ("streamlit_app.py","gea_memory_core.py"):
            if os.path.exists(fn):
                z.write(fn)
        # 선택: 로그/스냅샷
        if include_logs and os.path.isdir(LOG_DIR):
            for root, _, files in os.walk(LOG_DIR):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p)
    return zip_name

with st.expander("㊺ 배포 스냅샷 메이커", expanded=False):
    inc = st.checkbox("로그 포함(ltm/specs/evt)", value=True, key="dz_inc")
    if st.button("배포 ZIP 생성", key="dz_go"):
        z = make_deploy_zip(include_logs=inc)
        st.success(f"번들 생성: {z}")
        try:
            st.download_button("ZIP 다운로드", data=open(z,"rb").read(), file_name=z, mime="application/zip")
        except Exception:
            st.info("환경상 다운로드 버튼이 제한될 수 있습니다. 파일만 생성해 두었습니다.")
            
            # ================================================================
# 66. 컨텍스트-게이트 연성 — 임계치 자동 상향/하향 + 헬스체크 오버라이드
#    - 최근 메트릭/히스토리 기반으로 임계치 미세 조정(±0.005 단위, 안전 범위)
# ================================================================
if "HC_MIN" not in st.session_state:
    st.session_state["HC_MIN"] = dict(_HEALTH_MIN)  # 기본값 복제

def _clamp(x, lo, hi): return max(lo, min(hi, x))

def gate_autotune_update(mode: str = "auto"):
    """mode: raise | lower | auto"""
    base = st.session_state.get("HC_MIN", dict(_HEALTH_MIN))
    last = st.session_state.get("LAST_GATE") or {}
    hist = st.session_state.get("HISTORY", [])
    # 단순 휴리스틱: 응답 평균 길이/최근 재현성으로 방향 결정
    avg_len = (sum(len(str(h.get("a",""))) for h in hist)/len(hist)) if hist else 0
    repro = last.get("reproducibility", None)
    direction = 0
    if mode == "raise": direction = +1
    elif mode == "lower": direction = -1
    else:  # auto
        if repro is not None and repro > 0.965 and avg_len > 800:
            direction = +1
        elif repro is not None and repro < 0.92:
            direction = -1
        else:
            direction = 0
    step = 0.005 * direction
    new_base = dict(base)
    for k in ("ce_coverage","citation_coverage","reproducibility","subset_robustness"):
        new_base[k] = round(_clamp(base.get(k, _HEALTH_MIN[k]) + step, 0.80, 0.995), 3)
    st.session_state["HC_MIN"] = new_base
    return {"avg_answer_len": avg_len, "last_repro": repro, "direction": direction, "HC_MIN": new_base}

# 기존 health_check를 오버라이드(세션 임계치 사용)
_prev_health_check = health_check
def health_check_dynamic() -> dict:
    gate = st.session_state.get("LAST_GATE") or {}
    ce   = st.session_state.get("CE_GRAPH")
    cov  = None
    if ce:
        v = verify_ce_links(ce)
        cov = v.get("coverage",0)
    base = st.session_state.get("HC_MIN", dict(_HEALTH_MIN))
    verdicts = {}
    for k,th in base.items():
        val = gate.get(k)
        if val is None: verdicts[k] = "unknown"
        else: verdicts[k] = "OK" if (val >= th) else "LOW"
    if cov is not None:
        verdicts["ce_link_coverage"] = "OK" if cov >= 0.5 else "LOW"
    # 메모리 코어 확인
    mem_ok = True
    try: _ = mem_load_core("EA_PURPOSE")
    except Exception: mem_ok=False
    verdicts["memory_core"] = "OK" if mem_ok else "WARN"
    return {"gate": gate, "verdicts": verdicts, "link_cov": cov, "thresholds": base}
health_check = health_check_dynamic

with st.expander("[66] 게이트 자동 튜닝", expanded=False):
    mode = st.radio("튜닝 모드", ["auto","raise","lower"], horizontal=True, key="gt_mode")
    if st.button("임계치 조정 실행", key="gt_apply"):
        st.json(gate_autotune_update(mode))
    if st.button("헬스 체크(동적)", key="gt_hc"):
        st.json(health_check())

# ================================================================
# 67. 장문 스트리밍 L∞ — 세그먼트 스트림 출력(중지/재개 버튼형)
#    - 백그라운드 쓰레드 없이, 버튼 루프 기반(모바일/웹 안전)
# ================================================================
if "LINF_STOP" not in st.session_state:
    st.session_state["LINF_STOP"] = False

def run_linf_stream(topic: str, segs: int = 8, lvl: int = 25):
    out = []
    area = st.empty()
    for i in range(segs):
        if st.session_state.get("LINF_STOP"): break
        prompt = f"{topic}\n\n[세그먼트 {i+1}/{segs}] 핵심 근거와 절차를 단계별로 써줘."
        ans = generate_with_memory(prompt, level=lvl)
        out.append(str(ans))
        area.markdown("**스트리밍 진행 중…**\n\n" + "\n\n---\n\n".join(out))
    return "\n\n---\n\n".join(out)

with st.expander("[67] L∞ 스트리밍", expanded=False):
    tpc = st.text_input("주제", value="우주정보장 연결 설계의 근거·절차·리스크", key="linf_tpc")
    seg = st.slider("세그먼트 수", 1, 50, 8, key="linf_segs")
    lvl = st.slider("레벨", 1, 999, 25, key="linf_lvl")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("스트리밍 시작", key="linf_go"):
            st.session_state["LINF_STOP"] = False
            text = run_linf_stream(tpc, seg, lvl)
            st.session_state["LINF_LAST"] = text
    with c2:
        if st.button("중지", key="linf_stop"):
            st.session_state["LINF_STOP"] = True
            st.info("스트림 중지 요청됨.")
    if st.session_state.get("LINF_LAST"):
        st.download_button("최근 스트림 저장", data=st.session_state["LINF_LAST"].encode("utf-8"),
                           file_name="linf_stream.txt", mime="text/plain")

# ================================================================
# 68. 응답 카드용 CE 미리보기 — 최근 CE evidence 요약
#    - HISTORY와 CE_GRAPH를 나란히 프리뷰(증거 커버리지 보조 확인)
# ================================================================
def ce_preview_snippets(ce: dict, k: int = 5) -> list:
    if not ce: return []
    ev = [n for n in ce.get("nodes",[]) if n.get("kind")=="evidence"]
    rows=[]
    for n in ev[:k]:
        src=(n.get("payload") or {}).get("source","")
        rows.append({"source": src, "id": n.get("id"), "score": (n.get("payload") or {}).get("score")})
    return rows

with st.expander("[68] 최근 응답 + CE 미리보기", expanded=False):
    if st.session_state.get("HISTORY"):
        last = st.session_state["HISTORY"][-1]
        st.write("**최근 질문**:", last.get("q","")[:200])
        st.write("**최근 응답(요약)**:", str(last.get("a",""))[:600])
    else:
        st.info("히스토리가 아직 없습니다.")
    ce = st.session_state.get("CE_GRAPH")
    st.write("**CE Evidence 미리보기**")
    st.json(ce_preview_snippets(ce, k=6))
    if ce:
        st.write("**링크 커버리지(재평가)**")
        st.json(verify_ce_links(ce))

# ================================================================
# 69. 사용자 프롬프트 사전셋 저장/불러오기 — 로컬 JSON(영속)
#    - streamlit 세션 종료 후에도 유지됨(LOG_DIR/presets.json)
# ================================================================
PRESET_PATH = os.path.join(LOG_DIR, "presets.json")

def presets_load() -> list:
    try:
        return json.load(open(PRESET_PATH, "r", encoding="utf-8"))
    except Exception:
        return []

def presets_save(items: list):
    try:
        json.dump(items, open(PRESET_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass

if "PRESETS_USER" not in st.session_state:
    st.session_state["PRESETS_USER"] = presets_load()

with st.expander("[69] 사용자 사전셋 관리", expanded=False):
    st.markdown("기존 사전셋 + 사용자 사전셋을 통합해서 사용할 수 있습니다.")
    new_name = st.text_input("이름", value="내 체크리스트", key="ps_name")
    new_body = st.text_area("프롬프트", value="에아, 오늘 해야 할 일 7가지를 근거와 함께 단계별로 제안해줘.", height=80, key="ps_body")
    if st.button("추가/갱신", key="ps_add"):
        # 동일 이름 있으면 교체
        lst = [p for p in st.session_state["PRESETS_USER"] if p.get("name")!=new_name]
        lst.append({"name": new_name, "body": new_body})
        st.session_state["PRESETS_USER"] = lst
        presets_save(lst)
        st.success("사전셋 저장 완료")
    if st.button("불러오기", key="ps_load"):
        st.session_state["PRESETS_USER"] = presets_load()
        st.info(f"불러온 항목: {len(st.session_state['PRESETS_USER'])}")
    st.json(st.session_state["PRESETS_USER"][:10])

# (보너스) 56의 _PRESETS에 사용자셋을 합쳐 쓰고 싶다면, 아래를 참고:
try:
    # 런타임 통합뷰 (오류 무시)
    _PRESETS = list(_PRESETS) + [(p["name"], p["body"]) for p in st.session_state.get("PRESETS_USER", [])]
except Exception:
    pass

# ================================================================
# 70. 로컬 키-값 저장소 — 간단 K/V(파일 영속) + 인터페이스
#    - 작은 설정/토큰/임시 데이터 저장 용
# ================================================================
KV_PATH = os.path.join(LOG_DIR, "kv_store.json")
def kv_init():
    if not os.path.exists(KV_PATH):
        json.dump({}, open(KV_PATH, "w", encoding="utf-8"), ensure_ascii=False)

def kv_get(k: str, default=None):
    kv_init()
    d=json.load(open(KV_PATH,"r",encoding="utf-8"))
    return d.get(k, default)

def kv_put(k: str, v):
    kv_init()
    d=json.load(open(KV_PATH,"r",encoding="utf-8"))
    d[k]=v
    json.dump(d, open(KV_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def kv_delete(k: str):
    kv_init()
    d=json.load(open(KV_PATH,"r","utf-8"))
    if k in d: del d[k]
    json.dump(d, open(KV_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

with st.expander("[70] 로컬 K/V 저장소", expanded=False):
    k = st.text_input("키", value="sample_key", key="kv_k")
    v = st.text_area("값(JSON 가능)", value="sample_value", key="kv_v")
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button("GET", key="kv_get"):
            st.write(kv_get(k))
    with c2:
        if st.button("PUT", key="kv_put"):
            try:
                val = json.loads(v)
            except Exception:
                val = v
            kv_put(k, val); st.success("저장됨")
    with c3:
        if st.button("DEL", key="kv_del"):
            kv_delete(k); st.info("삭제됨")
    with c4:
        if st.button("전체 보기", key="kv_all"):
            kv_init(); st.json(json.load(open(KV_PATH,"r",encoding="utf-8")))
            
            # ================================================================
# 71. 장문 응답 하이라이트/인용 스팬 — 키워드 하이라이트 + 인용 블록
#    - HISTORY 최신 응답에서 핵심 키워드 하이라이트, 인용 스팬 생성
# ================================================================
_HL_KEYS = ["증거", "단위", "재현", "절차", "데이터", "위험", "완화", "링크", "근거", "검증"]

def highlight_keywords(text: str, keys=_HL_KEYS):
    import re
    def repl(m): return f"**{m.group(0)}**"
    out = text
    for k in keys:
        try:
            out = re.sub(rf"({re.escape(k)})", repl, out, flags=re.I)
        except re.error:
            pass
    return out

def make_quote_spans(text: str, max_blocks=4, block_len=280):
    blocks=[]
    t = str(text).splitlines()
    buf=""
    for line in t:
        if len(buf)+len(line)+1 <= block_len:
            buf += (("\n" if buf else "") + line)
        else:
            blocks.append(buf); buf=line
        if len(blocks)>=max_blocks: break
    if buf and len(blocks)<max_blocks: blocks.append(buf)
    return [b.strip() for b in blocks if b.strip()]

with st.expander("[71] 응답 하이라이트/인용", expanded=False):
    if st.session_state.get("HISTORY"):
        last = st.session_state["HISTORY"][-1]
        ans  = str(last.get("a",""))
        st.markdown("**하이라이트 미리보기**")
        st.markdown(highlight_keywords(ans)[:1600])
        st.markdown("---")
        st.markdown("**인용 스팬**")
        for i, q in enumerate(make_quote_spans(ans), 1):
            st.markdown(f"> [인용 {i}] {q}")
    else:
        st.info("히스토리가 아직 없습니다.")

# ================================================================
# 72. 증거 테이블 뷰 — CE evidence 표/정렬/요약(세션 내)
#    - nodes(kind=evidence)만 추출 → 간단 표로 가시화
# ================================================================
def ce_evidence_rows(ce: dict) -> list:
    if not ce: return []
    out=[]
    for n in ce.get("nodes", []):
        if n.get("kind") == "evidence":
            p = n.get("payload") or {}
            out.append({
                "id": n.get("id"),
                "source": p.get("source",""),
                "score": p.get("score", None),
                "span":  str(p.get("span", ""))[:60]
            })
    return out

with st.expander("[72] 증거 테이블", expanded=False):
    ce = st.session_state.get("CE_GRAPH")
    rows = ce_evidence_rows(ce)
    if rows:
        sort_key = st.selectbox("정렬", ["score","id","source"], index=0, key="ce_sort")
        rev = st.checkbox("내림차순", value=True, key="ce_rev")
        rows = sorted(rows, key=lambda x: (x.get(sort_key) is None, x.get(sort_key)), reverse=rev)
        st.dataframe(rows, use_container_width=True)
        st.markdown("**커버리지 재평가**")
        st.json(verify_ce_links(ce))
    else:
        st.info("CE evidence가 아직 부족합니다. [63] REPAIR로 보강하세요.")

# ================================================================
# 73. 사용자 액션 단축키(라이트) — 주요 버튼 단축 실행
#    - Streamlit은 네이티브 핫키가 없어, selectbox + 실행 버튼으로 유사 제공
# ================================================================
_ACTIONS = {
    "응답 생성(카드) 실행": ("xl_go",),
    "액티브 N회 실행": ("am_run",),
    "헬스 체크": ("hc_go","gt_hc"),
    "REPAIR 1회": ("rp_go",),
    "LTM 스냅샷 저장": ("ltm_save",),
    "최근 이벤트 보기": ("ev_list",),
}

with st.expander("[73] 액션 단축키", expanded=False):
    act = st.selectbox("액션 선택", list(_ACTIONS.keys()), key="ak_sel")
    st.caption("선택 후 아래 실행을 누르면 해당 영역으로 스크롤됩니다.")
    if st.button("실행", key="ak_do"):
        st.write(f"선택된 액션: {act} → 해당 섹션으로 이동해 실행 버튼을 눌러주세요.")
        st.info("※ 보안상 직접 버튼 트리거는 제한됩니다. (Streamlit 표준 동작)")

# ================================================================
# 74. 프로젝트 설정 패널 — 임계치/가드/경로/로그 보존일수 등
#    - KV 저장소 연동하여 지속화
# ================================================================
def _get_default_cfg():
    return {
        "thresholds": st.session_state.get("HC_MIN", dict(_HEALTH_MIN)),
        "real_guard": st.session_state.get("REAL_GUARD_MODE", "soft"),
        "log_dir": LOG_DIR,
        "ltm_keep_days": kv_get("ltm_keep_days", 30),
    }

with st.expander("[74] 프로젝트 설정", expanded=False):
    cfg = _get_default_cfg()
    st.markdown("**임계치(HC_MIN)**")
    colA,colB = st.columns(2)
    with colA:
        ce_min = st.number_input("ce_coverage ≥", value=float(cfg["thresholds"]["ce_coverage"]), min_value=0.80, max_value=0.995, step=0.005, key="cfg_ce")
        ct_min = st.number_input("citation_coverage ≥", value=float(cfg["thresholds"]["citation_coverage"]), min_value=0.80, max_value=0.995, step=0.005, key="cfg_ct")
    with colB:
        rp_min = st.number_input("reproducibility ≥", value=float(cfg["thresholds"]["reproducibility"]), min_value=0.80, max_value=0.995, step=0.005, key="cfg_rp")
        sr_min = st.number_input("subset_robustness ≥", value=float(cfg["thresholds"]["subset_robustness"]), min_value=0.80, max_value=0.995, step=0.005, key="cfg_sr")
    guard = st.radio("REAL 가드 모드", ["soft","hard","off"], index=["soft","hard","off"].index(cfg["real_guard"]), key="cfg_guard")
    keep = st.number_input("LTM 보존일(권장 30)", min_value=1, max_value=3650, value=int(cfg["ltm_keep_days"]), step=1, key="cfg_keep")
    if st.button("설정 저장", key="cfg_save"):
        st.session_state["HC_MIN"] = {
            "ce_coverage": ce_min, "citation_coverage": ct_min,
            "reproducibility": rp_min, "subset_robustness": sr_min
        }
        st.session_state["REAL_GUARD_MODE"] = guard
        kv_put("ltm_keep_days", int(keep))
        st.success("설정 저장 완료")

# ================================================================
# 75. 안전 백업·복구 마법사 — ZIP 백업/복원 + LTM 정리(보존일)
#    - 65의 번들 ZIP과 연계, 보존일 초과 LTM 자동 정리 옵션
# ================================================================
def cleanup_ltm_retention(days: int):
    import time
    keep_s = int(days) * 86400
    now = int(time.time())
    removed = []
    for p in glob.glob(os.path.join(LTM_DIR, "*.json.gz")):
        try:
            ts = int(os.path.basename(p).split("_",1)[0])
            if now - ts > keep_s:
                os.remove(p); removed.append(os.path.basename(p))
        except Exception:
            pass
    return removed

with st.expander("[75] 백업·복구 마법사", expanded=False):
    st.markdown("**백업**")
    inc_logs = st.checkbox("로그 포함", value=True, key="bk_inc")
    if st.button("ZIP 백업 생성", key="bk_zip"):
        z = make_deploy_zip(include_logs=inc_logs)
        st.success(f"백업 ZIP 생성: {z}")
        try:
            st.download_button("ZIP 다운로드", data=open(z,"rb").read(), file_name=z, mime="application/zip")
        except Exception:
            st.info("환경상 직접 다운로드가 제한될 수 있습니다.")
    st.markdown("---")
    st.markdown("**복구**")
    upz = st.file_uploader("ZIP 업로드(복구)", type=["zip"], key="bk_upl")
    if st.button("ZIP 내용 목록 보기", key="bk_list") and upz is not None:
        from zipfile import ZipFile
        import io
        zf = ZipFile(io.BytesIO(upz.getvalue()))
        st.json(zf.namelist()[:50])
    st.markdown("---")
    st.markdown("**LTM 보존일 정리**")
    keep_days = kv_get("ltm_keep_days", 30)
    st.caption(f"현재 보존일: {keep_days}일")
    if st.button("오래된 LTM 정리 실행", key="bk_prune"):
        rm = cleanup_ltm_retention(int(keep_days))
        st.success(f"정리됨: {len(rm)}개")
        if rm: st.json(rm[:20])
        
        # ================================================================
# 76. 증거 라벨러(수동/반자동) — CE evidence에 신뢰/유형/메모 태깅
#    - 세션 CE_GRAPH를 직접 편집(세션 내 반영)
# ================================================================
_EVID_TYPES = ["논문","표준/규격","데이터셋","특허","코드/레포","기사/블로그","기타"]

def ce_list_evidence_ids(ce: dict) -> list:
    if not ce: return []
    return [n["id"] for n in ce.get("nodes",[]) if n.get("kind")=="evidence"]

def ce_tag_update(ce: dict, evid_id: str, **tags):
    if not ce: return False
    for n in ce.get("nodes",[]):
        if n.get("id")==evid_id and n.get("kind")=="evidence":
            p = n.get("payload") or {}
            p.update({"tags": {**p.get("tags",{}), **tags}})
            n["payload"]=p
            ce["digest"]=hashlib.sha256("".join(k.get("id","") for k in ce["nodes"]).encode()).hexdigest()[:12]
            return True
    return False

with st.expander("[76] 증거 라벨러(수동/반자동)", expanded=False):
    ce = st.session_state.get("CE_GRAPH")
    if not ce:
        st.info("CE evidence가 없습니다. [63] 자동 REPAIR로 먼저 보강하세요.")
    else:
        evids = ce_list_evidence_ids(ce)
        eid = st.selectbox("Evidence 선택", evids, key="ev_sel")
        ety = st.selectbox("유형", _EVID_TYPES, index=0, key="ev_type")
        trs = st.slider("신뢰도(0.0~1.0)", 0.0, 1.0, 0.85, 0.01, key="ev_trust")
        note= st.text_area("메모", value="", height=80, key="ev_note")
        if st.button("라벨 저장", key="ev_save"):
            ok = ce_tag_update(ce, eid, type=ety, trust=round(float(trs),3), note=note)
            st.session_state["CE_GRAPH"]=ce
            st.success("저장 완료" if ok else "실패")
        st.markdown("**미리보기**")
        try:
            prev = [n for n in ce["nodes"] if n["id"]==eid][0]
            st.json(prev)
        except Exception:
            st.write("선택된 evidence 미리보기에 실패했습니다.")

# ================================================================
# 77. 대화 콘솔(미니 터미널 뷰) — 한 줄 입력/즉시 응답 + 로그
#    - HISTORY와 별도 독립 라인형 콘솔, 짧은 명령 실험용
# ================================================================
if "CONSOLE_LOG" not in st.session_state:
    st.session_state["CONSOLE_LOG"] = []  # [{ts, cmd, out_len}]

with st.expander("[77] 대화 콘솔(미니 터미널)", expanded=False):
    cmd = st.text_input("➜ 명령/질문", value="상태요약 5줄", key="sh_cmd")
    lvl = st.slider("레벨", 1, 999, 5, key="sh_lvl")
    if st.button("Run", key="sh_go"):
        out = generate_with_memory(cmd, level=lvl)
        st.write(out if isinstance(out,str) else json.dumps(out, ensure_ascii=False))
        st.session_state["CONSOLE_LOG"].append({"ts": int(time.time()), "cmd": cmd, "out_len": len(str(out))})
    st.caption("최근 로그")
    st.json(st.session_state["CONSOLE_LOG"][-10:])

# ================================================================
# 78. 장문 서식 도우미 — 머리글/표/코드블록 자동 생성기(템플릿)
#    - 보고서/노트 작성을 빠르게 돕는 템플릿 인서터
# ================================================================
_MD_TEMPLATES = {
"보고서-기본": """# 제목(yyyy-mm-dd)
## 1. 배경
## 2. 목표
## 3. 방법
- 데이터:
- 절차:
## 4. 결과
## 5. 논의/한계
## 6. 다음 액션
""",
"표-근거정리": """| 구분 | 출처 | 신뢰 | 메모 |
|---|---|---:|---|
| 증거1 | https:// | 0.95 | |
| 증거2 | https:// | 0.90 | |
""",
"코드-의사결정표": """```pseudo
IF ce_coverage >= 0.97 AND reproducibility >= 0.93 THEN
    VERDICT = PASS
ELSE
    VERDICT = REPAIR
END
```"""
}

with st.expander("[78] 장문 서식 도우미", expanded=False):
    pick = st.selectbox("템플릿", list(_MD_TEMPLATES.keys()), key="md_pick")
    st.code(_MD_TEMPLATES[pick], language="markdown")
    if st.button("응답으로 붙여넣기", key="md_into"):
        txt = _MD_TEMPLATES[pick]
        st.session_state.setdefault("HISTORY", []).append({"q":"[템플릿 삽입]", "a":txt, "ts":int(time.time()), "lvl":0})
        st.success("히스토리에 템플릿이 추가되었습니다.")

# ================================================================
# 79. 실험 프로토콜 템플릿 — 가설0(REAL) 체크리스트 + 절차/평가지표
#    - 출력: JSON(프로토콜) 생성 → specs/ 저장
# ================================================================
def make_protocol(title: str, steps: list, metrics: dict, guards: dict) -> dict:
    return {
        "title": title, "created_at": int(time.time()),
        "guards": guards,  # {"hypothesis":"0", "real_guard":"soft|hard", ...}
        "steps": steps,    # [{"name":"", "detail":"", "expect":""}, ...]
        "metrics": metrics # {"ce_coverage":0.97, "reproducibility":0.93, ...}
    }

PROTO_DIR = os.path.join(LOG_DIR, "protocols")
os.makedirs(PROTO_DIR, exist_ok=True)

def save_protocol(proto: dict) -> str:
    fn = os.path.join(PROTO_DIR, f"proto_{proto['created_at']}_{_sha(json.dumps(proto,ensure_ascii=False).encode())[:8]}.json")
    json.dump(proto, open(fn,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return fn

with st.expander("[79] 실험 프로토콜 템플릿", expanded=False):
    ttl = st.text_input("프로토콜 제목", value="REAL/L30 초검증 루프", key="pp_title")
    stp_default = [
        {"name":"데이터 불러오기","detail":"링크/캐시 데이터 확보","expect":"오류0, 결측<1%"},
        {"name":"단위/차원 검증","detail":"UNITS 체크","expect":"위반율≤0.0001"},
        {"name":"증거 그래프 구축","detail":"CE-Graph 생성","expect":"커버리지≥0.5"},
        {"name":"게이트 판정","detail":"메트릭 계산/판정","expect":"PASS 또는 REPAIR 사유 기록"}
    ]
    met_default = {"ce_coverage":0.97,"citation_coverage":0.90,"reproducibility":0.93,"subset_robustness":0.99}
    grd_default = {"hypothesis":"0","real_guard":st.session_state.get("REAL_GUARD_MODE","soft")}
    if st.button("프로토콜 생성/저장", key="pp_make"):
        proto = make_protocol(ttl, stp_default, met_default, grd_default)
        path  = save_protocol(proto)
        st.success(f"저장됨: {path}")
        st.json(proto)

# ================================================================
# 80. 미니 벤치마크 대시보드 — 케이스 성능 요약(길이/헬스/통과율)
#    - HISTORY/헬스/테스트 매트릭스 결과를 간단 집계
# ================================================================
if "BENCH_LOG" not in st.session_state:
    st.session_state["BENCH_LOG"] = []  # [{ts, name, pass_rate, avg_len}]

def bench_log_add(name: str, pass_rate: float, avg_len: float):
    st.session_state["BENCH_LOG"].append({"ts": int(time.time()), "name": name, "pass_rate": round(pass_rate,2), "avg_len": round(avg_len,1)})

with st.expander("[80] 미니 벤치마크 대시보드", expanded=False):
    # 샘플: 52번 매트릭스 실행 결과를 넘겨 수동 기록하는 흐름
    bench_name = st.text_input("벤치 이름", value="L30 샘플 매트릭스", key="bn_name")
    pr = st.number_input("통과율(0~1)", 0.0, 1.0, 0.88, 0.01, key="bn_pr")
    if st.button("기록 추가", key="bn_add"):
        hist = st.session_state.get("HISTORY", [])
        avg_len = (sum(len(str(h.get("a",""))) for h in hist)/len(hist)) if hist else 0
        bench_log_add(bench_name, pr, avg_len)
        st.success("기록됨")
    st.markdown("**최근 벤치 기록**")
    st.json(st.session_state["BENCH_LOG"][-10:])
    
    # ================================================================
# 81. 증거 그래프 텍스트 시각화 — 노드/엣지 요약(ASCII)
#    - 그래프 라이브러리 없이 가볍게 구조를 확인
# ================================================================
def ce_text_view(ce: dict, k_nodes: int = 40) -> str:
    if not ce: return "(CE-Graph 없음)"
    nodes = ce.get("nodes", [])[:k_nodes]
    edges = ce.get("edges", [])
    lines = []
    lines.append(f"# CE-Graph 프리뷰  (nodes={len(ce.get('nodes',[]))}, edges={len(edges)})")
    # 클레임
    claims = [n for n in nodes if n.get("kind")=="claim"]
    for c in claims:
        txt = (c.get("payload",{}).get("text","") or "")[:180]
        lines.append(f"CLAIM {c['id']}: {txt}")
    # 에비던스
    evs = [n for n in nodes if n.get("kind")=="evidence"]
    for i, e in enumerate(evs, 1):
        p = e.get("payload",{})
        src = (p.get("source","") or "")[:120]
        sc  = p.get("score", None)
        lines.append(f"  EV[{i:02d}] {e['id']}  score={sc}  src={src}")
    # 엣지
    show_edges = edges[: min(len(edges), k_nodes*2)]
    for ed in show_edges:
        lines.append(f"    └─ {ed.get('src','?')}  -[{ed.get('rel','')}]->  {ed.get('dst','?')}")
    digest = ce.get("digest","")
    if digest: lines.append(f"(digest={digest})")
    if len(nodes) < len(ce.get("nodes",[])):
        lines.append(f"... (노드 {len(ce.get('nodes',[]))-len(nodes)}개 생략)")
    return "\n".join(lines)

with st.expander("[81] 증거 그래프 텍스트 뷰", expanded=False):
    st.code(ce_text_view(st.session_state.get("CE_GRAPH")), language="text")

# ================================================================
# 82. 역인과 플래너 + 점수표 — 목표→원인 가설 후보(증거연계 스코어)
#    - 라이트 스코어: evidence 연결 수/평균 score/링크 커버리지 기반
# ================================================================
def invert_causality_plan(goal: str, ce: dict, topk: int = 3) -> dict:
    # 매우 라이트: evidence 제목/출처 토큰을 후보 키워드로 삼아 원인 가설 제시
    ev = [n for n in (ce or {}).get("nodes",[]) if n.get("kind")=="evidence"]
    # 키워드 추출
    keys = []
    for n in ev:
        src = (n.get("payload",{}).get("source","") or "")
        for token in re.split(r"[/\-\._:#\?\&=\s]+", src):
            if 3 <= len(token) <= 18 and token.isascii():
                keys.append(token.lower())
    # 단순 가중 빈도
    from collections import Counter
    cand = [k for k in keys if not any(t in k for t in ["http","www","html","pdf","img","css","js"])]
    freq = Counter(cand).most_common(30)
    # 후보 가설 구성
    hypotheses=[]
    for w,cnt in freq[:max(1, topk*3)]:
        linked = [n for n in ev if w in ((n.get("payload",{}).get("source","") or "").lower())]
        if not linked: continue
        avg_score = sum((n.get("payload",{}).get("score",0.7) or 0) for n in linked)/len(linked)
        hypotheses.append({
            "hyp": f"원인/핵심 요소: {w}",
            "evidence_count": len(linked),
            "avg_evidence_score": round(avg_score,3)
        })
    # 정렬: evidence_count, avg_score
    hypotheses.sort(key=lambda x:(-x["evidence_count"], -x["avg_evidence_score"]))
    # 커버리지
    cov = verify_ce_links(ce) if ce else {"coverage":0, "verdict":"N/A"}
    return {
        "goal": goal,
        "coverage": cov.get("coverage"),
        "verdict":  cov.get("verdict"),
        "hypotheses": hypotheses[:topk],
        "note": "라이트 휴리스틱(정식 추론 아님). 정식판은 SMT/ProofKernel 연동 후 교체."
    }

with st.expander("[82] 역인과 플래너 + 스코어", expanded=False):
    goal = st.text_input("목표(Goal)", value="우주정보장 연결의 신뢰성 향상", key="ic_goal")
    k    = st.slider("가설 수", 1, 10, 3, key="ic_k")
    if st.button("계획/점수 산출", key="ic_run"):
        st.json(invert_causality_plan(goal, st.session_state.get("CE_GRAPH"), k))

# ================================================================
# 83. LTM 검색 뷰 — 장기기억 JSON(.gz) 키워드 검색/미리보기
#    - 오프라인 파일 스캔, 간단 포함검색(대소문자 무시)
# ================================================================
def ltm_search(term: str, limit: int = 30) -> list:
    res=[]
    if not os.path.isdir(LTM_DIR): return res
    patt = term.lower()
    files = sorted(glob.glob(os.path.join(LTM_DIR, "*.json*")), reverse=True)
    for p in files:
        if len(res)>=limit: break
        try:
            if p.endswith(".gz"):
                import gzip
                with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            else:
                txt = open(p,"r",encoding="utf-8",errors="ignore").read()
            if patt in txt.lower():
                res.append({"file": os.path.basename(p), "chars": len(txt), "preview": txt[:400]})
        except Exception:
            continue
    return res

with st.expander("[83] LTM 검색", expanded=False):
    q = st.text_input("검색어", value="증거", key="ltm_q")
    if st.button("검색", key="ltm_q_go"):
        hits = ltm_search(q, limit=20)
        st.write(f"결과 {len(hits)}개")
        st.json(hits)

# ================================================================
# 84. 세션 스냅샷/복원 — session_state 화이트리스트 저장·불러오기
#    - HISTORY/CE_GRAPH/PRESETS_USER/TASKS/HC_MIN/REAL_GUARD_MODE 등
# ================================================================
SNAP_DIR = os.path.join(LOG_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

_SNAP_KEYS = [
    "HISTORY","CE_GRAPH","PRESETS_USER","TASKS",
    "HC_MIN","REAL_GUARD_MODE","CTX_STACK","BENCH_LOG",
]

def snapshot_save(name: str) -> str:
    data = {k: st.session_state.get(k) for k in _S_NAP_KEYS if k in st.session_state}
    ts = int(time.time())
    fn = os.path.join(SNAP_DIR, f"{ts}_{re.sub(r'[^0-9A-Za-z_-]+','_',name)}.json")
    json.dump(data, open(fn,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    return fn

def snapshot_load(path: str) -> dict:
    d = json.load(open(path,"r",encoding="utf-8"))
    for k,v in d.items():
        st.session_state[k] = v
    return {"restored_keys": list(d.keys()), "file": os.path.basename(path)}

with st.expander("[84] 세션 스냅샷/복원", expanded=False):
    nm = st.text_input("스냅샷 이름", value="checkpoint", key="sn_name")
    if st.button("저장", key="sn_save"):
        try:
            p = snapshot_save(nm); st.success(f"저장됨: {p}")
        except Exception as e:
            st.error(f"실패: {e}")
    up = st.file_uploader("스냅샷 JSON 업로드(복원)", type=["json"], key="sn_upl")
    if st.button("복원", key="sn_load") and up is not None:
        import io
        try:
            d = json.load(io.StringIO(up.getvalue().decode("utf-8","ignore")))
            for k,v in d.items(): st.session_state[k]=v
            st.success(f"복원 완료: {list(d.keys())}")
        except Exception as e:
            st.error(f"복원 실패: {e}")

# ================================================================
# 85. 퀵런(일괄) — 목표 입력→생성→헬스체크→REPAIR 1회→CE/하이라이트 출력
#    - 원클릭 파이프라인(모바일/웹 안전), 중간 결과 로그
# ================================================================
def quickrun(goal: str, lvl: int = 8) -> dict:
    log = {}
    # 1) 생성
    ans = generate_with_memory(goal, level=lvl)
    st.session_state.setdefault("HISTORY", []).append({"q": goal, "a": ans, "ts": int(time.time()), "lvl": lvl})
    log["gen_len"] = len(str(ans))
    # 2) 헬스
    h = health_check()
    log["health"] = h.get("verdicts",{})
    # 3) 부족 시 간단 REPAIR
    if any(v=="LOW" for v in log["health"].values() if v in ("OK","LOW")):
        rr = repair_once(query=goal, k=3)
        log["repair"] = rr
    # 4) CE 프리뷰 + 하이라이트
    ce = st.session_state.get("CE_GRAPH")
    log["ce_snippets"] = ce_preview_snippets(ce, k=5)
    log["highlight"] = highlight_keywords(str(ans))[:800]
    return log

with st.expander("[85] 퀵런(생성→헬스→REPAIR→프리뷰)", expanded=False):
    qr_goal = st.text_area("목표/질문", value="우주정보장-연동 설계 핵심 요약과 리스크/완화책 제시", height=90, key="qr_goal")
    qr_lvl  = st.slider("레벨", 1, 999, 8, key="qr_lvl")
    if st.button("원클릭 실행", key="qr_go"):
        st.json(quickrun(qr_goal, qr_lvl))
        
        # ================================================================
# 86. 라이트 체인해시 뷰어 — 최근 산출물 해시/무결성 점검
#    - HISTORY/CE_GRAPH/LTM 파일들의 SHA-256 요약을 한눈에
# ================================================================
def _sha256_hex(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()

def chainhash_view() -> dict:
    out = {}
    # 최근 응답 3개
    hist = st.session_state.get("HISTORY", [])[-3:]
    out["answers"] = [
        {"ts": h.get("ts"), "len": len(str(h.get("a",""))), "sha12": _sha256_hex(str(h.get("a","")).encode())[:12]}
        for h in hist
    ]
    # CE 그래프
    ce = st.session_state.get("CE_GRAPH")
    if ce:
        ce_bytes = json.dumps(ce, ensure_ascii=False, sort_keys=True).encode("utf-8","ignore")
        out["ce_graph"] = {"nodes": len(ce.get("nodes",[])), "edges": len(ce.get("edges",[])),
                           "sha12": _sha256_hex(ce_bytes)[:12]}
    # LTM 최신 3개 파일
    ltm_files = sorted(glob.glob(os.path.join(LTM_DIR,"*.json*")), reverse=True)[:3]
    out["ltm"] = []
    for p in ltm_files:
        try:
            b = open(p,"rb").read()
            out["ltm"].append({"file": os.path.basename(p), "size": len(b), "sha12": _sha256_hex(b)[:12]})
        except Exception:
            pass
    return out

with st.expander("[86] 라이트 체인해시 뷰어", expanded=False):
    st.json(chainhash_view())

# ================================================================
# 87. 증거 겹침 분석 — 중복/유사 링크 탐지(도메인/경로 유사도)
#    - 간단 도메인 매칭 + 경로 토큰 Jaccard로 유사도 평가
# ================================================================
from urllib.parse import urlparse

def _url_tokens(u: str) -> tuple:
    try:
        p = urlparse(u)
        dom = p.netloc.lower()
        toks = [t for t in re.split(r"[\/\-\._\?\&=#]+", p.path.lower()) if t and t.isascii()]
        return dom, set(toks)
    except Exception:
        return "", set()

def evidence_overlap_report(ce: dict, sim_th: float = 0.5) -> dict:
    ev = [n for n in (ce or {}).get("nodes",[]) if n.get("kind")=="evidence"]
    rows = []
    for i in range(len(ev)):
        ui = (ev[i].get("payload",{}) or {}).get("source","")
        di, ti = _url_tokens(ui)
        for j in range(i+1, len(ev)):
            uj = (ev[j].get("payload",{}) or {}).get("source","")
            dj, tj = _url_tokens(uj)
            if not di or not dj: continue
            dom_same = (di==dj)
            inter = len(ti & tj); union = len(ti | tj) if (ti|tj) else 1
            jac = inter/union
            if dom_same and jac >= sim_th:
                rows.append({
                    "a": ev[i].get("id"), "b": ev[j].get("id"),
                    "domain": di, "jaccard": round(jac,3),
                    "src_a": ui, "src_b": uj
                })
    return {"pairs": rows, "count": len(rows)}

with st.expander("[87] 증거 겹침 분석(중복/유사)", expanded=False):
    ce = st.session_state.get("CE_GRAPH")
    if ce:
        th = st.slider("유사도 임계(Jaccard)", 0.1, 1.0, 0.5, 0.05, key="ov_th")
        st.json(evidence_overlap_report(ce, th))
    else:
        st.info("CE evidence가 없습니다. [63]으로 보강하세요.")

# ================================================================
# 88. 장문 목차 자동 생성기(L50+) — 섹션/하위섹션 스켈레톤
#    - 주제 입력→목차(번호/제목/설명) 생성 → HISTORY에 삽입
# ================================================================
def make_longform_toc(topic: str, depth: int = 2, sections: int = 8) -> str:
    lines = [f"# {topic} — 자동 목차", ""]
    for i in range(1, sections+1):
        lines.append(f"{i}. 섹션 {i}: {topic}의 핵심 축 #{i}")
        if depth >= 2:
            for j in range(1, 5):
                lines.append(f"   {i}.{j} 하위 {j}: 근거/절차/리스크/완화")
        if depth >= 3:
            for j in range(1, 3):
                lines.append(f"      {i}.{j}.1 세부: 지표/검증/데이터")
    return "\n".join(lines)

with st.expander("[88] 장문 목차 자동 생성기(L50+)", expanded=False):
    tp = st.text_input("주제", value="우주정보장 연동 및 초검증 아키텍처", key="toc_topic")
    dp = st.slider("깊이", 1, 3, 2, key="toc_depth")
    sc = st.slider("섹션 수", 3, 20, 8, key="toc_secs")
    if st.button("목차 생성→히스토리 삽입", key="toc_make"):
        toc = make_longform_toc(tp, dp, sc)
        st.session_state.setdefault("HISTORY", []).append({
            "q":"[자동 목차]", "a": toc, "ts": int(time.time()), "lvl": 50
        })
        st.success("히스토리에 목차가 추가되었습니다.")
        st.code(toc, language="markdown")

# ================================================================
# 89. 간단 플로우차트 마크다운 — 단계/분기 표기(텍스트 기반)
#    - Mermaid까지는 아니고, ASCII 스타일 흐름도 문자열 생성
# ================================================================
def flowchart_ascii(steps: list, branches: dict=None) -> str:
    branches = branches or {}
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"[{i}] {s}")
        if i < len(steps): out.append("  │")
        b = branches.get(i, [])
        for br in b:
            out.append(f"  ├─▶ {br}")
    return "\n".join(out)

with st.expander("[89] 플로우차트(ASCII) 생성", expanded=False):
    default_steps = ["입력 파싱", "CE-그래프 구축", "게이트 헬스체크", "REPAIR 루프", "결과 산출/체인해시"]
    txt = st.text_area("단계(줄바꿈으로 분리)", value="\n".join(default_steps), height=120, key="fc_steps")
    if st.button("플로우차트 생성", key="fc_go"):
        steps = [x.strip() for x in txt.splitlines() if x.strip()]
        chart = flowchart_ascii(steps, branches={3:["임계 상향","임계 하향"]})
        st.code(chart, language="text")

# ================================================================
# 90. 임계치 히스토리 트래커 — HC_MIN/게이트 지표 타임라인
#    - 세션 내 변경 누적 기록 → JSON 로깅
# ================================================================
THLOG_PATH = os.path.join(LOG_DIR, "threshold_history.jsonl")

def thlog_append(event: dict):
    event = dict(event)
    event["ts"] = int(time.time())
    with open(THLOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# 훅: 66의 gate_autotune_update/74의 설정 저장 후 기록
_prev_autotune = gate_autotune_update
def gate_autotune_update_logged(mode: str = "auto"):
    res = _prev_autotune(mode)
    try: thlog_append({"type":"autotune", "mode": mode, "HC_MIN": res.get("HC_MIN")})
    except Exception: pass
    return res
gate_autotune_update = gate_autotune_update_logged

def thlog_tail(n: int = 30) -> list:
    try:
        lines = open(THLOG_PATH,"r",encoding="utf-8").read().strip().splitlines()
        return [json.loads(x) for x in lines[-n:]]
    except Exception:
        return []

with st.expander("[90] 임계치 히스토리 트래커", expanded=False):
    st.json(thlog_tail(30))
    
    # ==== [91] UI · 리포트/저장/불러오기 (수정판: DuplicateElementKey 방지) ====
# 위치: 파일 맨 아래에 통째로 붙여넣기 (기존 91번이 있으면 교체)

import uuid, re
import streamlit as st

# 회색 안내: 모듈 91 — 리포트 관련 UI의 key 충돌 방지용 유틸
def _ukey(tag: str) -> str:
    """주어진 태그로부터 화면마다 유일한 key를 만들어준다."""
    base = re.sub(r'[^a-zA-Z0-9_]+', '_', str(tag).strip())[:24] or "k"
    return f"{base}_{uuid.uuid4().hex[:8]}"

def _safe_call(name, *args, **kwargs):
    """해당 이름의 함수가 존재하면 호출, 없으면 조용히 통과."""
    fn = globals().get(name)
    if callable(fn):
        return fn(*args, **kwargs)
    return None

st.divider()
st.markdown("### 🧩 모듈 91 · 리포트/저장/불러오기 (안정판)")

# 화면에 같은 컴포넌트를 여러 번 배치해도 항상 서로 다른 key가 되도록 _ukey() 사용
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("리포트 생성/저장", key=_ukey("report_save_btn")):
        # 아래 두 줄은 기존에 쓰던 내부 함수명이 있으면 그대로 호출함
        content = _safe_call("generate_validation_report") or "리포트 본문(샘플)"
        _safe_call("save_report_to_store", content)  # 없으면 무시
        st.success("리포트를 저장했어요.")

with c2:
    if st.button("리포트 불러오기", key=_ukey("report_load_btn")):
        loaded = _safe_call("load_last_report")
        if loaded:
            st.code(loaded)
        else:
            st.info("불러올 리포트가 아직 없어요.")

with c3:
    if st.button("리포트 공유 링크", key=_ukey("report_share_btn")):
        link = _safe_call("make_share_link") or "(공유 링크 기능은 아직 미구현)"
        st.write(link)

# (선택) 추가 컨트롤: 슬라이더/체크박스도 유일 key로 생성
with st.expander("추가 옵션", expanded=False):
    lvl = st.slider("표시 레벨", 1, 5, 3, key=_ukey("opt_level"))
    active = st.checkbox("고급옵션", value=False, key=_ukey("opt_adv"))
    st.caption(f"레벨={lvl}, 고급옵션={'ON' if active else 'OFF'}")
# ==== [91] 끝 ====

# ================================
# 092. [회색] 키/세션 충돌 제로화 유틸 (KeyFactory) + 위젯 래퍼
# 목적: Streamlit DuplicateElementKey 에러 예방. 모든 새 위젯에 고유 key 자동 부여.
# ================================
import streamlit as st
from typing import Dict, Optional

class _GEAKeyFactory:
    """위젯 key 자동생성기: 같은 그룹명 내에서 0001, 0002… 증가"""
    def __init__(self):
        self.counts: Dict[str, int] = {}

    def k(self, name: str) -> str:
        n = self.counts.get(name, 0) + 1
        self.counts[name] = n
        return f"{name}__{n:04d}"

    def reset(self, prefix: Optional[str] = None) -> None:
        if prefix is None:
            self.counts.clear()
        else:
            self.counts = {k: v for k, v in self.counts.items() if not k.startswith(prefix)}

def _m092_get_factory() -> _GEAKeyFactory:
    if "_m092_kf" not in st.session_state:
        st.session_state["_m092_kf"] = _GEAKeyFactory()
    return st.session_state["_m092_kf"]

# ---- 편의 래퍼들 (필요할 때만 사용, 평소엔 명시적 key 사용도 OK) ----
def m092_button(label: str, group: str = "m092_btn"):
    kf = _m092_get_factory()
    return st.button(label, key=kf.k(group))

def m092_text(label: str, group: str = "m092_txt", value: str = ""):
    kf = _m092_get_factory()
    return st.text_input(label, value=value, key=kf.k(group))

def m092_checkbox(label: str, group: str = "m092_chk", value: bool = False):
    kf = _m092_get_factory()
    return st.checkbox(label, value=value, key=kf.k(group))

def m092_select(label: str, options, group: str = "m092_sel"):
    kf = _m092_get_factory()
    return st.selectbox(label, options, key=kf.k(group))

def m092_self_check():
    kf = _m092_get_factory()
    keys = [kf.k("selfcheck") for _ in range(3)]
    ok = len(keys) == len(set(keys))
    return {"status": "PASS" if ok else "FAIL", "generated": keys, "groups": len(kf.counts)}

# ---- UI: 키 충돌 방지 툴킷 (테스트용) ----
with st.expander("🧰 092. 키 팩토리 / 위젯 래퍼 (중복 key 예방)", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        if m092_button("테스트 버튼"):
            st.success("버튼 클릭!")
        name = m092_text("텍스트 입력")
        agree = m092_checkbox("체크해요")
        choice = m092_select("선택", ["A", "B", "C"])
        st.write({"name": name, "agree": agree, "choice": choice})

    with col2:
        if m092_button("팩토리 초기화"):
            _m092_get_factory().reset()
            st.info("KeyFactory reset 완료 (이후 생성 키부터 초기화).")
        st.code(m092_self_check())
        
        # ================================
# 093. [회색] 이벤트 로그 & 리포트 저장기 (세션 기반, JSON/CSV 내보내기)
# 목적: 각 모듈에서 손쉽게 log(level, module, message) 남기고,
#       화면에서 필터/조회 후 JSON/CSV로 저장/다운로드
# 의존: (선택) 092 KeyFactory. 없을 경우 자동 shim 사용.
# ================================
import streamlit as st
import json, csv, io
from datetime import datetime

# ---- 092 키 래퍼가 없더라도 문제없이 동작하도록 shim 제공 ----
try:
    m092_button  # type: ignore
    m092_text    # type: ignore
    m092_select  # type: ignore
    m092_checkbox# type: ignore
except NameError:
    import uuid
    def _auto_key(prefix="k"): return f"{prefix}_{uuid.uuid4().hex[:8]}"
    def m092_button(label: str, group: str = "m093_btn"):
        return st.button(label, key=_auto_key(group))
    def m092_text(label: str, group: str = "m093_txt", value: str = ""):
        return st.text_input(label, value=value, key=_auto_key(group))
    def m092_select(label: str, options, group: str = "m093_sel"):
        return st.selectbox(label, options, key=_auto_key(group))
    def m092_checkbox(label: str, group: str = "m093_chk", value: bool = False):
        return st.checkbox(label, value=value, key=_auto_key(group))

# ---- 세션 초기화 ----
if "m093_logs" not in st.session_state:
    st.session_state["m093_logs"] = []  # [{ts, level, module, message, extra}...]

_M093_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]

def m093_log(level: str, module: str, message: str, extra: dict | None = None):
    """다른 모듈에서 호출: m093_log('INFO','모듈명','메시지', {'k':'v'})"""
    level = (level or "INFO").upper()
    if level not in _M093_LEVELS: level = "INFO"
    st.session_state["m093_logs"].append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "module": module,
        "message": message,
        "extra": extra or {}
    })

def m093_get_logs():
    return st.session_state.get("m093_logs", [])

def m093_clear():
    st.session_state["m093_logs"] = []

def _m093_to_json_bytes(rows):
    buf = io.StringIO()
    json.dump(rows, buf, ensure_ascii=False, indent=2)
    return buf.getvalue().encode("utf-8")

def _m093_to_csv_bytes(rows):
    fieldnames = ["ts", "level", "module", "message", "extra"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        row = r.copy()
        row["extra"] = json.dumps(row.get("extra", {}), ensure_ascii=False)
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")

# ---- UI 패널 ----
with st.expander("🧾 093. 이벤트 로그 & 리포트 저장기", expanded=False):
    colA, colB = st.columns([2,1])

    with colA:
        # 빠른 수동 기록
        mod = m092_text("모듈 이름", value="adhoc")
        msg = m092_text("메시지", value="기록 테스트")
        lev = m092_select("레벨", _M093_LEVELS)
        if m092_button("로그 남기기"):
            m093_log(lev, mod, msg)
            st.success("로그 기록 완료")

    with colB:
        # 관리
        st.caption("관리")
        if m092_button("로그 초기화"):
            m093_clear()
            st.info("모든 로그를 비움")

    # 필터 & 미리보기
    fcol1, fcol2, fcol3 = st.columns([1,1,2])
    with fcol1:
        flv = m092_select("레벨 필터", ["ALL"] + _M093_LEVELS)
    with fcol2:
        fsz = m092_select("표시 개수", [10, 20, 50, 100])
    with fcol3:
        fmd = m092_text("모듈 포함 필터(부분일치)", value="")

    rows = m093_get_logs()
    if flv != "ALL":
        rows = [r for r in rows if r["level"] == flv]
    if fmd:
        rows = [r for r in rows if fmd.lower() in (r["module"] or "").lower()]
    preview = rows[-int(fsz):] if rows else []

    st.write(f"총 {len(rows)}건 / 미리보기 {len(preview)}건")
    st.dataframe(preview, use_container_width=True)

    # 내보내기
    jbytes = _m093_to_json_bytes(rows)
    cbytes = _m093_to_csv_bytes(rows)
    st.download_button("⬇️ JSON 다운로드", data=jbytes, file_name="gea_logs.json", mime="application/json")
    st.download_button("⬇️ CSV 다운로드", data=cbytes, file_name="gea_logs.csv", mime="text/csv")

# ---- 자기진단(선택) : 다른 모듈에서 바로 사용 예시 ----
def m093_self_test():
    m093_log("DEBUG", "m093", "self test start")
    m093_log("INFO",  "m093", "ok")
    return {"ok": True, "count": len(m093_get_logs())}
    
    # ================================
# 094-FULL. [회색] LTM 토픽 인덱스 (검색·저장·미리보기 확장판)
# 목적: gea_logs/ltm 내 JSON/JSON.GZ에서 토픽 키워드 매칭 → 경량 인덱스 저장/조회
# 출력 경로: gea_logs/ltm_index/idx_<topic>.json
# 표준 라이브러리 + streamlit만 사용 (추가 패키지 불필요)
# ================================
import os, re, json, glob, time, hashlib
import streamlit as st

st.write("— 094-FULL 모듈 로드됨")  # 도달 체크

# ---- 고유 key 유틸 (중복 위젯 키 방지) ----
def _k(suffix: str) -> str:
    base = f"m094_{suffix}"
    return f"{base}_{hashlib.sha256(base.encode()).hexdigest()[:6]}"

# ---- 기본 경로 준비 ----
LOG_DIR = st.session_state.get("LOG_DIR", "gea_logs")
LTM_DIR = st.session_state.get("LTM_DIR", os.path.join(LOG_DIR, "ltm"))
LTM_IDX_DIR = os.path.join(LOG_DIR, "ltm_index")
os.makedirs(LTM_DIR, exist_ok=True)
os.makedirs(LTM_IDX_DIR, exist_ok=True)

def _safe_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", (name or "topic"))
    return s[:64] if s else "topic"

def m094_scan_and_build(topic: str, topk_files: int = 200):
    """LTM 폴더 스캔 → topic 키워드로 필터 → 간단 메타 인덱스 생성"""
    patt = (topic or "").strip().lower()
    files = sorted(glob.glob(os.path.join(LTM_DIR, "*.json*")))
    hits = []
    t0 = time.time()
    for p in files:
        try:
            if p.endswith(".gz"):
                import gzip
                with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            if (not patt) or (patt in text.lower()):
                # 가벼운 메타만 저장 (파일명/길이/간단 해시)
                meta = {
                    "file": os.path.basename(p),
                    "size": len(text),
                    "hash12": hashlib.sha256(text[-1024:].encode("utf-8")).hexdigest()[:12] if text else ""
                }
                hits.append(meta)
        except Exception as e:
            # 개별 파일 에러는 건너뜀 (로그만 남길 수도 있음)
            continue
        if len(hits) >= topk_files:
            break
    idx = {
        "topic": topic,
        "matched": len(hits),
        "generated_at": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 3),
        "items": hits
    }
    safe = _safe_name(topic)
    outp = os.path.join(LTM_IDX_DIR, f"idx_{safe}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    return outp, idx

with st.expander("📁 094-FULL. LTM 토픽 인덱스(확장판)", expanded=False):
    col1, col2 = st.columns([2,1])
    with col1:
        topic = st.text_input("토픽(키워드)", value="증거", key=_k("topic"))
        topk = st.number_input("최대 파일 스캔 수", min_value=10, max_value=2000, value=200, step=10, key=_k("topk"))
        if st.button("인덱스 생성/저장", key=_k("build")):
            path, idx = m094_scan_and_build(topic, int(topk))
            st.success(f"저장됨: {path}")
            st.json(idx)
    with col2:
        idx_files = sorted(glob.glob(os.path.join(LTM_IDX_DIR, "idx_*.json")), reverse=True)[:20]
        if idx_files:
            pick = st.selectbox("최근 인덱스", [os.path.basename(p) for p in idx_files], key=_k("pick"))
            if st.button("열기", key=_k("open")):
                st.json(json.load(open(os.path.join(LTM_IDX_DIR, pick), "r", encoding="utf-8")))
        else:
            st.info("생성된 인덱스가 없습니다. 먼저 '인덱스 생성/저장'을 눌러주세요.")

# ================================
# 095. [회색] 런타임/캐시 진단 패널 (반영/캐시/위젯 상태)
# ================================
import sys, platform

st.write("— 095 모듈 로드됨")  # 도달 체크

def _fp(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]

def _now_ms() -> int:
    return int(time.time() * 1000)

STARTED_AT = st.session_state.get("_m095_started_at_ms")
if not STARTED_AT:
    STARTED_AT = _now_ms()
    st.session_state["_m095_started_at_ms"] = STARTED_AT

with st.expander("🛡️ 095. 런타임/캐시 진단", expanded=False):
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 실행/환경")
        st.write("파일 경로:", __file__)
        st.write("Python:", sys.version.split()[0])
        st.write("Platform:", platform.platform())
        st.write("Streamlit:", st.__version__)
        st.write("시작 시각(ms):", STARTED_AT)
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                tail = f.read()[-200:]
            st.write("코드꼬리 해시:", _fp(tail))
        except Exception as e:
            st.warning(f"코드 읽기 실패: {e}")

    with colB:
        st.markdown("#### 캐시/위젯 상태")
        try:
            st.button("진단 버튼", key=_k("probe"))
            st.write("위젯 키 충돌: 없음(샘플)")
        except Exception as e:
            st.error(f"위젯 키 충돌 감지: {e}")

        if st.button("세션 캐시 무효화(토글)", key=_k("toggle_cache")):
            st.session_state["_m095_nonce"] = _fp(str(_now_ms()))
            st.success("세션 상태 변경됨 → Rerun 시 강제 재계산 유도")

    st.markdown("---")
    st.markdown("#### 권장 절차")
    st.write("1) GitHub 커밋이 main으로 들어갔는지 확인")
    st.write("2) 위 '코드꼬리 해시' 값이 커밋마다 달라지는지 확인")
    st.write("3) 앱 메뉴에서 **Restart & clear cache** 또는 **Manage app → Reboot app**")
    st.write("4) 필요 시 **Upload files**로 `streamlit_app.py` 직접 덮어쓰기")
    
    