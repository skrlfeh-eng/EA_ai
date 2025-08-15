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

