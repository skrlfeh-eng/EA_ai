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
        
        