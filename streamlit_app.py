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
                
                