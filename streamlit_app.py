# ================================================================
# GEA · 최소 구동 P0 — UIS 연동(스텁) + CE-그래프 + 초검증 + E2E
# 규칙: 모듈은 항상 "맨 아래"에 추가(탑쌓기). 중간 삽입은 번호 자리수 확장.
# 에러나 충돌 시, 해당 "번호 블록" 통째 교체(부분 수정 금지).
# ================================================================

import streamlit as st
import hashlib, json, time, re
from typing import List, Dict, Tuple

st.set_page_config(page_title="GEA P0 (UIS+CE+Gate+E2E)", page_icon="💠", layout="wide")

# ───────────────────────────────────────────────
# 공용 유틸
def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def _id_key(mid: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in mid.split("-"))

# ───────────────────────────────────────────────
# [1] 표지/목표 (필수)
st.title("GEA · P0 — 우주정보장 연동(스텁) + 초검증 + E2E")
st.caption("최소구동: 입력→검색(스텁)→CE-그래프→초검증→원클릭 E2E")

if "goals" not in st.session_state:
    st.session_state.goals = {
        "now":  "소스 인덱스 구성 · CE-그래프 생성",
        "near": "초검증 PASS율 상향",
        "mid":  "현실 데이터 피드 연동",
        "far":  "자가진화/기억 통합"
    }

with st.expander("🎯 현재 목표", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.info(f"**단기**\n\n{st.session_state.goals['now']}")
    c2.success(f"**근시**\n\n{st.session_state.goals['near']}")
    c3.warning(f"**중기**\n\n{st.session_state.goals['mid']}")
    c4.error(f"**장기**\n\n{st.session_state.goals['far']}")

# ───────────────────────────────────────────────
# [2] 우주정보장(UIS) 연동 — 스텁 구현 (필수)
st.subheader("🌌 우주정보장 연동(스텁)")

# 초기 소스 (원하면 UI로 추가 가능)
if "uis_sources" not in st.session_state:
    st.session_state.uis_sources = [
        {"id":"src:arxiv:1602.03837","title":"Gravitational Waves (LIGO)","url":"https://arxiv.org/abs/1602.03837","year":2016,"trust":0.98},
        {"id":"src:nist:constants","title":"CODATA Constants (NIST)","url":"https://physics.nist.gov/constants","year":2022,"trust":0.99},
        {"id":"src:ligo:open","title":"LIGO Open Data","url":"https://losc.ligo.org","year":2024,"trust":0.97},
    ]

with st.expander("소스 인덱스 관리(선택)", expanded=False):
    new_id = st.text_input("소스 ID", placeholder="src:my:paper")
    new_title = st.text_input("제목", placeholder="My Important Paper")
    new_url = st.text_input("URL", placeholder="https://…")
    new_year = st.number_input("연도", 1800, 2100, 2024)
    new_trust = st.slider("신뢰도", 0.0, 1.0, 0.95, 0.01)
    if st.button("소스 추가"):
        st.session_state.uis_sources.append({
            "id":new_id, "title":new_title, "url":new_url,
            "year":int(new_year), "trust":float(new_trust)
        })
        st.success("소스 추가 완료")
    st.markdown("**현재 소스(상위 20)**")
    for s in st.session_state.uis_sources[:20]:
        st.markdown(f"- `{s['id']}` · {s['title']} · {s['year']} · trust={s['trust']}")

def uis_search(q: str, k: int = 8) -> List[Dict]:
    """아주 단순한 스텁 검색기."""
    ql = (q or "").lower().strip()
    hits = []
    for src in st.session_state.uis_sources:
        blob = f"{src['id']} {src['title']} {src.get('url','')}".lower()
        score = 0.95 if ql and ql in blob else 0.6 + 0.05*len(set(ql.split()) & set(blob.split()))
        hits.append({
            "id": src["id"], "title": src["title"],
            "url": src.get("url",""), "score": round(min(0.99, score),3)
        })
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k]

def build_ce_graph(claim: str, hits: List[Dict]) -> Dict:
    """Claim–Evidence 그래프 스텁."""
    claim_text = claim or "(no-claim)"
    nodes = [{"id": f"claim:{_sha(claim_text)[:12]}", "kind":"claim", "payload":{"text":claim_text}}]
    edges = []
    for h in hits:
        evid_id = f"evi:{_sha(h['id'])[:10]}"
        nodes.append({"id": evid_id, "kind":"evidence",
                      "payload":{"src":h["id"],"title":h["title"],"url":h["url"],"score":h["score"]}})
        edges.append({"src": evid_id, "dst": nodes[0]["id"], "rel":"supports"})
    digest = _sha(json.dumps({"nodes":nodes,"edges":edges}, sort_keys=True))
    return {"nodes":nodes,"edges":edges,"digest":digest,"created_at":time.time()}

st.markdown("—")
claim = st.text_input("주장(Claim)", placeholder="예: LIGO 데이터로 h≈ΔL/L 검증 경로 구성")
query = st.text_input("검색 질의(Query)", placeholder="예: LIGO, NIST, gravitational waves")
k = st.slider("검색 개수(k)", 1, 12, 6)
if st.button("질의→그래프 생성"):
    hits = uis_search(query or claim, k=k)
    ce = build_ce_graph(claim or query or "default-claim", hits)
    st.session_state["last_ce_graph"] = ce
    st.success(f"CE-그래프 생성 완료 · digest={ce['digest'][:12]}…")
    st.json({"hits":hits, "ce_graph":ce})
    st.download_button("CE-그래프 JSON 다운로드",
                       data=json.dumps(ce, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="ce_graph.json", mime="application/json")

# ───────────────────────────────────────────────
# [3] 초검증(게이트) — 스텁 (필수)
st.subheader("🧪 초검증(게이트) — 스텁")

SIGNAL_BASELINES = {
    "ce_coverage_min": 0.97,
    "citation_min":    0.90,
    "reprod_min":      0.93,
    "logic_violation_max": 0.0005,
    "unit_violation_max":  0.0001,
    "surprise_p_max":  0.005
}

def _metrics_from_ce(ce_graph: Dict) -> Dict[str, float]:
    if not ce_graph: return {"ce_coverage":0.0,"citation":0.0}
    n_evi = sum(1 for n in ce_graph["nodes"] if n["kind"]=="evidence")
    n_edges = len(ce_graph["edges"])
    ce_cov = 0.8 + min(0.2, 0.02 * n_evi + 0.01 * n_edges)  # 스텁 휴리스틱
    citation = 0.85 + min(0.15, 0.02 * n_evi)               # 스텁 휴리스틱
    return {"ce_coverage": round(ce_cov,3), "citation": round(citation,3)}

def _metrics_from_text(text: str) -> Dict[str, float]:
    if not text:
        return {"reprod":0.0,"logic_violation":0.001,"unit_violation":0.0002,"surprise_p":0.01}
    tok_eq = len(re.findall(r"=|≈|∝|≤|≥", text))
    tok_units = len(re.findall(r"\b(m|s|kg|J|Hz|N|Pa)\b", text))
    tok_refs  = len(re.findall(r"https?://|src:", text))
    reprod = min(0.99, 0.90 + 0.01*tok_eq + 0.01*tok_refs)
    logic_viol = max(0.00005, 0.0008 - 0.0001*tok_eq)
    unit_viol  = max(0.00002, 0.0003 - 0.00002*tok_units)
    surprise_p = max(0.001, 0.02 - 0.002*(tok_eq+tok_units))
    return {
        "reprod": round(reprod,3),
        "logic_violation": round(logic_viol,6),
        "unit_violation": round(unit_viol,6),
        "surprise_p": round(surprise_p,3)
    }

def gate_verdict(m: Dict[str,float]) -> Tuple[str,str]:
    if m["ce_coverage"]   < SIGNAL_BASELINES["ce_coverage_min"]:    return ("REPAIR","증거 하한 미달")
    if m["citation"]      < SIGNAL_BASELINES["citation_min"]:       return ("REPAIR","인용 하한 미달")
    if m["reprod"]        < SIGNAL_BASELINES["reprod_min"]:         return ("REPAIR","재현성 미달")
    if m["logic_violation"] > SIGNAL_BASELINES["logic_violation_max"]: return ("REPAIR","논리 위반율 초과")
    if m["unit_violation"]  > SIGNAL_BASELINES["unit_violation_max"]:  return ("REPAIR","단위/차원 위반율 초과")
    if m["surprise_p"]      > SIGNAL_BASELINES["surprise_p_max"]:      return ("REPAIR","놀라움 p 초과")
    return ("PASS","ok")

ce = st.session_state.get("last_ce_graph")
st.caption(f"참조 CE-그래프: {'있음' if ce else '없음(먼저 [2]에서 생성)'}")
body = st.text_area("본문/설명(검증용 텍스트)", height=150,
                    placeholder="예: h ≈ ΔL/L, 단위 m/m (무차원), src: https://losc.ligo.org")
if st.button("초검증 실행"):
    m1 = _metrics_from_ce(ce) if ce else {"ce_coverage":0.0,"citation":0.0}
    m2 = _metrics_from_text(body or "")
    metrics = {
        "ce_coverage": m1["ce_coverage"], "citation": m1["citation"],
        "reprod": m2["reprod"], "logic_violation": m2["logic_violation"],
        "unit_violation": m2["unit_violation"], "surprise_p": m2["surprise_p"],
    }
    verdict, reason = gate_verdict(metrics)
    st.json({"metrics":metrics, "verdict":verdict, "reason":reason})
    st.success("✅ PASS — 기준 충족") if verdict=="PASS" else st.warning(f"🔧 REPAIR — {reason}")

# ───────────────────────────────────────────────
# [4] E2E 원클릭 하트비트 (필수)
st.subheader("🫀 E2E 하트비트(원클릭)")
_default_claim = "LIGO 데이터로 h≈ΔL/L 경로 구성"
_default_query = "LIGO gravitational waves NIST constants"

if st.button("E2E 실행(기본)"):
    hits = uis_search(_default_query, k=6)
    ce2 = build_ce_graph(_default_claim, hits)
    st.session_state["last_ce_graph"] = ce2
    body_text = "h ≈ ΔL / L, 단위: m/m (무차원). src: https://losc.ligo.org"
    m1 = _metrics_from_ce(ce2)
    m2 = _metrics_from_text(body_text)
    metrics = {**m1, **m2}
    verdict, reason = gate_verdict(metrics)
    st.json({
        "hits": hits,
        "ce_graph_digest": ce2["digest"][:12],
        "metrics": metrics,
        "verdict": verdict,
        "reason": reason
    })
    st.success("✅ E2E PASS — 경로 정상") if verdict=="PASS" else st.warning(f"🔧 E2E REPAIR — " + reason)

# ───────────────────────────────────────────────
# [1-0] 평면 목차(요약) — P0에선 간단 표기
st.subheader("📖 평면 목차(요약)")
st.markdown("| 번호 | 이름 | 기능 |")
st.markdown("|---:|---|---|")
st.markdown("| `1` | 첫 장 표지 | 목표 카드 |")
st.markdown("| `2` | UIS 연동(스텁) | 소스 등록·검색, CE-그래프 생성 |")
st.markdown("| `3` | 초검증(스텁) | 신호 계산 + PASS/REPAIR |")
st.markdown("| `4` | E2E 하트비트 | 원클릭 경로 확인 |")