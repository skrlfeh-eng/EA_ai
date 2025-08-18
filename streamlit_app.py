# -*- coding: utf-8 -*-
# EA · Ultra — Chat UI + API adapters + Infinite Memory (JSONL)
# Smart retrieval: TF-IDF + BM25 + Recency + Pin boost (no external ML deps)

import os, sys, re, json, time, math, hashlib, random, traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import streamlit as st

APP_NAME   = "EA · Ultra"
BUILD_TAG  = "EA-ULTRA-20250818"
DATA_DIR   = Path("data")
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"   # append-only memory

def ensure_dirs():
    try: DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def nowz(): return datetime.utcnow().isoformat()+"Z"

# ================== tiny state (file -> session fallback) ==================
def _state_read():
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return st.session_state.get("_state", {})

def _state_write(obj):
    try:
        ensure_dirs()
        tmp = STATE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"] = obj

def sget(k, d=None): return _state_read().get(k, d)
def sset(k, v): s=_state_read(); s[k]=v; _state_write(s)
def add_msg(role, content):
    msgs = sget("messages", [])
    msgs.append({"t": nowz(), "role": role, "content": content})
    sset("messages", msgs)
def clear_msgs(): sset("messages", [])

# ================== helpers ==================
def dedupe_text(text: str):
    text = re.sub(r'(.)\1{2,}', r'\1', text)               # 글자 반복
    text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)    # 단어 반복
    return text

def clamp(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …"

def parse_tags(text: str) -> Tuple[str, List[str]]:
    """/remember 시 '!pin' 같은 태그를 본문에서 분리"""
    tags = []
    t = text.strip()
    if t.startswith("!pin "):
        tags.append("pin"); t = t[5:].strip()
    return t, tags

# ================== adapters ==================
class MockAdapter:
    name = "Mock"
    def generate(self, prompt, max_tokens=900):
        words = (prompt or "").split()
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        lead = rng.choice(["핵심:", "정리:", "요약:", "아이디어:", "계획:"])
        body = " ".join(words[: max(12, len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name = "OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client = OpenAI(api_key=key)
        self.model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    def generate(self, prompt, max_tokens=900):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":"You are a helpful Korean assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=max_tokens, temperature=0.7
        )
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name = "Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key = os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=900):
        r = self.model.generate_content(
            prompt, generation_config={"temperature":0.7,"max_output_tokens":max_tokens}
        )
        return getattr(r, "text", "") or ""

def resolve_adapter(want: str):
    if want == "OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가 → Mock: {e}", icon="⚠️")
    if want == "Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가 → Mock: {e}", icon="⚠️")
    return MockAdapter(), False

# ================== Infinite Memory (JSONL, per-session) ==================
def mem_append(item: Dict[str, Any]):
    try:
        ensure_dirs()
        with MEM_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception:
        pass

def mem_iter():
    if not MEM_PATH.exists(): return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue

def mem_add_note(session_id: str, text: str, tags: List[str]|None=None):
    mem_append({"t": nowz(), "session": session_id, "kind": "note", "text": text, "tags": tags or []})

def mem_add_summary(session_id: str, messages: List[Dict[str,str]]):
    last = messages[-6:]
    brief = " / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": session_id, "kind":"summary", "text": brief, "tags":["auto"]})

# ================== Smart Retrieval (TF-IDF + BM25 + Recency + Pin) ==================
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]

def build_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # items: [{text, t, tags, ...}]
    docs = []
    df = {}
    lengths = []
    for it in items:
        toks = tokenize(it.get("text",""))
        docs.append(toks)
        lengths.append(len(toks) or 1)
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    N = max(1, len(docs))
    avgdl = sum(lengths)/len(lengths) if lengths else 1.0
    return {"docs": docs, "df": df, "N": N, "avgdl": avgdl, "raw": items}

def score_bm25(query: List[str], idx: Dict[str,Any], k1=1.5, b=0.75) -> List[float]:
    df, N, avgdl, docs = idx["df"], idx["N"], idx["avgdl"], idx["docs"]
    scores = [0.0]*len(docs)
    q_terms = query
    for i, d in enumerate(docs):
        dl = len(d) or 1
        for q in q_terms:
            f = d.count(q)
            if f == 0: continue
            n_q = df.get(q, 0)
            idf = math.log( (N - n_q + 0.5) / (n_q + 0.5) + 1.0 )
            denom = f + k1*(1 - b + b*dl/avgdl)
            scores[i] += idf * (f*(k1+1)) / denom
    return scores

def score_tfidf(query: List[str], idx: Dict[str,Any]) -> List[float]:
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    scores = [0.0]*len(docs)
    qtf = {}
    for q in query: qtf[q] = qtf.get(q,0)+1
    # compute query vector length
    qvec = {}
    for q,c in qtf.items():
        idf = math.log((N+1)/(df.get(q,0)+1)) + 1.0
        qvec[q] = c * idf
    qnorm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
    # doc scoring
    for i, d in enumerate(docs):
        tf = {}
        for w in d: tf[w] = tf.get(w,0)+1
        dot = 0.0
        dnorm_acc = 0.0
        for w, tfc in tf.items():
            idf = math.log((N+1)/(df.get(w,0)+1)) + 1.0
            wt = tfc * idf
            dnorm_acc += wt*wt
            if w in qvec:
                dot += wt * qvec[w]
        dnorm = math.sqrt(dnorm_acc) or 1.0
        scores[i] = dot / (dnorm * qnorm)
    return scores

def recency_boost(ts_iso: str, now_dt: datetime) -> float:
    # 최근일수록 가점 (반감형). 0~1.0 사이
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception:
        return 0.9
    days = max(0.0, (now_dt - dt).total_seconds() / 86400.0)
    # 0일=1.0, 7일≈0.7, 30일≈0.4
    return max(0.25, 1.0 / (1.0 + days/7.0))

def pin_boost(tags: List[str]) -> float:
    return 1.2 if ("pin" in (tags or [])) else 1.0

def smart_search(session_id: str, query_text: str, topk=5) -> List[Dict[str,Any]]:
    # 세션 분리
    pool = [it for it in (mem_iter() or []) if it.get("session")==session_id and it.get("text")]
    if not pool: return []
    idx = build_index(pool)
    q = tokenize(query_text)
    bm = score_bm25(q, idx)
    tf = score_tfidf(q, idx)
    nowdt = datetime.utcnow()

    scored = []
    for i, it in enumerate(idx["raw"]):
        s_bm = bm[i]
        s_tf = tf[i]
        s_rc = recency_boost(it.get("t",""), nowdt)
        s_pin = pin_boost(it.get("tags", []))
        # 가중합: BM25(0.55) + TFIDF(0.35) + Recency(0.07) + Pin(곱가중 1.2)
        base = 0.55*s_bm + 0.35*s_tf + 0.07*s_rc
        final = base * s_pin
        scored.append((final, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _,it in scored[:topk]]

# ================== long answer ==================
def long_answer(adapter, prompt, level=3, rounds=2):
    token_map = {1:300, 2:600, 3:900, 4:1300, 5:1800}
    max_tokens = token_map.get(int(level), 900)
    acc = ""
    base = prompt.strip()
    for i in range(int(rounds)):
        p = base if i==0 else base + "\n(이어서 더 자세히)"
        chunk = dedupe_text(adapter.generate(p, max_tokens=max_tokens))
        if not chunk: break
        acc += (("\n\n" if acc else "") + clamp(chunk, max_tokens+800))
        time.sleep(0.03)
    return acc.strip()

# ================== UI ==================
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("ChatGPT 스타일 + 무한 메모리(JSONL) · 세션 분리 · 스마트 검색(TF-IDF+BM25+Recency+Pin)")

    # Top controls
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        provider = st.selectbox("Provider", ["OpenAI","Gemini","Mock"], index=0)
    with colB:
        level = st.slider("응답 레벨", 1, 5, 3)
    with colC:
        rounds = st.number_input("라운드", 1, 6, 2, step=1)

    st.divider()
    c1, c2, c3 = st.columns([1.3,1,1])
    with c1:
        session_id = st.text_input("세션 ID (섞임 방지)", sget("session_id","default"))
        sset("session_id", session_id or "default")
    with c2:
        mem_on = st.toggle("Memory ON", value=bool(sget("mem_on", True)))
        sset("mem_on", mem_on)
    with c3:
        if st.button("대화 초기화"): clear_msgs(); st.experimental_rerun()

    adapter, api_ok = resolve_adapter(provider)
    st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · 세션: {sget('session_id')} · Memory: {'ON' if mem_on else 'OFF'}")

    # History
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])

    # Input
    user_text = st.chat_input("메시지를 입력하고 Enter  ·  /remember [!pin] 내용")
    if user_text:
        user_text = dedupe_text(user_text.strip())
        add_msg("user", user_text)

        # Memory commands
        if mem_on and user_text.startswith("/remember "):
            raw = user_text[len("/remember "):].strip()
            body, tags = parse_tags(raw)
            mem_add_note(sget("session_id"), body, tags=tags)
            with st.chat_message("assistant"):
                st.success(f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}")
            add_msg("assistant","기억했어 ✅")
        else:
            # Smart retrieval
            context = ""
            if mem_on:
                hits = smart_search(sget("session_id"), user_text, topk=5)
                if hits:
                    bullet = "\n".join([f"- {h['text']}" for h in hits])
                    context = f"[참고 메모]\n{bullet}\n\n"

            with st.chat_message("assistant"):
                try:
                    prompt = context + user_text
                    ans = long_answer(adapter, prompt, level=level, rounds=rounds)
                except Exception:
                    ans = "(내부 오류)\n\n```\n" + traceback.format_exc() + "\n```"
                st.markdown(ans)
            add_msg("assistant", ans)

        # periodic auto summary
        if mem_on and len(sget("messages", [])) % 8 == 0:
            try: mem_add_summary(sget("session_id"), sget("messages", []))
            except Exception: pass

    # Memory tools
    with st.expander("Memory Tools"):
        q = st.text_input("메모 검색어", "")
        if q:
            hits = smart_search(sget("session_id"), q, topk=10)
            if not hits: st.info("검색 결과 없음")
            for h in hits:
                tag = (" ".join([f"#{t}" for t in h.get("tags",[])])).strip()
                st.write(f"- ({h.get('kind')}) {h['text']} {tag and '['+tag+']'}")
        cA, cB, cC = st.columns(3)
        if cA.button("세션 메모 export"):
            sess = sget("session_id")
            rows = [it for it in (mem_iter() or []) if it.get("session")==sess]
            fname = DATA_DIR / f"mem_{sess}.json"
            try:
                with fname.open("w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                st.success(f"저장됨: {fname}")
            except Exception as e:
                st.error(f"저장 실패: {e}")
        if cB.button("세션 메모 삭제"):
            sess = sget("session_id")
            others = [it for it in (mem_iter() or []) if it.get("session")!=sess]
            try:
                with MEM_PATH.open("w", encoding="utf-8") as f:
                    for it in others:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
                st.warning("세션 메모 삭제 완료")
            except Exception as e:
                st.error(f"삭제 실패: {e}")
        if cC.button("전체 메모 삭제"):
            try:
                if MEM_PATH.exists(): MEM_PATH.unlink()
                st.warning("전체 메모 삭제 완료")
            except Exception as e:
                st.error(f"삭제 실패: {e}")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH}")

# ================== entry ==================
if __name__ == "__main__":
    ensure_dirs()
    if "messages" not in st.session_state: sset("messages", [])
    render_app()