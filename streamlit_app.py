# -*- coding: utf-8 -*-
# EA · Ultra — Chat UI + API adapters + Infinite Memory(JSONL, global session)
# Smart retrieval: TF-IDF + BM25 + Recency + Pin, Persistent Dialog
# Response level: 1 ~ 9999 (scales token budget; capped by model limits)

import os, sys, re, json, time, math, hashlib, random, traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import streamlit as st

APP_NAME   = "EA · Ultra"
BUILD_TAG  = "EA-ULTRA-20250818"
DATA_DIR   = Path("data")
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"    # 영구 메모리(JSONL, append-only)
DIALOG_LOG = DATA_DIR / "dialog.jsonl"    # 모든 대화 영구 로그(JSONL, append-only)
GLOBAL_SESSION = "global"                  # 항상 같은 세션으로 기억(융합 이슈 없음)

# ================== FS helpers ==================
def ensure_dirs():
    try: DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def nowz(): return datetime.utcnow().isoformat() + "Z"

def jsonl_append(path: Path, obj: dict):
    try:
        ensure_dirs()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def jsonl_tail(path: Path, max_lines: int) -> List[dict]:
    if not path.exists(): return []
    try:
        # 간단한 tail 읽기
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-max_lines:]
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: continue
        return out
    except Exception:
        return []

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

# ================== conversation state (persistent + memory) ==================
def add_msg(role: str, content: str):
    """세션 상태 + dialog.jsonl + memory.jsonl(항상 저장) 모두 기록."""
    entry = {"t": nowz(), "role": role, "content": content}
    # 1) in-memory session list
    msgs = sget("messages", [])
    msgs.append(entry)
    sset("messages", msgs)
    # 2) persistent dialog log
    jsonl_append(DIALOG_LOG, entry)
    # 3) memory: 모든 메시지를 자동 기억
    mem_append({
        "t": entry["t"],
        "session": GLOBAL_SESSION,
        "kind": "dialog",
        "role": role,
        "text": content,
        "tags": []
    })

def load_session_from_dialog(max_turns: int = 200):
    """앱 시작/새로고침 시 dialog.jsonl 마지막 N턴을 세션 메모리에 복구."""
    if "messages" in st.session_state:  # 이미 로드됨
        return
    tail = jsonl_tail(DIALOG_LOG, max_turns)
    sset("messages", tail)

def clear_msgs():
    """대화창(세션 상태)만 초기화. 영구 로그는 남겨둠."""
    sset("messages", [])

# ================== text utils ==================
def dedupe_text(text: str):
    text = re.sub(r'(.)\1{2,}', r'\1', text)               # 글자 반복
    text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)    # 단어 반복
    return text

def clamp(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …"

def parse_tags(text: str) -> Tuple[str, List[str]]:
    """/remember 시 '!pin' 태그를 본문에서 분리"""
    tags = []
    t = text.strip()
    if t.startswith("!pin "):
        tags.append("pin"); t = t[5:].strip()
    return t, tags

# ================== adapters ==================
class MockAdapter:
    name = "Mock"
    def generate(self, prompt, max_tokens=1200):
        words = (prompt or "").split()
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        lead = rng.choice(["핵심:", "정리:", "요약:", "아이디어:", "계획:"])
        body = " ".join(words[: max(16, len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name = "OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client = OpenAI(api_key=key)
        self.model  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    def generate(self, prompt, max_tokens=1200):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":"You are a helpful Korean assistant. Keep answers clear and structured."},
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
    def generate(self, prompt, max_tokens=1200):
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

# ================== Infinite Memory (JSONL, global session) ==================
def mem_append(item: Dict[str, Any]):
    jsonl_append(MEM_PATH, item)

def mem_iter():
    if not MEM_PATH.exists(): return
    try:
        with MEM_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try: yield json.loads(line)
                except Exception: continue
    except Exception:
        return

def mem_add_note(text: str, tags: List[str]|None=None):
    mem_append({"t": nowz(), "session": GLOBAL_SESSION, "kind": "note", "text": text, "tags": tags or []})

def mem_add_summary(messages: List[Dict[str,str]]):
    last = messages[-6:]
    brief = " / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": GLOBAL_SESSION, "kind":"summary", "text": brief, "tags":["auto"]})

# ================== Smart Retrieval (TF-IDF + BM25 + Recency + Pin) ==================
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]

def build_index(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    docs, df, lengths = [], {}, []
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
    for i, d in enumerate(docs):
        dl = len(d) or 1
        for q in query:
            f = d.count(q)
            if f == 0: continue
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            denom = f + k1*(1 - b + b*dl/avgdl)
            scores[i] += idf * (f*(k1+1)) / denom
    return scores

def score_tfidf(query: List[str], idx: Dict[str,Any]) -> List[float]:
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    scores = [0.0]*len(docs)
    qtf = {}
    for q in query: qtf[q] = qtf.get(q,0)+1
    qvec = {}
    for q,c in qtf.items():
        idf = math.log((N+1)/(df.get(q,0)+1)) + 1.0
        qvec[q] = c * idf
    qnorm = math.sqrt(sum(v*v for v in qvec.values())) or 1.0
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
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception:
        return 0.9
    days = max(0.0, (now_dt - dt).total_seconds() / 86400.0)
    return max(0.25, 1.0 / (1.0 + days/7.0))  # 0일=1.0, 7일≈0.7, 30일≈0.4

def pin_boost(tags: List[str]) -> float:
    return 1.2 if ("pin" in (tags or [])) else 1.0

def smart_search(query_text: str, topk=5) -> List[Dict[str,Any]]:
    pool = [it for it in (mem_iter() or []) if it.get("session")==GLOBAL_SESSION and it.get("text")]
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
        base = 0.55*s_bm + 0.35*s_tf + 0.07*s_rc
        final = base * s_pin
        scored.append((final, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _,it in scored[:topk]]

# ================== response length (1~9999 level) ==================
def level_to_tokens(level: int) -> int:
    """
    레벨을 토큰 예산으로 매핑.
    - 1 → ~300 토큰
    - 10 → ~1,500
    - 100 → ~6,000
    - 1000 → ~12,000
    - 9999 → ~16,000 (상한)
    실제 출력은 모델 제한에 의해 더 짧아질 수 있음.
    """
    level = max(1, int(level))
    # 로그성 스케일 + 선형 완충
    est = int(300 + 120 * math.log10(level + 9) * 100)  # 부드러운 증가
    HARD_CAP = int(os.getenv("MAX_TOKENS_CAP", "16000"))  # 모델 한계 상한
    return min(max(est, 300), HARD_CAP)

def long_answer(adapter, prompt, level=3, rounds=2):
    max_tokens = level_to_tokens(level)
    acc = ""
    base = prompt.strip()
    for i in range(int(rounds)):
        p = base if i==0 else base + "\n(이어서 더 자세히)"
        chunk = dedupe_text(adapter.generate(p, max_tokens=max_tokens))
        if not chunk: break
        acc += (("\n\n" if acc else "") + clamp(chunk, max_tokens + 500))
        time.sleep(0.02)
    return acc.strip()

# ================== UI ==================
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    load_session_from_dialog(max_turns=300)  # 앱 시작 시 과거 대화 복구

    st.markdown(f"### {APP_NAME}")
    st.caption("ChatGPT 스타일 · 모든 대화 영구 저장 · 항상 같은 세션(global) · 스마트 메모리 검색")

    # Top controls
    colA, colB, colC = st.columns([1,1.2,1])
    with colA:
        provider = st.selectbox("Provider", ["OpenAI","Gemini","Mock"], index=0)
    with colB:
        level = st.number_input("응답 레벨 (1~9999)", min_value=1, max_value=9999, value=3, step=1)
        st.caption(f"예산≈{level_to_tokens(level)} tokens (모델 제한 내)")
    with colC:
        rounds = st.number_input("라운드(연결 요청 횟수)", min_value=1, max_value=6, value=2, step=1)

    if st.button("대화창만 초기화 (영구로그 유지)"):
        clear_msgs(); st.experimental_rerun()

    adapter, api_ok = resolve_adapter(provider)
    st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · session='{GLOBAL_SESSION}' · level={int(level)} · rounds={int(rounds)}")

    # History (session messages)
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(m["content"])

    # Input
    user_text = st.chat_input("메시지를 입력하고 Enter  ·  /remember [!pin] 내용")
    if user_text:
        user_text = dedupe_text(user_text.strip())
        add_msg("user", user_text)

        # /remember 강제 저장
        if user_text.startswith("/remember "):
            raw = user_text[len("/remember "):].strip()
            body, tags = parse_tags(raw)
            mem_add_note(body, tags=tags)
            with st.chat_message("assistant"):
                st.success(f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}")
            add_msg("assistant", "기억했어 ✅")
        else:
            # 스마트 검색 → 상위 메모 5개 컨텍스트
            hits = smart_search(user_text, topk=5)
            context = ""
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

        # 주기적 자동 요약(8턴 주기)
        try:
            if len(sget("messages", [])) % 8 == 0:
                mem_add_summary(sget("messages", []))
        except Exception:
            pass

    # Tools
    with st.expander("Memory / Logs"):
        q = st.text_input("메모 검색어", "")
        if q:
            hits = smart_search(q, topk=10)
            if not hits: st.info("검색 결과 없음")
            for h in hits:
                tag = (" ".join([f"#{t}" for t in h.get("tags",[])])).strip()
                st.write(f"- ({h.get('kind')}) {h['text']} {tag and '['+tag+']'}")
        c1, c2, c3 = st.columns(3)
        if c1.button("dialog.jsonl → 최근 50줄 미리보기"):
            tail = jsonl_tail(DIALOG_LOG, 50)
            st.code(json.dumps(tail, ensure_ascii=False, indent=2), language="json")
        if c2.button("memory.jsonl → 최근 50줄 미리보기"):
            tail = jsonl_tail(MEM_PATH, 50)
            st.code(json.dumps(tail, ensure_ascii=False, indent=2), language="json")
        if c3.button("전체 메모 삭제(주의)"):
            try:
                if MEM_PATH.exists(): MEM_PATH.unlink()
                st.warning("전체 메모 삭제 완료")
            except Exception as e:
                st.error(f"삭제 실패: {e}")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH} · log={DIALOG_LOG}")

# ================== entry ==================
if __name__ == "__main__":
    ensure_dirs()
    render_app()