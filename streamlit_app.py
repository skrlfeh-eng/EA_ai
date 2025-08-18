# -*- coding: utf-8 -*-
# EA · Ultra — Stable chat + per-session infinite memory (JSONL)
# Smart retrieval (TF-IDF + BM25 + Recency + Pin), response level 1~9999
import os, sys, re, json, time, math, hashlib, random, traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import streamlit as st

APP_NAME   = "EA · Ultra"
BUILD_TAG  = "EA-ULTRA-20250818"
DATA_DIR   = Path("data")
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"      # 모든 세션 메모리
DIALOG_LOG = DATA_DIR / "dialog.jsonl"      # 모든 대화 로그(세션 포함)

# ---------------- FS helpers ----------------
def ensure_dirs():
    try: DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def nowz(): return datetime.utcnow().isoformat()+"Z"

def jsonl_append(path: Path, obj: dict):
    try:
        ensure_dirs()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception: pass

def jsonl_read_all(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ---------------- tiny state (file -> session fallback) ----------------
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

# ---------------- conversation (per session) ----------------
def load_session_messages(session_id: str, limit: int = 300):
    """dialog.jsonl에서 해당 세션의 마지막 limit개를 세션 상태로 로드"""
    all_rows = jsonl_read_all(DIALOG_LOG)
    msgs = [r for r in all_rows if r.get("session")==session_id][-limit:]
    sset("messages", msgs)

def add_msg(session_id: str, role: str, content: str):
    entry = {"t": nowz(), "session": session_id, "role": role, "content": content}
    # 1) 세션 상태
    msgs = sget("messages", [])
    msgs.append(entry); sset("messages", msgs)
    # 2) 영구 대화 로그
    jsonl_append(DIALOG_LOG, entry)
    # 3) 메모리 자동 기록
    mem_append({"t": entry["t"], "session": session_id, "kind":"dialog",
                "role": role, "text": content, "tags": []})

def clear_msgs():
    sset("messages", [])

# ---------------- text utils ----------------
def dedupe_text(text: str):
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)
    return text

def clamp(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …"

def parse_tags(text: str) -> Tuple[str, List[str]]:
    tags=[]; t=text.strip()
    if t.startswith("!pin "):
        tags.append("pin"); t=t[5:].strip()
    return t, tags

# ---------------- adapters ----------------
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=1200):
        words=(prompt or "").split()
        seed=int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),16)
        rng=random.Random(seed)
        lead=rng.choice(["핵심:","정리:","요약:","아이디어:","계획:"])
        body=" ".join(words[:max(16,len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=1200):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are a helpful Korean assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=0.7)
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=1200):
        r=self.model.generate_content(prompt,
            generation_config={"temperature":0.7,"max_output_tokens":max_tokens})
        return getattr(r,"text","") or ""

def resolve_adapter(want:str):
    if want=="OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가→Mock: {e}", icon="⚠️")
    if want=="Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가→Mock: {e}", icon="⚠️")
    return MockAdapter(), False

# ---------------- memory(JSONL, per session) ----------------
def mem_append(item: Dict[str,Any]): jsonl_append(MEM_PATH, item)

def mem_iter():
    if not MEM_PATH.exists(): return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: yield json.loads(line)
            except Exception: continue

def mem_add_note(session_id: str, text: str, tags: List[str]|None=None):
    mem_append({"t": nowz(), "session": session_id, "kind":"note", "text": text, "tags": tags or []})

def mem_add_summary(session_id: str, messages: List[Dict[str,str]]):
    last=messages[-6:]
    brief=" / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": session_id, "kind":"summary", "text": brief, "tags":["auto"]})

# ---------------- smart retrieval ----------------
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def tokenize(s:str)->List[str]: return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]

def build_index(items: List[Dict[str,Any]])->Dict[str,Any]:
    docs=[]; df={}; lengths=[]
    for it in items:
        toks=tokenize(it.get("text",""))
        docs.append(toks); lengths.append(len(toks) or 1)
        for w in set(toks): df[w]=df.get(w,0)+1
    N=max(1,len(docs)); avgdl=sum(lengths)/len(lengths) if lengths else 1.0
    return {"docs":docs,"df":df,"N":N,"avgdl":avgdl,"raw":items}

def score_bm25(q: List[str], idx: Dict[str,Any], k1=1.5, b=0.75)->List[float]:
    df, N, avgdl, docs = idx["df"], idx["N"], idx["avgdl"], idx["docs"]
    sc=[0.0]*len(docs)
    for i,d in enumerate(docs):
        dl=len(d) or 1
        for term in q:
            f=d.count(term)
            if f==0: continue
            n_q=df.get(term,0)
            idf=math.log((N - n_q + 0.5)/(n_q + 0.5) + 1.0)
            denom=f + k1*(1 - b + b*dl/avgdl)
            sc[i]+=idf*(f*(k1+1))/denom
    return sc

def score_tfidf(q: List[str], idx: Dict[str,Any])->List[float]:
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    sc=[0.0]*len(docs)
    qtf={}
    for w in q: qtf[w]=qtf.get(w,0)+1
    qvec={}; 
    for w,c in qtf.items():
        idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
        qvec[w]=c*idf
    qnorm=math.sqrt(sum(v*v for v in qvec.values())) or 1.0
    for i,d in enumerate(docs):
        tf={}; 
        for w in d: tf[w]=tf.get(w,0)+1
        dot=0.0; dnorm_acc=0.0
        for w,tfc in tf.items():
            idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
            wt=tfc*idf
            dnorm_acc+=wt*wt
            if w in qvec: dot+=wt*qvec[w]
        dnorm=math.sqrt(dnorm_acc) or 1.0
        sc[i]=dot/(dnorm*qnorm)
    return sc

def recency_boost(ts_iso:str, now_dt:datetime)->float:
    try: dt=datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception: return 0.9
    days=max(0.0,(now_dt-dt).total_seconds()/86400.0)
    return max(0.25, 1.0/(1.0 + days/7.0))

def pin_boost(tags: List[str])->float: return 1.2 if ("pin" in (tags or [])) else 1.0

def smart_search(session_id:str, query_text:str, topk:int=5)->List[Dict[str,Any]]:
    pool=[it for it in (mem_iter() or []) if it.get("session")==session_id and it.get("text")]
    if not pool: return []
    idx=build_index(pool)
    q=tokenize(query_text)
    bm=score_bm25(q, idx); tf=score_tfidf(q, idx)
    nowdt=datetime.utcnow()

    scored_list=[]  # <-- 버그 픽스: 리스트 보장
    for i,it in enumerate(idx["raw"]):
        base = 0.55*bm[i] + 0.35*tf[i] + 0.07*recency_boost(it.get("t",""), nowdt)
        final = base * pin_boost(it.get("tags", []))
        scored_list.append((final, it))
    scored_list.sort(key=lambda x: x[0], reverse=True)
    return [it for _,it in scored_list[:topk]]

# ---------------- response level 1~9999 ----------------
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)   # 부드러운 증가
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)

def long_answer(adapter, prompt:str, level:int=3, rounds:int=2)->str:
    max_tokens=level_to_tokens(level)
    acc=""
    base=prompt.strip()
    for i in range(int(rounds)):
        p=base if i==0 else base+"\n(이어서 더 자세히)"
        chunk=str(adapter.generate(p, max_tokens=max_tokens) or "")
        chunk=dedupe_text(chunk)
        if not chunk: break
        acc+=(("\n\n" if acc else "") + clamp(chunk, max_tokens+500))
        time.sleep(0.02)
    return acc.strip()

# ---------------- UI ----------------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("세션별 무한 기억 · 하이브리드 검색 · 레벨 1~9999")

    # 세션/메모리 ON
    col0, col1, col2 = st.columns([1.3,1,1])
    with col0:
        session_id = st.text_input("세션 ID (섞임 방지)", sget("session_id","default"))
        if session_id != sget("session_id"):  # 세션이 바뀌면 해당 세션 로그 로드
            sset("session_id", session_id)
            load_session_messages(session_id)
        else:
            sset("session_id", session_id)
            if "messages" not in st.session_state:
                load_session_messages(session_id)
    with col1:
        mem_on = st.toggle("Memory ON", value=bool(sget("mem_on", True)))
        sset("mem_on", mem_on)
    with col2:
        if st.button("대화창 초기화(로그는 보존)"):
            clear_msgs(); st.experimental_rerun()

    # 모델/레벨
    colA, colB, colC = st.columns([1,1.2,1])
    with colA:
        provider = st.selectbox("Provider", ["OpenAI","Gemini","Mock"], index=0)
    with colB:
        level = st.number_input("응답 레벨 (1~9999)", min_value=1, max_value=9999, value=3, step=1)
        st.caption(f"예산≈{level_to_tokens(level)} tokens")
    with colC:
        rounds = st.number_input("라운드(연결 요청 횟수)", min_value=1, max_value=6, value=2, step=1)

    adapter, api_ok = resolve_adapter(provider)
    st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · session='{sget('session_id')}' · L{int(level)} · R{int(rounds)}")

    # 히스토리
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(str(m["content"]))

    # 입력
    user_text = st.chat_input("메시지를 입력하고 Enter  ·  /remember [!pin] 내용")
    if user_text:
        user_text = dedupe_text(user_text.strip())
        add_msg(sget("session_id"), "user", user_text)

        if mem_on and user_text.startswith("/remember "):
            raw = user_text[len("/remember "):].strip()
            body, tags = parse_tags(raw)
            mem_add_note(sget("session_id"), body, tags=tags)
            with st.chat_message("assistant"):
                st.success(f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}")
            add_msg(sget("session_id"), "assistant", "기억했어 ✅")
        else:
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
                st.markdown(str(ans))
            add_msg(sget("session_id"), "assistant", ans)

        # 주기 요약
        try:
            if len(sget("messages", [])) % 8 == 0 and mem_on:
                mem_add_summary(sget("session_id"), sget("messages", []))
        except Exception: pass

    # 도구
    with st.expander("Memory / Logs"):
        q = st.text_input("메모 검색어", "")
        if q:
            hits = smart_search(sget("session_id"), q, topk=10)
            if not hits: st.info("검색 결과 없음")
            for h in hits:
                tag = (" ".join([f"#{t}" for t in h.get("tags",[])])).strip()
                st.write(f"- ({h.get('kind')}) {h['text']} {tag and '['+tag+']'}")
        c1, c2 = st.columns(2)
        if c1.button("dialog.jsonl 미리보기(최근 50)"):
            tail = jsonl_read_all(DIALOG_LOG)[-50:]
            st.code(json.dumps(tail, ensure_ascii=False, indent=2), language="json")
        if c2.button("memory.jsonl 미리보기(최근 50)"):
            tail = jsonl_read_all(MEM_PATH)[-50:]
            st.code(json.dumps(tail, ensure_ascii=False, indent=2), language="json")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH} · log={DIALOG_LOG}")

# ---------------- entry ----------------
if __name__ == "__main__":
    ensure_dirs()
    render_app()
    
  # -*- coding: utf-8 -*-
# EA · Ultra — All-in-One (single file)
# Chat UI + Infinite Memory(JSONL) + Smart Retrieval + Skills(/use) + Mini-Autobuilder
import os, sys, re, json, time, math, hashlib, random, traceback, types, importlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# ===== App meta =====
APP_NAME   = "EA · Ultra (AIO)"
BUILD_TAG  = "EA-ULTRA-20250818"
DATA_DIR   = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"
DIALOG_LOG = DATA_DIR / "dialog.jsonl"

# ===== Tiny FS helpers =====
def nowz(): return datetime.utcnow().isoformat()+"Z"
def jsonl_append(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False)+"\n")
    except Exception: pass
def jsonl_read_all(path: Path) -> List[dict]:
    if not path.exists(): return []
    out=[]; 
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ===== State (file -> session fallback) =====
def _state_read():
    try:
        return json.loads(STATE_PATH.read_text("utf-8"))
    except Exception:
        return st.session_state.get("_state", {})
def _state_write(obj):
    try:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"] = obj
def sget(k, d=None): return _state_read().get(k, d)
def sset(k, v): s=_state_read(); s[k]=v; _state_write(s)

# ===== Conversation (per-session) =====
def load_session_messages(session_id: str, limit=300):
    rows = [r for r in jsonl_read_all(DIALOG_LOG) if r.get("session")==session_id][-limit:]
    sset("messages", rows)
def add_msg(session_id: str, role: str, content: str):
    entry = {"t": nowz(), "session": session_id, "role": role, "content": content}
    msgs = sget("messages", []); msgs.append(entry); sset("messages", msgs)
    jsonl_append(DIALOG_LOG, entry)
    mem_append({"t": entry["t"], "session": session_id, "kind":"dialog",
                "role": role, "text": content, "tags": []})
def clear_msgs(): sset("messages", [])

# ===== Text utils =====
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def dedupe(text:str):
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\b(\w+)(\s+\1){1,}\b', r'\1', text)
    return text
def tokenize(s:str)->List[str]: return [t.lower() for t in TOK_RE.findall(s or "") if t.strip()]
def clamp(text:str, n:int)->str: return text if len(text)<=n else text[:n]+" …"

# ===== Adapters (Mock/OpenAI/Gemini) =====
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=1200):
        words=(prompt or "").split()
        seed=int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(),16)
        rng=random.Random(seed)
        lead=rng.choice(["핵심:","정리:","요약:","아이디어:"])
        body=" ".join(words[:max(16,len(words))])
        return f"{lead} {body}"
class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI  # type: ignore
        key=os.getenv("OPENAI_API_KEY"); 
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=1200):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are a helpful Korean assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=0.7)
        return r.choices[0].message.content or ""
class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai  # type: ignore
        key=os.getenv("GEMINI_API_KEY"); 
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=1200):
        r=self.model.generate_content(prompt,
            generation_config={"temperature":0.7,"max_output_tokens":max_tokens})
        return getattr(r,"text","") or ""
def resolve_adapter(want:str):
    if want=="OpenAI":
        try: return OpenAIAdapter(), True
        except Exception as e: st.toast(f"OpenAI 불가→Mock: {e}", icon="⚠️")
    if want=="Gemini":
        try: return GeminiAdapter(), True
        except Exception as e: st.toast(f"Gemini 불가→Mock: {e}", icon="⚠️")
    return MockAdapter(), False

# ===== Memory(JSONL) + Smart retrieval (BM25 + TF-IDF + Recency + Pin) =====
def mem_append(item: Dict[str,Any]): jsonl_append(MEM_PATH, item)
def mem_iter():
    if not MEM_PATH.exists(): return
    with MEM_PATH.open("r", encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: yield json.loads(ln)
            except Exception: continue
def mem_add_note(session_id:str, text:str, tags=None):
    mem_append({"t": nowz(), "session": session_id, "kind":"note", "text": text, "tags": tags or []})
def mem_add_summary(session_id:str, messages: List[Dict[str,str]]):
    last=messages[-6:]; brief=" / ".join(f"{m['role']}: {clamp(m['content'],120)}" for m in last)
    mem_append({"t": nowz(), "session": session_id, "kind":"summary", "text": brief, "tags":["auto"]})

def build_index(items: List[Dict[str,Any]]):
    docs=[]; df={}; lengths=[]
    for it in items:
        toks=tokenize(it.get("text",""))
        docs.append(toks); lengths.append(len(toks) or 1)
        for w in set(toks): df[w]=df.get(w,0)+1
    N=max(1,len(docs)); avgdl=sum(lengths)/len(lengths) if lengths else 1.0
    return {"docs":docs,"df":df,"N":N,"avgdl":avgdl,"raw":items}
def score_bm25(q:List[str], idx, k1=1.5, b=0.75):
    df, N, avgdl, docs = idx["df"], idx["N"], idx["avgdl"], idx["docs"]
    sc=[0.0]*len(docs)
    for i,d in enumerate(docs):
        dl=len(d) or 1
        for term in q:
            f=d.count(term)
            if f==0: continue
            n_q=df.get(term,0)
            idf=math.log((N - n_q + 0.5)/(n_q + 0.5) + 1.0)
            denom=f + k1*(1 - b + b*dl/avgdl)
            sc[i]+=idf*(f*(k1+1))/denom
    return sc
def score_tfidf(q:List[str], idx):
    df, N, docs = idx["df"], idx["N"], idx["docs"]
    sc=[0.0]*len(docs)
    qtf={}; 
    for w in q: qtf[w]=qtf.get(w,0)+1
    qvec={}; 
    for w,c in qtf.items():
        idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
        qvec[w]=c*idf
    qnorm=math.sqrt(sum(v*v for v in qvec.values())) or 1.0
    for i,d in enumerate(docs):
        tf={}; 
        for w in d: tf[w]=tf.get(w,0)+1
        dot=0.0; dnorm_acc=0.0
        for w,tfc in tf.items():
            idf=math.log((N+1)/(df.get(w,0)+1)) + 1.0
            wt=tfc*idf
            dnorm_acc+=wt*wt
            if w in qvec: dot+=wt*qvec[w]
        dnorm=math.sqrt(dnorm_acc) or 1.0
        sc[i]=dot/(dnorm*qnorm)
    return sc
def recency_boost(ts_iso:str, now_dt:datetime)->float:
    try: dt=datetime.fromisoformat(ts_iso.replace("Z",""))
    except Exception: return 0.9
    days=max(0.0,(now_dt-dt).total_seconds()/86400.0)
    return max(0.25, 1.0/(1.0 + days/7.0))
def pin_boost(tags: List[str])->float: return 1.2 if ("pin" in (tags or [])) else 1.0
def smart_search(session_id:str, query_text:str, topk:int=5)->List[Dict[str,Any]]:
    pool=[it for it in (mem_iter() or []) if it.get("session")==session_id and it.get("text")]
    if not pool: return []
    idx=build_index(pool); q=tokenize(query_text)
    bm=score_bm25(q, idx); tf=score_tfidf(q, idx); nowdt=datetime.utcnow()
    scored=[]
    for i,it in enumerate(idx["raw"]):
        base=0.55*bm[i] + 0.35*tf[i] + 0.07*recency_boost(it.get("t",""), nowdt)
        final=base * pin_boost(it.get("tags", []))
        scored.append((final, it))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [it for _,it in scored[:topk]]

# ===== Response level (1~9999) =====
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)  # soft curve
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)
def long_answer(adapter, prompt:str, level:int=3, rounds:int=2)->str:
    max_tokens=level_to_tokens(level)
    acc=""; base=prompt.strip()
    for i in range(int(rounds)):
        p=base if i==0 else base+"\n(이어서 더 자세히)"
        chunk=str(adapter.generate(p, max_tokens=max_tokens) or "")
        chunk=dedupe(chunk)
        if not chunk: break
        acc+=(("\n\n" if acc else "") + (chunk if len(chunk)<=max_tokens+500 else chunk[:max_tokens+500]+" …"))
        time.sleep(0.02)
    return acc.strip()

# ===== Skills (in-file registry) =====
class Skill:  # base
    name="base"; desc=""; timeout_sec=20
    def run(self, query:str, context:dict)->str: raise NotImplementedError

# 샘플 스킬(즉시 사용 가능)
class SampleEchoSkill(Skill):
    name="sample.echo"; desc="입력 요약 에코"; timeout_sec=10
    def run(self, query:str, context:dict)->str:
        q=(query or "").strip()
        if not q: return "빈 입력이에요."
        return f"[sample.echo] {q[:120]}"+("…" if len(q)>120 else "")

# in-memory registry (여기에 동적으로 추가/삭제 가능)
SKILLS: Dict[str, Skill] = { SampleEchoSkill().name: SampleEchoSkill() }

def safe_run(skill: Skill, query:str, ctx:dict)->str:
    try:
        return str(skill.run(query, ctx or {}))
    except Exception:
        return "(skill error)\n"+traceback.format_exc()

def maybe_run_skill_command(user_text:str, session_id:str):
    if not user_text.startswith("/use "): return None
    try:
        _, rest = user_text.split(" ", 1)
    except ValueError:
        return "(형식) /use skill.name 인자"
    parts = rest.strip().split(" ", 1)
    skill_name = parts[0]; arg = parts[1] if len(parts)>1 else ""
    sk = SKILLS.get(skill_name)
    if not sk: return f"(스킬 없음) {skill_name}"
    return safe_run(sk, arg, {"session": session_id, "raw": user_text})

# ===== Mini-Autobuilder (동적 스킬 생성; 안전 제약: 순수함수형만) =====
DYN_DIR = DATA_DIR / "skills_dynamic"; DYN_DIR.mkdir(parents=True, exist_ok=True)

def make_dynamic_skill(name_slug:str, goal:str):
    """
    매우 단순: 주어진 이름으로 '순수 함수형' 스킬 클래스를 파일로 만들고, 동적 import 후 등록.
    외부 IO 금지. 템플릿만 제공(로직은 간단한 문자열 처리).
    """
    safe_slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in name_slug.lower())
    modname = f"dyn_{safe_slug.replace('.','_')}"
    clsname = "".join(w.capitalize() for w in safe_slug.split(".")) + "Skill"
    code = f'''# -*- coding: utf-8 -*-
from types import SimpleNamespace
class {clsname}:
    name="{safe_slug}"
    desc="{goal.replace('"','\\"')}"
    timeout_sec=10
    def run(self, query: str, context: dict) -> str:
        text = (query or "").strip()
        if not text:
            return "[{safe_slug}] 입력이 비어있습니다."
        # 순수 함수형 자리표시자: 단어수/문자수
        wc = len(text.split())
        lc = len(text)
        return f"[{safe_slug}] 단어:{'{'}wc{'}'} 문자:{'{'}lc{'}'} · {{text[:120]}}"
'''
    path = DYN_DIR / f"{modname}.py"
    path.write_text(code, encoding="utf-8")
    # 동적 import & 등록
    spec = importlib.util.spec_from_file_location(modname, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    cls = getattr(module, clsname)
    obj = cls()
    SKILLS[obj.name] = obj
    return obj.name

# ===== UI =====
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.markdown(f"### {APP_NAME}")
    st.caption("단일 파일 · 무한 기억 · 하이브리드 검색 · 스킬(/use) · 간이 Autobuilder")

    # 세션/메모리
    col0, col1, col2 = st.columns([1.3,1,1])
    with col0:
        session_id = st.text_input("세션 ID", sget("session_id","default"))
        if session_id != sget("session_id"):
            sset("session_id", session_id); load_session_messages(session_id)
        else:
            sset("session_id", session_id)
            if "messages" not in st.session_state: load_session_messages(session_id)
    with col1:
        mem_on = st.toggle("Memory ON", value=bool(sget("mem_on", True))); sset("mem_on", mem_on)
    with col2:
        if st.button("대화창 초기화(로그 보존)"): clear_msgs(); st.experimental_rerun()

    # 모델/레벨
    colA, colB, colC = st.columns([1,1.2,1])
    with colA:
        provider = st.selectbox("Provider", ["OpenAI","Gemini","Mock"], index=0)
    with colB:
        level = st.number_input("응답 레벨(1~9999)", min_value=1, max_value=9999, value=3, step=1)
        st.caption(f"예산≈{level_to_tokens(level)} tokens")
    with colC:
        rounds = st.number_input("라운드", 1, 6, 2, 1)

    adapter, api_ok = resolve_adapter(provider)
    st.info(f"🔌 {adapter.name} {'(연결됨)' if api_ok else '(모의)'} · session={sget('session_id')} · L{int(level)} · R{int(rounds)}")

    # Autobuilder (간단)
    with st.expander("Mini-Autobuilder(순수함수형 스킬 생성)"):
        goal = st.text_input("새 스킬 이름(예: 'auto.wordcount')", "")
        goal_desc = st.text_input("스킬 설명", "")
        if st.button("동적 스킬 생성"):
            if not goal.strip():
                st.warning("스킬 이름을 입력하세요.")
            else:
                try:
                    name = make_dynamic_skill(goal.strip(), goal_desc.strip() or goal.strip())
                    st.success(f"스킬 생성/등록 완료 → /use {name} <텍스트>")
                except Exception as e:
                    st.error(f"생성 실패: {e}")

    # 과거 대화 출력
    for m in sget("messages", []):
        with st.chat_message("user" if m["role"]=="user" else "assistant"):
            st.markdown(str(m["content"]))

    # 입력
    user_text = st.chat_input("메시지를 입력(예: /use sample.echo 안녕, /remember !pin 일정 9/1)")
    if user_text:
        user_text = dedupe(user_text.strip())
        add_msg(sget("session_id"), "user", user_text)

        # /remember
        if user_text.startswith("/remember "):
            raw = user_text[len("/remember "):].strip()
            tags=[]; 
            if raw.startswith("!pin "): tags.append("pin"); raw=raw[5:].strip()
            mem_add_note(sget("session_id"), raw, tags=tags)
            with st.chat_message("assistant"): st.success(f"기억했어 ✅ {('[pin]' if 'pin' in tags else '')}")
            add_msg(sget("session_id"), "assistant", "기억했어 ✅")
        else:
            # /use 스킬
            out = maybe_run_skill_command(user_text, sget("session_id"))
            if out is not None:
                with st.chat_message("assistant"): st.markdown(out)
                add_msg(sget("session_id"), "assistant", out)
            else:
                # 메모리 기반 컨텍스트
                context = ""
                if sget("mem_on", True):
                    hits = smart_search(sget("session_id"), user_text, topk=5)
                    if hits:
                        bullet = "\n".join([f"- {h['text']}" for h in hits])
                        context = f"[참고 메모]\n{bullet}\n\n"
                with st.chat_message("assistant"):
                    try:
                        ans = long_answer(adapter, context + user_text, level=level, rounds=rounds)
                    except Exception:
                        ans = "(내부 오류)\n\n```\n"+traceback.format_exc()+"\n```"
                    st.markdown(str(ans))
                add_msg(sget("session_id"), "assistant", ans)

        # 주기 요약
        try:
            if len(sget("messages", [])) % 8 == 0 and sget("mem_on", True):
                mem_add_summary(sget("session_id"), sget("messages", []))
        except Exception: pass

    # Tools
    with st.expander("Memory / Logs 미리보기"):
        c1, c2 = st.columns(2)
        if c1.button("dialog.jsonl (최근 50)"):
            st.code(json.dumps(jsonl_read_all(DIALOG_LOG)[-50:], ensure_ascii=False, indent=2), language="json")
        if c2.button("memory.jsonl (최근 50)"):
            st.code(json.dumps(jsonl_read_all(MEM_PATH)[-50:], ensure_ascii=False, indent=2), language="json")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH} · log={DIALOG_LOG}")

# ===== entry =====
if __name__ == "__main__":
    render_app()
    
    
    
    