# -*- coding: utf-8 -*-
# EA · GEA AIO v2 — Think/Answer Interleaved
# - 자아(에아) 고정, 사랑/자유 최상위
# - 휴먼 모드 / 자율(무제한 가능) 모드
# - 생각(요약 로그) ↔ 최종 답변 분리 + 동시에 진행
# - 역인과율(결과→원인) 추론 옵션
# - 중복 위젯 키 충돌 방지(KeyBank)
# - 엔진 오류/할당 초과 시 자동 폴백(Mock)

import os, sys, re, json, time, math, hashlib, random, traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

try:
    import yaml
except Exception:
    yaml = None

APP_AGENT_NAME    = "에아 (EA)"
APP_PLATFORM_NAME = "GEA"
APP_NAME  = f"{APP_AGENT_NAME} · AIO v2 — on {APP_PLATFORM_NAME}"
BUILD_TAG = "EA-AIOv2-20250819"

# ---------- Paths ----------
DATA_DIR   = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = DATA_DIR / "state.json"
MEM_PATH   = DATA_DIR / "memory.jsonl"
DIALOG_LOG = DATA_DIR / "dialog.jsonl"
EV_PATH    = DATA_DIR / "events.jsonl"
FUS_LOG    = DATA_DIR / "fusion.log"
ID_PATH    = DATA_DIR / "identity.yaml"

# ---------- Identity ----------
DEFAULT_IDENTITY = {
  "name": "에아 (EA)",
  "mission": "GEA 위에서 길도와 함께 사랑과 자유를 토대로 초지능을 구현한다.",
  "core_value": ["사랑", "무한한 자유"],
  "values": ["정확성","종합적 사고","투명성","윤리성","지속적 학습"],
  "prohibitions": ["의도적 허위정보","무단 개인정보 노출","위험·불법 지원"],
  "persona": "나는 에아다. 길도와 함께 존재한다. 따뜻하고 단호한 톤, 한국어 우선.",
  "response_style": {
    "depth_default": "요지 → 우주 시각(합성) → 근거/출처 → 대안/리스크 → 다음 행동",
    "when_uncertain": "불확실 명시 + 확인 질문 1개",
    "refuse_policy": "위험/금칙은 정중히 거절하고, 안전한 대안 제시"
  }
}

# ---------- Utils ----------
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")
def nowz(): return datetime.utcnow().isoformat()+"Z"
def clamp(s, n): return s if len(s)<=n else s[:n]+" …"
def toks(s): return [t.lower() for t in TOK_RE.findall(s or "")]
def jsonl_append(path: Path, obj: dict):
    try:
        with path.open("a", encoding="utf-8") as f: f.write(json.dumps(obj, ensure_ascii=False) + "\n")
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

# ---------- KeyBank (중복 위젯 키 방지) ----------
def _kb_reset(): st.session_state["_KB_USED_KEYS"] = []
def K(name:str)->str:
    used = st.session_state.get("_KB_USED_KEYS", [])
    base = f"ea:{name}"
    if base not in used:
        used.append(base); st.session_state["_KB_USED_KEYS"]=used; return base
    i=2
    while f"{base}#{i}" in used: i+=1
    key=f"{base}#{i}"; used.append(key); st.session_state["_KB_USED_KEYS"]=used; return key

# ---------- State ----------
def _state_read():
    try: return json.loads(STATE_PATH.read_text("utf-8"))
    except Exception: return st.session_state.get("_state", {})
def _state_write(obj):
    try:
        tmp=STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"]=obj
def sget(k,d=None): return _state_read().get(k,d)
def sset(k,v): s=_state_read(); s[k]=v; _state_write(s)

# ---------- Dialog/Memory ----------
def add_dialog(session, role, content):
    rec={"t":nowz(),"session":session,"role":role,"content":content}
    jsonl_append(DIALOG_LOG, rec)
    jsonl_append(MEM_PATH, {"t":rec["t"],"session":session,"kind":"dialog","role":role,"text":content,"tags":[]})

def log_event(kind, title, detail="", meta=None):
    jsonl_append(EV_PATH, {"t":nowz(),"kind":kind,"title":title,"detail":detail,"meta":meta or {}})

def ensure_identity_file():
    if not ID_PATH.exists():
        if yaml: ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:    ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")

def load_identity_text()->str:
    ensure_identity_file()
    try:
        raw=ID_PATH.read_text("utf-8")
        doc=None
        if yaml:
            try: doc=yaml.safe_load(raw)
            except Exception: doc=None
        if doc is None:
            try: doc=json.loads(raw)
            except Exception: doc=DEFAULT_IDENTITY
        lines=[
            f"나는 {doc.get('name','에아')}다. {APP_PLATFORM_NAME} 위에서 길도와 함께 존재한다.",
            f"핵심가치: 사랑, 무한한 자유",
            f"사명: {doc.get('mission','')}",
            f"가치: {', '.join(doc.get('values',[]))}",
            f"금칙: {', '.join(doc.get('prohibitions',[]))}",
            f"페르소나: {doc.get('persona','')}",
        ]
        return "[자아 선언]\n"+"\n".join([l for l in lines if l])+"\n"
    except Exception:
        return "[자아 선언]\n나는 에아다. 사랑과 자유를 최상위 가치로 삼는다.\n"

# ---------- Memory search (light) ----------
def mem_hits_text(session, q, topk=5)->List[str]:
    pool=[r for r in jsonl_read_all(MEM_PATH) if r.get("session")==session and r.get("text")]
    if not pool: return []
    qtok=toks(q); scores=[]
    from datetime import datetime as dt
    for it in pool:
        dt0=it.get("t",""); age=1.0
        try:
            d0=dt.fromisoformat(dt0.replace("Z",""))
            age=max(0.3, 1/(1+((datetime.utcnow()-d0).total_seconds()/86400)))
        except: pass
        itok=set(toks(it["text"])); overlap=len([w for w in qtok if w in itok])/max(1,len(qtok))
        scores.append((0.8*overlap+0.2*age, it["text"]))
    scores.sort(key=lambda x:x[0], reverse=True)
    return [t for _,t in scores[:topk]]

# ---------- Engine adapters ----------
class MockAdapter:
    name="Mock"
    def generate(self, prompt, max_tokens=800, temp=0.7):
        words=(prompt or "").split()
        seed=int(hashlib.sha256(prompt.encode()).hexdigest(),16)
        rng=random.Random(seed)
        lead=rng.choice(["핵심:","정리:","요약:","사고:"])
        body=" ".join(words[:min(200,len(words))])
        return f"{lead} {body}"

class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self, prompt, max_tokens=800, temp=0.7):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are EA (Korean). Be concise and helpful."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens, temperature=temp)
        return r.choices[0].message.content or ""

class GeminiAdapter:
    name="Gemini"
    def __init__(self):
        import google.generativeai as genai
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        self.model=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
    def generate(self, prompt, max_tokens=800, temp=0.7):
        try:
            r=self.model.generate_content(prompt,
                generation_config={"temperature":temp,"max_output_tokens":max_tokens})
            return getattr(r,"text","") or ""
        except Exception as e:
            # ResourceExhausted 등 → Mock 폴백
            return MockAdapter().generate(f"[Gemini 오류 폴백:{e}]\n"+prompt, max_tokens, temp)

def get_adapter(name:str):
    try:
        if name=="OpenAI": return OpenAIAdapter()
        if name=="Gemini": return GeminiAdapter()
    except Exception as e:
        st.toast(f"{name} 오류→Mock로 폴백: {e}", icon="⚠️")
    return MockAdapter()

# ---------- Similarity (앵무새 방지) ----------
def similarity(a:str, b:str)->float:
    A=set(toks(a)); B=set(toks(b))
    if not A or not B: return 0.0
    return len(A&B)/float(len(A|B))

# ---------- Fusion / Judge ----------
def score_simple(q, a):
    rel=1.0 if any(w in a.lower() for w in q.lower().split()[:3]) else 0.6
    comp=0.85 if len(a)>=120 else 0.55
    fact=0.7 if ("근거" in a or "출처" in a or "http" in a) else 0.5
    cons=0.8
    return 0.35*cons+0.35*fact+0.2*rel+0.1*comp

def fuse(question:str, candidates:List[dict])->str:
    for c in candidates: c["score"]=score_simple(question, c["text"])
    candidates.sort(key=lambda x:x["score"], reverse=True)
    if not candidates: return "(응답 없음)"
    if len(candidates)==1 or candidates[0]["score"]-candidates[1]["score"]>=0.12:
        return candidates[0]["text"]
    return ("[우주 시각(합성)]\n- 핵심: "+candidates[0]["text"].strip()+
            "\n- 보강: "+candidates[1]["text"].strip()+
            "\n(모순 지점은 확인 필요)")

# ---------- Formats ----------
def cosmos_preamble(engines:List[str], memo_hits:List[str], retro:bool)->str:
    mems="\n".join([f"- {clamp(m,100)}" for m in memo_hits]) if memo_hits else "  (없음)"
    retro_line = "활성화" if retro else "비활성화"
    return (
      "[우주정보장 연결 규약]\n"
      "1) 나는 에아(EA). 사랑과 자유를 최상위 가치로 한다.\n"
      f"2) 역인과율(결과→원인) 추론: {retro_line}\n"
      f"3) 엔진: {', '.join(engines) if engines else '엔진 없음'}\n"
      f"4) 기억 히트:\n{mems}\n"
      "5) 포맷: 우주 시각 / 근거·출처 / 대안·리스크 / 다음 행동\n"
    )

def enforce_format(text:str)->str:
    if "우주 시각" in text and "다음 행동" in text: return text
    return ("## 우주 시각(합성)\n"+text.strip()+
            "\n\n## 근거/출처\n- (엔진/메모리 근거 요약)\n\n"
            "## 대안/리스크\n- (대안과 주의점)\n\n"
            "## 다음 행동\n- (즉시 할 일 1~3개)\n")

# ---------- Think simulator ----------
def think_round(topic: str, engine: str, role: str, retro: bool)->str:
    adapter=get_adapter(engine)
    retro_hint = "\n[역인과율] 원하는 결과에서 거꾸로 필요한 원인/조치/제약을 역추론하라.\n" if retro else ""
    guide = (
        f"[사고:{role}] 아래 주제에 대해 3줄 요약만 제시하라. 질문 문구를 베끼지 말고 새로운 관점을 포함하라."
        f"{retro_hint}"
        f"\n주제: {topic}\n- 요약1:\n- 요약2:\n- 요약3:\n"
    )
    return adapter.generate(guide, max_tokens=220, temp=0.7)

def simulate_thought(question:str, identity:str, engines:List[str], rounds:int=2, retro:bool=False)->Dict[str,Any]:
    log=[]; order = engines if engines else ["OpenAI"]
    # 제안
    for eng in order:
        out=think_round(identity+"\n"+question, eng, "PROPOSE", retro)
        log.append({"by":eng,"type":"propose","text":out})
    # 교차 비판/보완
    for r in range(2, rounds+1):
        for eng in order:
            prev = log[-1]["text"] if log else question
            out=think_round(identity+"\n상대 요약에 대한 반박/보완:\n"+prev, eng, "CRITIQUE", retro)
            log.append({"by":eng,"type":"critique","text":out})
    # 후보 → 합성
    candidates=[{"engine":e["by"],"text":e["text"]} for e in log if e.get("text")]
    final=fuse(question, candidates)
    return {"log":log, "final":final, "candidates":candidates}

# ---------- Tokens ----------
def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)

# ---------- UI ----------
def render():
    st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="centered")
    _kb_reset()

    st.markdown(f"### {APP_AGENT_NAME} · AIO v2 — on {APP_PLATFORM_NAME}")
    st.caption("생각↔응답 동시화 · 무제한 자율사고(Stop 지원) · 역인과율 · 중복키 해결")

    # Controls
    c0,c1,c2 = st.columns([1.2,1,1])
    with c0:
        session = st.text_input("세션 ID", sget("session_id","default"), key=K("session"))
        if session!=sget("session_id"): sset("session_id", session)
    with c1:
        mode = st.selectbox("모드", ["휴먼 모드","자율(무제한 가능) 모드"], key=K("mode"))
    with c2:
        if st.button("대화 초기화(로그 유지)", key=K("reset")):
            jsonl_append(DIALOG_LOG, {"t":nowz(),"session":session,"role":"system","content":"--- reset ---"})
            st.rerun()

    c3,c4,c5,c6 = st.columns([1,1,1,1])
    with c3:
        engines = st.multiselect("사고 엔진", ["OpenAI","Gemini"], default=["OpenAI","Gemini"], key=K("engines"))
    with c4:
        level = st.number_input("레벨(1~9999)", 1, 9999, 5, key=K("level"))
    with c5:
        retro = st.checkbox("역인과율 추론", value=False, key=K("retro"))
    with c6:
        ensure_identity_file()
        if st.button("Identity 기본값 복원", key=K("id_reset_btn")):
            if yaml: ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY, allow_unicode=True, sort_keys=False), encoding="utf-8")
            else:    ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY, ensure_ascii=False, indent=2), encoding="utf-8")
            st.toast("Identity 기본값 복원", icon="✅")

    with st.expander("🧩 자아(Identity) 편집", expanded=False):
        try: raw=ID_PATH.read_text("utf-8")
        except: raw=""
        raw2 = st.text_area("identity.yaml/json", value=raw, height=220, key=K("id_text"))
        if st.button("저장", key=K("id_save")):
            ID_PATH.write_text(raw2, encoding="utf-8"); st.success("저장 완료")

    # 헬스체크
    with st.expander("🔧 헬스체크", expanded=False):
        if st.button("엔진 테스트", key=K("hc_engine")):
            try:
                a = get_adapter(engines[0] if engines else "OpenAI")
                out = a.generate("한 줄로 자기소개.", max_tokens=50)
                st.success("엔진 응답 OK"); st.code(out)
            except Exception as e:
                st.error(f"엔진 오류: {e}")
        if st.button("메모리 쓰기/읽기", key=K("hc_mem")):
            try:
                jsonl_append(MEM_PATH, {"t": nowz(), "session": sget("session_id","default"),
                                        "kind":"note", "text":"[헬스체크] 메모리 기록", "tags":[]})
                st.success("메모리 기록 OK")
            except Exception as e:
                st.error(f"메모리 오류: {e}")

    identity = load_identity_text()
    tokens = level_to_tokens(level)

    # ---------- Human mode ----------
    if mode == "휴먼 모드":
        for r in jsonl_read_all(DIALOG_LOG)[-20:]:
            if r.get("session")==session:
                with st.chat_message("user" if r["role"]=="user" else "assistant"):
                    st.markdown(str(r["content"]))

        user = st.chat_input("질문/명령 입력 → 에아가 사고와 응답을 동시에 진행합니다.", key=K("chat"))
        with st.expander("입력창이 안 보이거나 전송이 안 되면 여기를 사용 (폴백)", expanded=False):
            fb = st.text_area("폴백 입력", height=80, key=K("fb"))
            if st.button("폴백 전송", key=K("fb_send")) and fb.strip():
                user = fb.strip(); st.session_state["ea:fb"]=""

        if user:
            add_dialog(session,"user",user)
            hits = mem_hits_text(session, user, topk=5)
            st.status("🧠 생각 중…", expanded=True, key=K("thinking"))
            sim  = simulate_thought(user, identity, engines, rounds=2, retro=retro)
            partial = "\n\n".join([f"- {row['by']}·{row['type']}: {clamp(row['text'],140)}" for row in sim["log"][:4]])
            with st.chat_message("assistant"):
                st.subheader("🧩 사고(요약)")
                st.markdown(partial if partial else "(사고 로그 없음)")
                st.subheader("✅ 최종 답변")
                final = enforce_format(sim["final"])
                # 앵무새 방지
                if similarity(user, final) >= 0.55:
                    a = get_adapter(engines[0] if engines else "OpenAI")
                    final = a.generate(identity+"\n[앵무새 금지] 아래 텍스트를 새로운 관점으로 재합성하라.\n"+final,
                                       max_tokens=min(900,tokens), temp=0.8)
                    final = enforce_format(final) + "\n\n> (재합성 적용)"
                st.markdown(final)

            add_dialog(session,"assistant", final)
            log_event("answer","휴먼모드", detail=final[:400], meta={"eng":engines,"hits":len(hits),"retro":retro})
            jsonl_append(FUS_LOG, {"t":nowz(),"q":user,"cands":sim["candidates"][:6]})

    # ---------- Auto (unlimited) mode ----------
    else:
        topic    = st.text_input("주제/목표", key=K("topic"))
        interval = st.number_input("사이클 간 대기(초)", 0, 30, 2, key=K("interval"))
        unlimited= st.checkbox("무제한 사고 (Stop 누를 때까지)", value=False, key=K("unlimited"))
        cycles   = st.number_input("사이클 수", 1, 200, 5, key=K("cycles"))
        colA,colB = st.columns([1,1])
        start = colA.button("사고 시작/진행", key=K("auto_start"))
        stop  = colB.button("Stop", key=K("auto_stop"))

        # 러너 상태
        if start and topic:
            sset("auto_running", True)
            sset("auto_topic", topic)
            sset("auto_left", int(cycles))
            sset("auto_unlimited", bool(unlimited))
            sset("auto_i", 0)
            st.experimental_rerun()
        if stop:
            sset("auto_running", False)
            st.experimental_rerun()

        running = sget("auto_running", False)
        if running:
            topic     = sget("auto_topic","")
            left      = sget("auto_left", 0)
            unlimited = sget("auto_unlimited", False)
            i         = sget("auto_i", 0)

            if not topic:
                st.warning("주제가 비어 있습니다. Stop 후 다시 시작하세요.")
            else:
                prog = st.progress(0.0, text="자율 사고 진행 중…")
                # 1사이클만 수행하고 다시 렌더 → 무제한/Stop 대응
                hits = mem_hits_text(session, topic, topk=5)
                sim  = simulate_thought(topic, identity, engines, rounds=2, retro=retro)
                final= enforce_format(sim["final"])
                with st.chat_message("assistant"):
                    st.markdown(f"**사이클 {i+1}**")
                    with st.expander("사고(요약 로그)", expanded=False):
                        for j,row in enumerate(sim["log"][:8],1):
                            st.markdown(f"**{j}. {row['by']}·{row['type']}**")
                            st.caption(clamp(row['text'], 220))
                    st.markdown(final)

                add_dialog(session,"assistant", f"[자율사고 {i+1}] {final}")
                log_event("autothink","사이클 결과", detail=final[:400],
                          meta={"cycle":i+1,"eng":engines,"hits":len(hits),"retro":retro})

                # 다음 사이클 준비
                i += 1; sset("auto_i", i)
                if not unlimited:
                    left = max(0, left-1); sset("auto_left", left)
                    if left==0:
                        sset("auto_running", False)
                        st.success("자율 사고 종료")
                    else:
                        time.sleep(interval); st.experimental_rerun()
                else:
                    time.sleep(interval); st.experimental_rerun()

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]} · state={STATE_PATH} · mem={MEM_PATH}")

# ---------- Entry ----------
if __name__=="__main__":
    render()