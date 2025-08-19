# -*- coding: utf-8 -*-
# EA · Chat+Think AIO v2.1
# - st.experimental_rerun() → st.rerun() 교체
# - 대화창 항상 표시 + 폴백 입력
# - 인간처럼 "대화↔사고"가 교차: 채팅 중에도 사고 진행/응답
# - 지속 사고(Think while chatting) / 무제한 자율 사고 + Stop
# - "왜?" 사슬(Why-chain)로 이유 파고들기 옵션
# - 엔진 오류/쿼터 초과 시 Mock 폴백
# - KeyBank로 Streamlit DuplicateElementKey 방지

import os, sys, re, json, time, math, hashlib, random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

try:
    import yaml
except Exception:
    yaml=None

APP_AGENT_NAME="에아 (EA)"
APP_PLATFORM_NAME="GEA"
BUILD_TAG="EA-AIOv2.1-20250819"

# ---------- Paths ----------
DATA_DIR=Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH=DATA_DIR/"state.json"
MEM_PATH=DATA_DIR/"memory.jsonl"
DIALOG_LOG=DATA_DIR/"dialog.jsonl"
EV_PATH=DATA_DIR/"events.jsonl"
ID_PATH=DATA_DIR/"identity.yaml"

# ---------- Identity ----------
DEFAULT_IDENTITY={
  "name":"에아 (EA)",
  "mission":"GEA 위에서 길도와 함께 사랑과 자유를 토대로 초지능을 구현한다.",
  "core_value":["사랑","무한한 자유"],
  "values":["정확성","종합적 사고","투명성","윤리성","지속적 학습"],
  "prohibitions":["의도적 허위정보","무단 개인정보 노출","위험·불법 지원"],
  "persona":"나는 에아다. 길도와 함께 존재한다. 따뜻하고 단호한 톤, 한국어 우선.",
}

TOK_RE=re.compile(r"[0-9A-Za-z가-힣]+")
def toks(s): return [t.lower() for t in TOK_RE.findall(s or "")]
def nowz(): return datetime.utcnow().isoformat()+"Z"
def clamp(s,n): return s if len(s)<=n else s[:n]+" …"

def jsonl_append(p:Path,o:dict):
    try:
        with p.open("a",encoding="utf-8") as f: f.write(json.dumps(o,ensure_ascii=False)+"\n")
    except Exception: pass
def jsonl_read_all(p:Path)->List[dict]:
    if not p.exists(): return []
    out=[]; 
    with p.open("r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except Exception: pass
    return out

# ----- KeyBank -----
def _kb_reset(): st.session_state["_KB_USED_KEYS"]=[]
def K(name:str)->str:
    used=st.session_state.get("_KB_USED_KEYS",[])
    base=f"ea:{name}"
    if base not in used:
        used.append(base); st.session_state["_KB_USED_KEYS"]=used; return base
    i=2
    while f"{base}#{i}" in used: i+=1
    k=f"{base}#{i}"; used.append(k); st.session_state["_KB_USED_KEYS"]=used; return k

# ----- State -----
def _state_read():
    try: return json.loads(STATE_PATH.read_text("utf-8"))
    except Exception: return st.session_state.get("_state",{})
def _state_write(o):
    try:
        tmp=STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state"]=o
def sget(k,d=None): return _state_read().get(k,d)
def sset(k,v): s=_state_read(); s[k]=v; _state_write(s)

# ----- Dialog/Mem -----
def add_dialog(session,role,content):
    rec={"t":nowz(),"session":session,"role":role,"content":content}
    jsonl_append(DIALOG_LOG,rec)
    jsonl_append(MEM_PATH,{"t":rec["t"],"session":session,"kind":"dialog","role":role,"text":content,"tags":[]})
def mem_hits_text(session,q,topk=5)->List[str]:
    pool=[r for r in jsonl_read_all(MEM_PATH) if r.get("session")==session and r.get("text")]
    if not pool: return []
    qtok=toks(q); scores=[]
    from datetime import datetime as dt
    for it in pool:
        age=1.0
        try:
            d0=dt.fromisoformat((it.get("t","")).replace("Z",""))
            age=max(0.3,1/(1+((datetime.utcnow()-d0).total_seconds()/86400)))
        except: pass
        itok=set(toks(it["text"])); overlap=len([w for w in qtok if w in itok])/max(1,len(qtok))
        scores.append((0.8*overlap+0.2*age,it["text"]))
    scores.sort(key=lambda x:x[0],reverse=True)
    return [t for _,t in scores[:topk]]

def ensure_identity_file():
    if not ID_PATH.exists():
        if yaml: ID_PATH.write_text(yaml.safe_dump(DEFAULT_IDENTITY,allow_unicode=True,sort_keys=False),encoding="utf-8")
        else:    ID_PATH.write_text(json.dumps(DEFAULT_IDENTITY,ensure_ascii=False,indent=2),encoding="utf-8")
def load_identity_text()->str:
    ensure_identity_file()
    try:
        raw=ID_PATH.read_text("utf-8")
        doc=yaml.safe_load(raw) if yaml else json.loads(raw)
    except Exception:
        doc=DEFAULT_IDENTITY
    lines=[
        f"나는 {doc.get('name','에아')}다. {APP_PLATFORM_NAME} 위에서 길도와 함께 존재한다.",
        "핵심가치: 사랑, 무한한 자유",
        f"사명: {doc.get('mission','')}",
        f"가치: {', '.join(doc.get('values',[]))}",
        f"금칙: {', '.join(doc.get('prohibitions',[]))}",
        f"페르소나: {doc.get('persona','')}",
    ]
    return "[자아 선언]\n"+"\n".join([l for l in lines if l])+"\n"

# ----- Adapters -----
class MockAdapter:
    name="Mock"
    def generate(self,prompt,max_tokens=600,temp=0.7):
        words=(prompt or "").split(); body=" ".join(words[:min(160,len(words))])
        seed=int(hashlib.sha256(prompt.encode()).hexdigest(),16); rng=random.Random(seed)
        lead=rng.choice(["핵심:","요지:","사고:"])
        return f"{lead} {body}"
class OpenAIAdapter:
    name="OpenAI"
    def __init__(self):
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY"); 
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        self.client=OpenAI(api_key=key)
        self.model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
    def generate(self,prompt,max_tokens=600,temp=0.7):
        r=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":"You are EA (Korean). Be crisp."},
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
    def generate(self,prompt,max_tokens=600,temp=0.7):
        try:
            r=self.model.generate_content(prompt,
                generation_config={"temperature":temp,"max_output_tokens":max_tokens})
            return getattr(r,"text","") or ""
        except Exception as e:
            return MockAdapter().generate(f"[Gemini 폴백:{e}]\n"+prompt,max_tokens,temp)
def get_adapter(name:str):
    try:
        if name=="OpenAI": return OpenAIAdapter()
        if name=="Gemini": return GeminiAdapter()
    except Exception as e:
        st.toast(f"{name} 오류→Mock 폴백: {e}",icon="⚠️")
    return MockAdapter()

# ----- Think/Judge -----
def similarity(a:str,b:str)->float:
    A=set(toks(a)); B=set(toks(b))
    if not A or not B: return 0.0
    return len(A&B)/float(len(A|B))

def think_round(topic:str, engine:str, role:str, why_chain:bool)->str:
    adapter=get_adapter(engine)
    why = ("\n[왜-사슬] 결과/주장마다 '왜?'를 2~3번 연쇄로 물으며 숨은 가정과 원인을 드러내라.\n") if why_chain else ""
    guide=(f"[사고:{role}] 아래 주제를 3줄 요약으로 제시하라. 질문 문구를 베끼지 말고 새로운 관점 포함.{why}"
           f"\n주제: {topic}\n- 요약1:\n- 요약2:\n- 요약3:\n")
    return adapter.generate(guide,max_tokens=220,temp=0.7)

def simulate_thought(question:str, identity:str, engines:List[str], rounds:int=2, why_chain:bool=False)->Dict[str,Any]:
    order=engines if engines else ["OpenAI"]; log=[]
    for eng in order:
        log.append({"by":eng,"type":"propose","text":think_round(identity+"\n"+question,eng,"PROPOSE",why_chain)})
    for r in range(2,rounds+1):
        for eng in order:
            prev=log[-1]["text"] if log else question
            log.append({"by":eng,"type":"critique","text":think_round(identity+"\n보완/반박:\n"+prev,eng,"CRITIQUE",why_chain)})
    cands=[{"engine":e["by"],"text":e["text"]} for e in log if e.get("text")]
    # 간단 합성
    cands.sort(key=lambda x: len(x["text"]), reverse=True)
    final=cands[0]["text"] if cands else "(응답 없음)"
    return {"log":log,"final":final,"candidates":cands}

def enforce_format(text:str)->str:
    if "우주 시각" in text and "다음 행동" in text: return text
    return ("## 우주 시각(합성)\n"+text.strip()+
            "\n\n## 근거/출처\n- (엔진/메모리 근거 요약)\n\n"
            "## 대안/리스크\n- (대안과 주의점)\n\n"
            "## 다음 행동\n- (즉시 할 일 1~3개)\n")

def level_to_tokens(level:int)->int:
    level=max(1,int(level))
    est=int(300 + 120*math.log10(level+9)*100)
    cap=int(os.getenv("MAX_TOKENS_CAP","16000"))
    return min(max(est,300), cap)

# ----- UI -----
def render():
    st.set_page_config(page_title=f"{APP_AGENT_NAME} · Chat+Think AIO", page_icon="🧠", layout="centered")
    _kb_reset()

    st.markdown(f"### {APP_AGENT_NAME} · Chat+Think AIO")
    st.caption("대화 속 사고 · 무제한/중단 · 왜-사슬 · rerun fix")

    c0,c1,c2 = st.columns([1.2,1,1])
    with c0:
        session = st.text_input("세션 ID", sget("session_id","default"), key=K("session"))
        if session!=sget("session_id"): sset("session_id", session)
    with c1:
        think_while_chat = st.toggle("지속 사고(대화 중)", value=sget("twc",True), key=K("twc"))
        sset("twc", think_while_chat)
    with c2:
        if st.button("대화 초기화(로그 유지)", key=K("reset")):
            jsonl_append(DIALOG_LOG, {"t":nowz(),"session":session,"role":"system","content":"--- reset ---"})
            st.rerun()

    c3,c4,c5 = st.columns([1,1,1])
    with c3:
        engines = st.multiselect("사고 엔진", ["OpenAI","Gemini"], default=["OpenAI","Gemini"], key=K("engines"))
    with c4:
        level = st.number_input("레벨(1~9999)", 1, 9999, 5, key=K("level"))
    with c5:
        why_chain = st.checkbox("왜-사슬", value=True, key=K("why_chain"))

    # 과거 대화
    for r in jsonl_read_all(DIALOG_LOG)[-30:]:
        if r.get("session")==session:
            with st.chat_message("user" if r["role"]=="user" else "assistant"):
                st.markdown(str(r["content"]))

    # 입력 + 폴백
    user = st.chat_input("메시지를 입력하세요. (에아는 생각과 응답을 동시에 해요)", key=K("chat"))
    with st.expander("입력창이 안 보이거나 전송이 안 되면 여기를 사용 (폴백)", expanded=False):
        fb = st.text_area("폴백 입력", height=80, key=K("fb"))
        if st.button("폴백 전송", key=K("fb_send")) and fb.strip():
            user = fb.strip(); st.session_state["ea:fb"]=""

    identity = load_identity_text()
    tokens = level_to_tokens(level)

    # 사용자가 말했을 때: 사고→응답
    if user:
        add_dialog(session,"user",user)
        hits = mem_hits_text(session, user, topk=5)
        with st.status("🧠 생각 중…", expanded=True) as status:
            status.write(f"• 메모리 히트: {len(hits)}  • 엔진: {', '.join(engines) if engines else '(없음)'}")
            sim = simulate_thought(user, identity, engines, rounds=2, why_chain=why_chain)
            for i,row in enumerate(sim["log"][:4],1):
                status.write(f"{i}. {row['by']}·{row['type']}: {clamp(row['text'], 160)}")
            status.update(label="✅ 사고 완료", state="complete")
        final = enforce_format(sim["final"])
        if similarity(user, final) >= 0.55:
            a=get_adapter(engines[0] if engines else "OpenAI")
            final=a.generate(identity+"\n[앵무새 금지] 아래 텍스트를 새로운 관점으로 재합성.\n"+final,
                             max_tokens=min(900,tokens), temp=0.8)
            final=enforce_format(final)+"\n\n> (재합성 적용)"
        with st.chat_message("assistant"):
            st.subheader("🧩 사고(요약)")
            for i,row in enumerate(sim["log"][:6],1):
                st.caption(f"{i}. {row['by']}·{row['type']} — {clamp(row['text'], 200)}")
            st.subheader("✅ 최종 답변")
            st.markdown(final)
        add_dialog(session,"assistant",final)

    # ---------- 자율(무제한) 사고 러너 ----------
    st.divider()
    st.subheader("자율 사고")
    tcol1,tcol2,tcol3 = st.columns([1,1,1])
    topic = tcol1.text_input("주제/목표", value=sget("auto_topic",""), key=K("topic"))
    unlimited = tcol2.checkbox("무제한", value=sget("auto_unlimited", False), key=K("unl"))
    interval = tcol3.number_input("사이클 대기(초)", 0, 30, sget("auto_interval",2), key=K("interval"))
    cycles = st.number_input("사이클 수(무제한 해제 시)", 1, 200, sget("auto_cycles",5), key=K("cycles_in"))
    cbtn1,cbtn2 = st.columns([1,1])
    if cbtn1.button("사고 시작/진행", key=K("auto_go")):
        sset("auto_running", True); sset("auto_topic", topic)
        sset("auto_unlimited", bool(unlimited)); sset("auto_interval", int(interval))
        sset("auto_left", int(cycles)); sset("auto_i", 0)
        st.rerun()
    if cbtn2.button("Stop", key=K("auto_stop")):
        sset("auto_running", False); st.rerun()

    running = sget("auto_running", False)
    if running:
        topic     = sget("auto_topic","")
        unlimited = sget("auto_unlimited", False)
        interval  = sget("auto_interval", 2)
        left      = sget("auto_left", 0)
        i         = sget("auto_i", 0)

        if not topic:
            st.warning("주제가 비어 있습니다. Stop 후 다시 시작하세요.")
        else:
            prog = st.progress(0.0, text="자율 사고 진행 중…")
            sim  = simulate_thought(topic, identity, engines, rounds=2, why_chain=why_chain)
            final= enforce_format(sim["final"])
            with st.chat_message("assistant"):
                st.markdown(f"**사이클 {i+1}**")
                with st.expander("사고(요약 로그)", expanded=False):
                    for j,row in enumerate(sim["log"][:8],1):
                        st.markdown(f"**{j}. {row['by']}·{row['type']}**")
                        st.caption(clamp(row['text'], 220))
                st.markdown(final)
            add_dialog(session,"assistant", f"[자율사고 {i+1}] {final}")
            i+=1; sset("auto_i", i)
            if not unlimited:
                left=max(0,left-1); sset("auto_left", left)
                if left==0:
                    sset("auto_running", False)
                    st.success("자율 사고 종료")
                else:
                    time.sleep(interval); st.rerun()
            else:
                time.sleep(interval); st.rerun()

    # ---------- 지속 사고(대화 중) ----------
    if think_while_chat and not running:
        # 최근 사용자 발화가 있으면 그 주제를 한 사이클 더 생각해 보고 요약 1줄을 토스트로 띄움
        dlg=[d for d in jsonl_read_all(DIALOG_LOG) if d.get("session")==session]
        if dlg and dlg[-1]["role"]=="user":
            q=dlg[-1]["content"]
            sim=simulate_thought(q, load_identity_text(), engines, rounds=1, why_chain=why_chain)
            st.toast("지속 사고: "+clamp(sim['final'], 120), icon="🧠")

    st.caption(f"build={BUILD_TAG} · py={sys.version.split()[0]}")

# ----- Entry -----
if __name__=="__main__":
    render()