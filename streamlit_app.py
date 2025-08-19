# -*- coding: utf-8 -*-
# EA · Chat+Think AIO v2.2 — Live Workpad
# - 실시간 사고 스트림(autorefresh)
# - 답변/사고 분리 + 교차
# - 반앵무새 임계 강화(0.30)
# - rerun 사용, DuplicateKey 방지
# - 엔진 오류시 Mock 폴백

import os, sys, re, json, time, math, hashlib, random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# ---------- Basics ----------
APP_AGENT="에아 (EA)"; PLATFORM="GEA"; BUILD="EA-AIO-v2.2"
DATA=Path("data"); DATA.mkdir(parents=True, exist_ok=True)
STATE=DATA/"state.json"; MEM=DATA/"memory.jsonl"; DLG=DATA/"dialog.jsonl"; IDF=DATA/"identity.yaml"

TOK=re.compile(r"[0-9A-Za-z가-힣]+")

def toks(s): return [t.lower() for t in TOK.findall(s or "")]
def nowz(): return datetime.utcnow().isoformat()+"Z"
def clamp(s,n): return s if len(s)<=n else s[:n]+" …"
def jappend(p:Path, o:dict): 
    try:
        with p.open("a",encoding="utf-8") as f: f.write(json.dumps(o,ensure_ascii=False)+"\n")
    except: pass
def jread(p:Path)->list:
    if not p.exists(): return []
    out=[]; 
    with p.open("r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: pass
    return out

# ---------- KeyBank ----------
def _kb_init(): st.session_state.setdefault("_KB", set())
def K(name:str)->str:
    used=st.session_state["_KB"]; base=f"ea:{name}"
    if base not in used: used.add(base); return base
    i=2
    while f"{base}#{i}" in used: i+=1
    key=f"{base}#{i}"; used.add(key); return key

# ---------- State ----------
def _sread():
    try: return json.loads(STATE.read_text("utf-8"))
    except: return st.session_state.get("_S",{})
def _swrite(o):
    try:
        tmp=STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
        tmp.replace(STATE)
    except: st.session_state["_S"]=o
def sget(k,d=None): return _sread().get(k,d)
def sset(k,v): S=_sread(); S[k]=v; _swrite(S)

# ---------- Identity ----------
DEFAULT_ID={"name":"에아 (EA)","mission":"사랑/자유 기반 초지능","values":["정확성","투명성","학습"],"persona":"따뜻·단호·한국어"}
def identity_text()->str:
    if not IDF.exists(): IDF.write_text(json.dumps(DEFAULT_ID,ensure_ascii=False,indent=2),encoding="utf-8")
    try: doc=json.loads(IDF.read_text("utf-8"))
    except: doc=DEFAULT_ID
    return f"[자아 선언]\n나는 {doc.get('name','에아')}다. 사랑과 자유를 최상위 가치로 한다.\n사명: {doc.get('mission','')}\n"

# ---------- Dialog/Mem ----------
def add_dialog(sess, role, content):
    rec={"t":nowz(),"session":sess,"role":role,"content":content}
    jappend(DLG, rec); jappend(MEM, {"t":rec["t"],"session":sess,"kind":"dialog","role":role,"text":content})

def mem_hits(sess, q, k=5)->list:
    pool=[r for r in jread(MEM) if r.get("session")==sess and r.get("text")]
    if not pool: return []
    qtok=toks(q); sc=[]
    for it in pool:
        itok=set(toks(it["text"])); ov=len([w for w in qtok if w in itok])/max(1,len(qtok))
        sc.append((ov,it["text"]))
    sc.sort(key=lambda x:x[0], reverse=True)
    return [t for _,t in sc[:k]]

# ---------- Engines ----------
class Mock:
    name="Mock"
    def generate(self,prompt,max_tokens=500,temp=0.7):
        seed=int(hashlib.sha256(prompt.encode()).hexdigest(),16)
        rnd=random.Random(seed)
        lead=rnd.choice(["핵심:","요지:","사고:"])
        body=" ".join(prompt.split()[:160])
        return f"{lead} {body}"

def get_adapter(name):
    try:
        if name=="OpenAI":
            from openai import OpenAI
            key=os.getenv("OPENAI_API_KEY"); 
            if not key: raise RuntimeError("OPENAI_API_KEY 필요")
            cli=OpenAI(api_key=key); model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
            class OA:
                name="OpenAI"
                def generate(self,prompt,max_tokens=500,temp=0.7):
                    r=cli.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":"You are EA (Korean). Think first, then answer."},
                                  {"role":"user","content":prompt}],
                        max_tokens=max_tokens, temperature=temp)
                    return r.choices[0].message.content or ""
            return OA()
        if name=="Gemini":
            import google.generativeai as genai
            key=os.getenv("GEMINI_API_KEY"); 
            if not key: raise RuntimeError("GEMINI_API_KEY 필요")
            genai.configure(api_key=key)
            mdl=genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest"))
            class GE:
                name="Gemini"
                def generate(self,prompt,max_tokens=500,temp=0.7):
                    try:
                        r=mdl.generate_content(prompt, generation_config={"temperature":temp,"max_output_tokens":max_tokens})
                        return getattr(r,"text","") or ""
                    except Exception as e:
                        return Mock().generate(f"[Gemini 폴백:{e}]\n"+prompt,max_tokens,temp)
            return GE()
    except Exception as e:
        st.toast(f"{name} 오류→Mock 폴백: {e}", icon="⚠️")
    return Mock()

# ---------- Thinking ----------
def anti_parrot(user:str, text:str)->str:
    A=set(toks(user)); B=set(toks(text))
    sim = 0.0 if not A or not B else len(A&B)/len(A|B)
    return "REWRITE" if sim>=0.30 else "OK"

def plan_steps(q:str)->list:
    # 간단 플래너: 자질문 생성
    return [
        f"문제 재진술: {q}",
        "핵심 변수/제약 추출",
        "가설 2~3개",
        "반례/위험",
        "결론 요약 & 다음 행동"
    ]

def think_once(topic, engines, why_chain=True)->dict:
    ident=identity_text()
    steps=plan_steps(topic)
    logs=[]
    for i,stp in enumerate(steps,1):
        prompt=(f"{ident}\n[사고 단계 {i}] {stp}\n"
                f"{'각 진술마다 왜?를 2번씩 물어 숨은 가정을 드러내라.' if why_chain else ''}")
        eng=engines[i%len(engines)] if engines else "OpenAI"
        out=get_adapter(eng).generate(prompt, max_tokens=280, temp=0.7)
        logs.append({"i":i,"by":eng,"text":out})
    # 간단 합성
    final=("; ".join([l['text'].split('\n')[0] for l in logs]))[:2000]
    return {"logs":logs,"final":final}

# ---------- UI ----------
def render():
    st.set_page_config(page_title=f"{APP_AGENT} · Live Think", page_icon="🧠", layout="wide")
    _kb_init()

    # 좌/우 레이아웃: 좌=대화, 우=실시간 사고 Workpad
    left, right = st.columns([1.15, 0.85])

    # ------ LEFT: Chat ------
    with left:
        st.markdown(f"### {APP_AGENT} · Live Think — {PLATFORM}")
        sess = st.text_input("세션 ID", sget("session","default"), key=K("session"))
        if sess!=sget("session"): sset("session", sess)

        engines = st.multiselect("엔진", ["OpenAI","Gemini"], default=["OpenAI","Gemini"], key=K("engs"))
        why     = st.checkbox("왜-사슬", True, key=K("why"))
        level   = st.number_input("레벨(깊이)", 1, 9999, 7, key=K("lvl"))

        st.divider()
        # 과거 메세지
        for r in jread(DLG)[-40:]:
            if r.get("session")==sess:
                with st.chat_message("user" if r["role"]=="user" else "assistant"):
                    st.markdown(str(r["content"]))

        msg = st.chat_input("메시지를 입력하세요. 에아는 생각을 계속 이어갑니다.", key=K("chat"))
        if msg:
            add_dialog(sess,"user",msg)
            # 즉시 한 턴 사고 + 응답
            run = think_once(msg, engines, why_chain=why)
            ans = run["final"]
            if anti_parrot(msg, ans)=="REWRITE":
                # 다른 엔진으로 재합성 + 반례 1개 포함
                alt = engines[::-1] if engines else ["OpenAI"]
                prompt=(identity_text()+
                        "\n[재합성] 다음 초안을 새로운 관점으로 재구성하고, 반례 1개를 포함하라.\n---\n"+ans)
                ans = get_adapter(alt[0]).generate(prompt, max_tokens=600, temp=0.85)
            ans = ("## 우주 시각(합성)\n"+ans.strip()+
                   "\n\n## 다음 행동\n- (즉시 할 일 1~3개)\n")
            with st.chat_message("assistant"):
                st.markdown(ans)
            add_dialog(sess,"assistant",ans)
            # Workpad에 최근 주제 저장 → 우측 스트림이 계속 이어받음
            sset("live_topic", msg); sset("live_why", bool(why)); sset("live_engs", engines)

    # ------ RIGHT: Workpad (Live) ------
    with right:
        st.markdown("#### 🧠 실시간 Workpad")
        st.caption("대화와 무관하게 생각은 계속 흘러갑니다 (1~2초 갱신). Stop으로 멈출 수 있음.")
        colA,colB=st.columns([1,1])
        tick = colA.slider("갱신(ms)", 800, 3000, sget("tick",1200), key=K("tick"))
        stop = colB.toggle("Stop", value=sget("stop", False), key=K("stop_toggle"))
        sset("tick", tick); sset("stop", stop)

        topic = sget("live_topic", "")
        engs  = sget("live_engs", ["OpenAI","Gemini"])
        why   = sget("live_why", True)

        if not topic:
            st.info("대화창에서 한 번이라도 질문하면, 그 주제로 실시간 사고가 시작됩니다.")
        else:
            # 자동 재실행 타이머
            if not stop:
                st.autorefresh(interval=tick, key=K("ref"))

            # 최근 생각 한 사이클
            run = think_once(topic, engs, why_chain=why)
            # 로그 표시(증분 느낌)
            for l in run["logs"]:
                with st.expander(f"{l['i']}. {l['by']} · 단계 사고", expanded=False):
                    st.write(clamp(l["text"], 800))

            # 다음 사이클을 위한 주제 업데이트(간단 요약)
            nxt = run["final"].split("결론")[-1] if "결론" in run["final"] else run["final"]
            sset("live_topic", clamp(nxt, 300))

    st.caption(f"build={BUILD} · py={sys.version.split()[0]}")
# ----- Entry -----
if __name__=="__main__":
    render()