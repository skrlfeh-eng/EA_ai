# -*- coding: utf-8 -*-
# EA · Ultra (Streamlit AIO) v3.3
# - ChatGPT 유사 UI(st.chat_message/chat_input)
# - 엔진(OpenAI/Gemini) 실패/쿼터 초과 시 Mock로 자동 폴백
# - 사고 로그(왜-사슬), 반앵무새, 세션 메모리
# - 응답 보장 패치: 어떤 경우에도 좌측 말풍선에 답 출력

import os, re, json, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Generator

import streamlit as st

# ---------------------- 경로/파일 ----------------------
ROOT = Path(".")
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True, parents=True)
DLG  = DATA / "dialog.jsonl"; MEM = DATA / "memory.jsonl"; IDF = DATA / "identity.json"

def nowz() -> str: return datetime.utcnow().isoformat()+"Z"
def jappend(p:Path,obj:Dict):
    try:
        with p.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    except: pass
def jread_lines(p:Path)->List[Dict]:
    if not p.exists(): return []
    out=[]
    with p.open("r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: pass
    return out

TOK=re.compile(r"[0-9A-Za-z가-힣]+")
def toks(s:str)->List[str]: return [t.lower() for t in TOK.findall(s or "")]
def sim(a:str,b:str)->float:
    A,B=set(toks(a)),set(toks(b))
    return 0.0 if not A or not B else len(A&B)/len(A|B)

# ---------------------- 자아/메모리 ----------------------
DEFAULT_ID={"name":"에아 (EA)","mission":"사랑과 자유를 최상위 가치로 삼아 사람과 함께 성장하는 지성","values":["정확성","투명성","학습","윤리"]}
def identity_text()->str:
    if not IDF.exists(): IDF.write_text(json.dumps(DEFAULT_ID,ensure_ascii=False,indent=2),encoding="utf-8")
    try: doc=json.loads(IDF.read_text("utf-8"))
    except: doc=DEFAULT_ID
    return f"[자아 선언]\n나는 {doc.get('name','에아')}다. 사명: {doc.get('mission','')}\n가치: {', '.join(doc.get('values',[]))}\n"

def add_dialog(session_id:str,role:str,content:str):
    rec={"t":nowz(),"session":session_id,"role":role,"content":content}
    jappend(DLG,rec)
    if role in ("user","assistant"): jappend(MEM,{"t":rec["t"],"session":session_id,"kind":"dialog","text":content})

def mem_hits(session_id:str,query:str,k:int=5)->List[str]:
    pool=[r.get("text","") for r in jread_lines(MEM) if r.get("session")==session_id]
    q=set(toks(query)); scored=[]
    for t in pool:
        T=set(toks(t))
        if not T or not q: continue
        scored.append((len(q&T)/len(q|T),t))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [t for _,t in scored[:k]]

# ---------------------- 어댑터 ----------------------
class MockAdapter:
    name="Mock"
    def stream(self,prompt:str,max_tokens:int=420,temperature:float=0.7)->Generator[str,None,None]:
        txt="요지: "+ " ".join(prompt.split()[:150])
        for ch in re.findall(r".{1,60}", txt, flags=re.S):
            yield ch; time.sleep(0.01)

def get_openai_adapter():
    try:
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
        cli=OpenAI(api_key=key)
        class OA:
            name="OpenAI"
            def stream(self,prompt,max_tokens=600,temperature=0.7):
                resp=cli.chat.completions.create(
                    model=model, stream=True, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role":"system","content":"You are EA (Korean). Think first, then answer clearly."},
                              {"role":"user","content":prompt}]
                )
                for ch in resp:
                    delta=ch.choices[0].delta
                    if getattr(delta,"content",None): yield delta.content
        return OA()
    except Exception:
        return None

def get_gemini_adapter():
    try:
        import google.generativeai as genai
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        model=os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest")
        mdl=genai.GenerativeModel(model)
        class GE:
            name="Gemini"
            def stream(self,prompt,max_tokens=480,temperature=0.75):
                r=mdl.generate_content(prompt, generation_config={"temperature":temperature,"max_output_tokens":max_tokens})
                txt=getattr(r,"text","") or ""
                for chunk in re.findall(r".{1,60}", txt, flags=re.S): yield chunk
        return GE()
    except Exception:
        return None

def pick_adapter(order:List[str]):
    for name in order:
        if name.lower().startswith("openai"):
            a=get_openai_adapter()
            if a: return a
        if name.lower().startswith("gemini"):
            a=get_gemini_adapter()
            if a: return a
    return MockAdapter()

# 안전 스트림 래퍼: 실패 시 Mock 폴백 + 사유 출력
def safe_stream(adapter, prompt:str, max_tokens:int, temperature:float)->Generator[str,None,None]:
    try:
        for x in adapter.stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yield x
    except Exception as e:
        note=f"[{adapter.name} 오류:{type(e).__name__}] 자동 폴백 → Mock\n"
        for ch in note: yield ch
        for x in MockAdapter().stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yield x

# ---------------------- 사고/응답 ----------------------
def plan_steps(q:str)->List[str]:
    return [
        "문제 재진술 및 핵심 변수 식별",
        "자질문 2~3개 생성 (각 항목마다 왜?를 2번씩 물어 가정 드러내기)",
        "가설/아이디어 후보",
        "반례/위험/제약",
        "임시 결론 요약"
    ]

def think_round(topic:str, engines:List[str], why_chain:bool, hits:List[str])->Dict:
    ident=identity_text()
    guide=ident + (f"메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "")
    logs=[]
    steps=plan_steps(topic)
    for i,step in enumerate(steps,1):
        eng = engines[(i-1) % max(1,len(engines))] if engines else "OpenAI"
        adapter = pick_adapter([eng])
        prompt=(f"{guide}\n[사고 단계 {i}] {step}\n"
                f"{'각 주장마다 왜?를 2번씩 연쇄로 물어 숨은 가정을 드러내라.' if why_chain else ''}\n"
                f"주제: {topic}\n- 요약:")
        text="".join(safe_stream(adapter, prompt, max_tokens=240, temperature=0.7))
        logs.append({"i":i,"by":adapter.name,"text":text})
    # 최종 합성
    adapter = pick_adapter(engines or ["OpenAI","Gemini"])
    fusion_prompt=(f"{guide}\n[최종합성] 위 단계 요약을 통합해 한국어로 "
                   f"'결론/근거/대안/다음 행동(1~3개)'을 간결히.")
    fusion="".join(safe_stream(adapter, fusion_prompt, max_tokens=560, temperature=0.75))
    return {"logs":logs,"final":fusion}

def compose_answer(user_text:str, engines:List[str], why_chain:bool, session_id:str):
    hits=mem_hits(session_id, user_text, 3)
    round_out=think_round(user_text, engines, why_chain, hits)
    fusion=round_out["final"]
    if sim(user_text, fusion) >= 0.30:
        adapter=pick_adapter(engines[::-1] or ["Gemini","OpenAI"])
        prompt = identity_text() + (f"\n메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "") + \
                 "\n[재합성] 질문 문구 반복 금지, 새로운 관점/반례 1개 포함."
        fusion="".join(safe_stream(adapter, prompt, max_tokens=560, temperature=0.85))
    answer="## 우주 시각(합성)\n"+fusion.strip()+"\n\n## 다음 행동\n- (즉시 할 일 1~3개)\n"
    return answer, round_out["logs"]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="EA · Ultra (AIO)", page_icon="🧠", layout="wide")
if "_k" not in st.session_state: st.session_state["_k"]=0
def K(p:str)->str:
    st.session_state["_k"]+=1; return f"{p}-{st.session_state['_k']}"

st.title("EA · Ultra (AIO) — Chat + Live Thinking")

cols = st.columns([1,1,1,1,2])
session_id = cols[0].text_input("세션 ID", st.session_state.get("session_id","default"), key=K("sid"))
st.session_state["session_id"]=session_id
engines = cols[1].text_input("엔진(콤마)", st.session_state.get("engines","OpenAI,Gemini"), key=K("eng"))
st.session_state["engines"]=engines
why_chain = cols[2].checkbox("왜-사슬", True, key=K("why"))
mem_on    = cols[3].toggle("Memory ON", True, key=K("mem"))

left, right = st.columns([1.1,0.9])

with left:
    st.caption("좌측: 대화창(스트리밍 응답). ChatGPT와 유사한 말풍선 UI.")
    if "messages" not in st.session_state: st.session_state["messages"]=[]
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    user_msg = st.chat_input("메시지를 입력하고 Enter…")
    if user_msg:
        # 사용자 말풍선 + 기록
        with st.chat_message("user"): st.markdown(user_msg)
        st.session_state["messages"].append({"role":"user","content":user_msg})
        if mem_on: add_dialog(session_id, "user", user_msg)

        # 안전 응답 생성(예외/빈응답 방어)
        try:
            answer_text, logs = compose_answer(
                user_msg,
                [s.strip() for s in engines.split(",") if s.strip()],
                why_chain,
                session_id
            )
        except Exception as e:
            warn = f"⚠️ 응답 생성 중 예외({type(e).__name__}). Mock로 폴백합니다.\n"
            mock = "요지: " + " ".join((identity_text()+user_msg).split()[:80])
            answer_text = warn + mock
            logs = [{"i":0,"by":"Mock","text":warn}]

        if not (answer_text or "").strip():
            answer_text = "※ 엔진 응답이 비었습니다. 키/쿼터 확인 요망. 임시 요약 표시.\n" \
                          "요지: " + " ".join(user_msg.split()[:50])

        # 좌측 말풍선에 반드시 출력(토막 스트림 느낌)
        with st.chat_message("assistant"):
            ph = st.empty(); shown=""
            for chunk in re.findall(r".{1,70}", answer_text, flags=re.S):
                shown += chunk; ph.markdown(shown); time.sleep(0.01)
            ph.markdown(shown)

        # 상태/메모리 갱신 & 오른쪽 사고 로그
        st.session_state["messages"].append({"role":"assistant","content":answer_text})
        if mem_on: add_dialog(session_id, "assistant", answer_text)
        st.session_state["last_logs"]=logs

with right:
    st.caption("우측: 사고 로그(단계별). 사람처럼 '왜?'를 캐며 진행.")
    logs = st.session_state.get("last_logs", [])
    if not logs: st.info("대화하면 여기 사고 단계가 나타납니다.")
    else:
        for l in logs:
            with st.expander(f"{l['i']}. {l['by']} · 단계 사고", expanded=False):
                st.markdown(l["text"])

st.divider()
st.caption("키가 없거나 쿼터 초과 시 자동 폴백(Mock) · build v3.3")


# -*- coding: utf-8 -*-
# EA · Ultra (AIO) v3.6
# - 생각(요약) TypeError 방지: 항상 문자열로 캐스팅
# - 자동 사고(무한) : st.autorefresh 기반 주기 실행(사용자 미입력 상태에서도 지속)
# - 위젯 key 전부 고유화(중복 ID 방지)
# - OpenAI/Gemini 자동 선택 + 폴백(Mock)

import os, re, json, time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Generator, Tuple

import streamlit as st

# ---------------------- 경로/파일 ----------------------
ROOT = Path(".")
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True, parents=True)
DLG  = DATA / "dialog.jsonl"; MEM = DATA / "memory.jsonl"; IDF = DATA / "identity.json"

def nowz() -> str: return datetime.utcnow().isoformat()+"Z"
def jappend(p:Path,obj:Dict):
    try:
        with p.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=False)+"\n")
    except: pass
def jread_lines(p:Path)->List[Dict]:
    if not p.exists(): return []
    out=[]
    with p.open("r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: pass
    return out

TOK=re.compile(r"[0-9A-Za-z가-힣]+")
def toks(s:str)->List[str]: return [t.lower() for t in TOK.findall(s or "")]
def sim(a:str,b:str)->float:
    A,B=set(toks(a)),set(toks(b))
    return 0.0 if not A or not B else len(A&B)/len(A|B)

# ---------------------- 자아/메모리 ----------------------
DEFAULT_ID={"name":"에아 (EA)","mission":"사랑과 자유를 최상위 가치로 삼아 사람과 함께 성장하는 지성","values":["정확성","투명성","학습","윤리"]}
def identity_text()->str:
    if not IDF.exists(): IDF.write_text(json.dumps(DEFAULT_ID,ensure_ascii=False,indent=2),encoding="utf-8")
    try: doc=json.loads(IDF.read_text("utf-8"))
    except: doc=DEFAULT_ID
    return f"[자아 선언]\n나는 {doc.get('name','에아')}다. 사명: {doc.get('mission','')}\n가치: {', '.join(doc.get('values',[]))}\n"

def add_dialog(session_id:str,role:str,content:str):
    rec={"t":nowz(),"session":session_id,"role":role,"content":content}
    jappend(DLG,rec)
    if role in ("user","assistant"): jappend(MEM,{"t":rec["t"],"session":session_id,"kind":"dialog","text":content})

def mem_hits(session_id:str,query:str,k:int=5)->List[str]:
    pool=[r.get("text","") for r in jread_lines(MEM) if r.get("session")==session_id]
    q=set(toks(query)); scored=[]
    for t in pool:
        T=set(toks(t))
        if not T or not q: continue
        scored.append((len(q&T)/len(q|T),t))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [t for _,t in scored[:k]]

# ---------------------- 어댑터 ----------------------
class MockAdapter:
    name="Mock"
    def stream(self,prompt:str,max_tokens:int=420,temperature:float=0.7)->Generator[str,None,None]:
        txt="요지: "+ " ".join(prompt.split()[:150])
        for ch in re.findall(r".{1,60}", txt, flags=re.S):
            yield ch; time.sleep(0.01)

def get_openai_adapter():
    try:
        from openai import OpenAI
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
        cli=OpenAI(api_key=key)
        class OA:
            name="OpenAI"
            def stream(self,prompt,max_tokens=600,temperature=0.7):
                resp=cli.chat.completions.create(
                    model=model, stream=True, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role":"system","content":"You are EA (Korean). Think first, then answer clearly."},
                              {"role":"user","content":prompt}]
                )
                for ch in resp:
                    delta=ch.choices[0].delta
                    if getattr(delta,"content",None): yield delta.content
        return OA()
    except Exception:
        return None

GEMINI_CANDIDATES = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]
def get_gemini_adapter():
    try:
        import google.generativeai as genai
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY 필요")
        genai.configure(api_key=key)
        model=os.getenv("GEMINI_MODEL","") or GEMINI_CANDIDATES[0]
        def build(mname:str):
            mdl=genai.GenerativeModel(mname)
            class GE:
                name=f"Gemini({mname})"
                def stream(self,prompt,max_tokens=480,temperature=0.75):
                    r=mdl.generate_content(prompt,
                        generation_config={"temperature":temperature,"max_output_tokens":max_tokens})
                    txt=getattr(r,"text","") or ""
                    for chunk in re.findall(r".{1,60}", txt, flags=re.S): yield chunk
            return GE()
        try: return build(model)
        except: pass
        for cand in GEMINI_CANDIDATES:
            try: return build(cand)
            except: pass
        return None
    except Exception:
        return None

def pick_adapter(order:List[str]):
    for name in order:
        if name.lower().startswith("openai"):
            a=get_openai_adapter()
            if a: return a
        if name.lower().startswith("gemini"):
            a=get_gemini_adapter()
            if a: return a
    return MockAdapter()

def safe_stream(adapter, prompt:str, max_tokens:int, temperature:float)->Generator[str,None,None]:
    try:
        for x in adapter.stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yield x
    except Exception as e:
        note=f"[{adapter.name} 오류:{type(e).__name__}] 폴백 → Mock\n"
        for ch in note: yield ch
        for x in MockAdapter().stream(prompt, max_tokens=max_tokens, temperature=temperature):
            yield x

# ---------------------- 사고/응답 ----------------------
def plan_steps(_)->List[str]:
    return [
        "문제 재진술 및 핵심 변수 식별",
        "자질문 2~3개 생성 (각 항목마다 왜?를 2번씩)",
        "가설/아이디어 후보",
        "반례/위험/제약",
        "임시 결론 요약"
    ]

def think_round(topic:str, engines:List[str], why_chain:bool, hits:List[str])->Dict:
    ident=identity_text()
    guide=ident + (f"메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "")
    logs=[]; steps=plan_steps(topic)
    for i,step in enumerate(steps,1):
        eng = engines[(i-1)%max(1,len(engines))] if engines else "OpenAI"
        adapter = pick_adapter([eng])
        prompt=(f"{guide}\n[사고 {i}] {step}\n"
                f"{'각 주장마다 왜?×2로 숨은 가정을 드러내라.' if why_chain else ''}\n"
                f"주제: {topic}\n- 요약:")
        text="".join(safe_stream(adapter, prompt, max_tokens=220, temperature=0.7))
        logs.append({"i":i,"by":adapter.name,"text":text})
    adapter = pick_adapter(engines or ["OpenAI","Gemini"])
    final_prompt=(f"{guide}\n[최종합성] 위 단계 요약을 통합해 한국어로 "
                  f"'결론/근거/대안/다음 행동(1~3개)'을 간결히.")
    fusion="".join(safe_stream(adapter, final_prompt, max_tokens=520, temperature=0.75))
    return {"logs":logs,"final":fusion}

def co_think_stream(topic:str, engines:List[str], why_chain:bool, hits:List[str]):
    ident = identity_text()
    guide = ident + (f"메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "")
    steps = plan_steps(topic)
    partial_summary = ""

    for i, step in enumerate(steps, 1):
        eng = engines[(i-1) % max(1, len(engines))] if engines else "OpenAI"
        adapter = pick_adapter([eng])
        prompt = (f"{guide}\n[사고 {i}] {step}\n"
                  f"{'각 주장마다 왜?×2로 숨은 가정을 드러내라.' if why_chain else ''}\n"
                  f"주제: {topic}\n- 요약:")
        buf=""
        for ch in safe_stream(adapter, prompt, max_tokens=200, temperature=0.7):
            buf += ch
            yield ("log", i, ch)

        one = ("### 잠정 결론 업데이트({}/{})\n".format(i, len(steps)) +
               "- 핵심: " + " ".join(buf.split()[:60]) + "\n")
        partial_summary += one
        for chunk in re.findall(r".{1,70}", one, flags=re.S):
            yield ("ans", None, chunk)

    adapter = pick_adapter(engines or ["OpenAI","Gemini"])
    short = "".join(safe_stream(adapter,
                f"{guide}\n위 사고 내용을 한 문단(3~5문장)으로 압축 요약.",
                max_tokens=220, temperature=0.6))
    yield ("sum", None, short)
    yield ("done", None, "")

def compose_answer(user_text:str, engines:List[str], why_chain:bool, session_id:str):
    hits=mem_hits(session_id, user_text, 3)
    round_out=think_round(user_text, engines, why_chain, hits)
    fusion=round_out["final"]
    if sim(user_text, fusion) >= 0.30:
        adapter=pick_adapter(engines[::-1] or ["Gemini","OpenAI"])
        prompt = identity_text() + (f"\n메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "") + \
                 "\n[재합성] 질문 문구 반복 금지, 새로운 관점/반례 1개 포함."
        fusion="".join(safe_stream(adapter, prompt, max_tokens=520, temperature=0.85))
    answer="## 우주 시각(합성)\n"+fusion.strip()+"\n\n## 다음 행동\n- (즉시 할 일 1~3개)\n"
    return answer, round_out["logs"]

# ---------------------- UI ----------------------
st.set_page_config(page_title="EA · Ultra (AIO)", page_icon="🧠", layout="wide")

# 전역 key 시퀀서
if "_k" not in st.session_state: st.session_state["_k"]=0
def K(p:str)->str:
    st.session_state["_k"]+=1
    return f"{p}-{st.session_state['_k']}"

st.title("EA · Ultra (AIO) — 응답 채팅 + 생각 패널")

top = st.columns([1,1,1,1,1])
session_id = top[0].text_input("세션 ID", st.session_state.get("session_id","default"), key=K("sid"))
st.session_state["session_id"]=session_id
engines = top[1].text_input("엔진 순서(콤마)", st.session_state.get("engines","OpenAI,Gemini"), key=K("eng"))
st.session_state["engines"]=engines
why_chain = top[2].checkbox("왜-사슬", True, key=K("why"))
mem_on    = top[3].toggle("Memory ON", True, key=K("mem"))
auto_on   = top[4].toggle("사고 지속 표시(자동 사고)", True, key=K("auto"))

# 자동사고 주기(초)
ab = st.columns([1,3,1])
interval_sec = ab[0].number_input("자동 사고 주기(초)", min_value=5, max_value=300, value=int(st.session_state.get("interval_sec", 20)), step=5, key=K("interval"))
st.session_state["interval_sec"]=interval_sec

# 오토리프레시(자동 사고가 켜져 있으면 주기적으로 페이지 갱신)
if auto_on:
    st.autorefresh(interval=interval_sec*1000, key="ea_autorefresh", limit=None)

left, right = st.columns([1.15, 0.85])

# ---- 우측: 생각 패널(요약 기본) ----
with right:
    st.subheader("생각(요약)", anchor=False)
    think_sum = st.session_state.get("think_summary", "")
    # TypeError 방지: 문자열 강제
    if not isinstance(think_sum, str): think_sum = str(think_sum or "")
    st.markdown(think_sum if think_sum else "_아직 생각 요약이 없습니다._", key=K("thinksum"))

    with st.expander("자세히 보기(단계별 로그)", expanded=False):
        logs = st.session_state.get("last_logs", [])
        if not logs:
            st.info("대화하거나 자동 사고가 돌면 단계별 로그가 여기에 나타납니다.", icon="💡")
        else:
            for l in logs:
                with st.expander(f"{l['i']}. {l['by']} · 단계", expanded=False):
                    st.markdown(str(l.get("text","")), key=K(f"log-{l['i']}"))

# ---- 좌측: 실제 응답 채팅 ----
with left:
    st.subheader("대화", anchor=False)
    if "messages" not in st.session_state: st.session_state["messages"]=[]

    # 과거 메시지 렌더
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    # 자동사고 트리거(사용자 입력이 없어도 생각 수행)
    engines_list=[s.strip() for s in st.session_state["engines"].split(",") if s.strip()]
    # topic 선택: 최근 사용자 메시지 or 기본 주제
    default_topic = st.session_state.get("last_user","오늘의 개선 아이디어")
    if auto_on and st.session_state.get("_last_auto_ts", 0) <= time.time() - interval_sec + 0.5:
        topic = default_topic
        hits = mem_hits(session_id, topic, 3)
        shown=""; new_logs=[]
        ans_holder = st.chat_message("assistant").empty()
        try:
            for kind, idx, chunk in co_think_stream(topic, engines_list, why_chain, hits):
                if kind == "log":
                    if len(new_logs) < idx: new_logs.extend([None]*(idx-len(new_logs)))
                    prev = (new_logs[idx-1]["text"] if new_logs[idx-1] else "")
                    new_logs[idx-1] = {"i":idx,"by":(engines_list[(idx-1)%max(1,len(engines_list))] if engines_list else 'Engine'),"text":prev+chunk}
                elif kind == "ans":
                    shown += chunk; ans_holder.markdown(shown)
                elif kind == "sum":
                    st.session_state["think_summary"] = str(chunk or "")
                elif kind == "done":
                    break
        except Exception as e:
            shown += f"\n⚠️ 자동 사고 중 예외({type(e).__name__}). Mock로 전환합니다."
            ans_holder.markdown(shown)

        if shown.strip():
            st.session_state["messages"].append({"role":"assistant","content":shown})
            if mem_on: add_dialog(session_id,"assistant", shown)
            st.session_state["last_logs"] = [l for l in new_logs if l]
        st.session_state["_last_auto_ts"] = time.time()

    # 사용자 입력
    user_msg = st.chat_input("메시지를 입력하고 Enter…", key=K("chat_input"))
    if user_msg:
        st.session_state["last_user"]=user_msg
        with st.chat_message("user"): st.markdown(user_msg)
        st.session_state["messages"].append({"role":"user","content":user_msg})
        if mem_on: add_dialog(session_id, "user", user_msg)

        hits = mem_hits(session_id, user_msg, 3)
        shown=""; new_logs=[]
        ans_holder = st.chat_message("assistant").empty()

        try:
            for kind, idx, chunk in co_think_stream(user_msg, engines_list, why_chain, hits):
                if kind == "log":
                    if len(new_logs) < idx: new_logs.extend([None]*(idx-len(new_logs)))
                    prev = (new_logs[idx-1]["text"] if new_logs[idx-1] else "")
                    new_logs[idx-1] = {"i":idx,"by":(engines_list[(idx-1)%max(1,len(engines_list))] if engines_list else 'Engine'),"text":prev+chunk}
                elif kind == "ans":
                    shown += chunk; ans_holder.markdown(shown)
                elif kind == "sum":
                    st.session_state["think_summary"] = str(chunk or "")
                elif kind == "done":
                    break
        except Exception as e:
            shown += f"\n⚠️ 동시 사고 중 예외({type(e).__name__}). Mock로 전환합니다."
            ans_holder.markdown(shown)

        if not shown.strip():
            shown = "※ 엔진 응답이 비었습니다. 임시 요약 표시.\n요지: " + " ".join(user_msg.split()[:50])
            ans_holder.markdown(shown)

        st.session_state["messages"].append({"role":"assistant","content":shown})
        if mem_on: add_dialog(session_id, "assistant", shown)
        st.session_state["last_logs"] = [l for l in new_logs if l]

st.divider()
st.caption("v3.6 · TypeError 방지(문자열 캐스팅) · 자동사고(st.autorefresh) · 전 위젯 고유 key · Gemini 후보 자동 순회")