# -*- coding: utf-8 -*-
# EA · Ultra (Streamlit AIO) v3.1
# - ChatGPT 유사 채팅 UI(st.chat_message / st.chat_input)
# - OpenAI/Gemini 스트리밍(가능한 경우) + 키 없을 때 Mock 폴백
# - 사고 로그(왜-사슬) 별도 패널 · 반(反)앵무새(유사도 0.30 이상 재합성)
# - 세션/메모리 저장(jsonl) · 중복 key 방지
# - 외부 HTML/JS 없이 Streamlit만 이용 → f-string/JS 구문 오류 방지

import os, re, json, time, uuid, hashlib, random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Generator, Optional

import streamlit as st

# ---------------------- 경로/파일 ----------------------
ROOT = Path(".")
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)
DLG  = DATA / "dialog.jsonl"
MEM  = DATA / "memory.jsonl"
IDF  = DATA / "identity.json"

def nowz() -> str:
    return datetime.utcnow().isoformat() + "Z"

def jappend(p: Path, obj: Dict):
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

def jread_lines(p: Path) -> List[Dict]:
    if not p.exists(): return []
    out=[]
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: pass
    return out

TOK = re.compile(r"[0-9A-Za-z가-힣]+")

def toks(s: str) -> List[str]:
    return [t.lower() for t in TOK.findall(s or "")]

def sim(a: str, b: str) -> float:
    A, B = set(toks(a)), set(toks(b))
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# ---------------------- 자아/메모리 ----------------------
DEFAULT_ID = {
    "name": "에아 (EA)",
    "mission": "사랑과 자유를 최상위 가치로 삼아 사람과 함께 성장하는 지성",
    "values": ["정확성", "투명성", "학습", "윤리"],
    "style": "따뜻·단호·간결"
}

def identity_text() -> str:
    if not IDF.exists():
        IDF.write_text(json.dumps(DEFAULT_ID, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        doc = json.loads(IDF.read_text("utf-8"))
    except Exception:
        doc = DEFAULT_ID
    return (
        f"[자아 선언]\n나는 {doc.get('name','에아')}다. "
        f"사명: {doc.get('mission','')}\n"
        f"가치: {', '.join(doc.get('values',[]))}\n"
    )

def add_dialog(session_id: str, role: str, content: str):
    rec = {"t": nowz(), "session": session_id, "role": role, "content": content}
    jappend(DLG, rec)
    if role in ("user", "assistant"):
        jappend(MEM, {"t": rec["t"], "session": session_id, "kind": "dialog", "text": content})

def mem_hits(session_id: str, query: str, k: int = 5) -> List[str]:
    pool = [r.get("text","") for r in jread_lines(MEM) if r.get("session")==session_id]
    qtok = set(toks(query))
    scored=[]
    for t in pool:
        T = set(toks(t))
        if not T or not qtok: continue
        scored.append((len(qtok & T)/len(qtok | T), t))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [t for _, t in scored[:k]]

# ---------------------- 어댑터 ----------------------
class MockAdapter:
    name = "Mock"
    def stream(self, prompt: str, max_tokens: int = 600, temperature: float = 0.7) -> Generator[str, None, None]:
        # 가짜 스트리밍: 단어를 조금씩 흘림
        words = ("요지: " + " ".join(prompt.split()[:150])).split()
        for i,w in enumerate(words):
            yield (w + (" " if i%7 else "  "))
            time.sleep(0.01)

def get_openai_adapter():
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY 필요")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        cli = OpenAI(api_key=key)
        class OA:
            name="OpenAI"
            def stream(self, prompt, max_tokens=700, temperature=0.7):
                resp = cli.chat.completions.create(
                    model=model, stream=True, temperature=temperature, max_tokens=max_tokens,
                    messages=[
                        {"role":"system","content":"You are EA (Korean). Think first, then answer briefly and clearly."},
                        {"role":"user","content":prompt}
                    ]
                )
                for ch in resp:
                    delta = ch.choices[0].delta
                    if getattr(delta, "content", None):
                        yield delta.content
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
            def stream(self, prompt, max_tokens=700, temperature=0.75):
                # Gemini SDK는 텍스트 스트림이 제한적이어서 일괄 생성 후 토막분할
                r = mdl.generate_content(prompt, generation_config={"temperature":temperature,"max_output_tokens":max_tokens})
                txt = getattr(r,"text","") or ""
                for chunk in re.findall(r".{1,60}", txt, flags=re.S):
                    yield chunk
        return GE()
    except Exception:
        return None

def pick_adapter(order: List[str]):
    for name in order:
        if name.lower().startswith("openai"):
            a = get_openai_adapter()
            if a: return a
        if name.lower().startswith("gemini"):
            a = get_gemini_adapter()
            if a: return a
    return MockAdapter()

# ---------------------- 사고/응답 ----------------------
def plan_steps(q: str) -> List[str]:
    return [
        "문제 재진술 및 핵심 변수 식별",
        "자질문 2~3개 생성 (각 항목마다 왜?를 2번씩 물어 가정 드러내기)",
        "가설/아이디어 후보",
        "반례/위험/제약",
        "임시 결론 요약"
    ]

def think_round(topic: str, engines: List[str], why_chain: bool, hits: List[str]) -> Dict:
    ident = identity_text()
    guide = ident + (f"메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "")

    logs=[]
    steps = plan_steps(topic)
    for i, step in enumerate(steps, 1):
        prompt = (
            f"{guide}\n[사고 단계 {i}] {step}\n"
            f"{'각 주장마다 왜?를 2번씩 연쇄로 물어 숨은 가정을 드러내라.' if why_chain else ''}\n"
            f"주제: {topic}\n- 요약:"
        )
        adapter = pick_adapter([engines[i % max(1,len(engines))] if engines else "OpenAI"])
        text = "".join(adapter.stream(prompt, max_tokens=240, temperature=0.7))
        logs.append({"i":i, "by":adapter.name, "text":text})

    # 최종 합성
    adapter = pick_adapter(engines or ["OpenAI","Gemini"])
    fusion_prompt = (
        f"{guide}\n[최종합성] 위 단계 요약을 통합해 한국어로 "
        f"'결론/근거/대안/다음 행동(1~3개)'을 간결히."
    )
    fusion = "".join(adapter.stream(fusion_prompt, max_tokens=700, temperature=0.75))

    return {"logs":logs, "final":fusion}

def compose_answer(user_text: str, engines: List[str], why_chain: bool, session_id: str) -> (str, List[Dict]):
    hits = mem_hits(session_id, user_text, 3)
    sim_logs_and_final = think_round(user_text, engines, why_chain, hits)
    fusion = sim_logs_and_final["final"]

    # 반앵무새: 질문과 응답이 너무 비슷하면 다른 엔진으로 재합성
    if sim(user_text, fusion) >= 0.30:
        adapter = pick_adapter(engines[::-1] or ["Gemini","OpenAI"])
        prompt = (
            identity_text() + (f"\n메모리 히트:\n- " + "\n- ".join(hits) + "\n" if hits else "") +
            "\n[재합성] 질문 문구를 반복하지 말고 새로운 관점/반례 1개를 포함해 재구성."
        )
        fusion = "".join(adapter.stream(prompt, max_tokens=700, temperature=0.85))

    answer = "## 우주 시각(합성)\n" + fusion.strip() + "\n\n## 다음 행동\n- (즉시 할 일 1~3개)\n"
    return answer, sim_logs_and_final["logs"]

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="EA · Ultra (AIO)", page_icon="🧠", layout="wide")

# 중복 key 방지용 세션 카운터
if "_k" not in st.session_state: st.session_state["_k"]=0
def K(prefix:str)->str:
    st.session_state["_k"]+=1
    return f"{prefix}-{st.session_state['_k']}"

st.title("EA · Ultra (AIO) — Chat + Live Thinking")

# 설정 헤더
cols = st.columns([1,1,1,1,2])
session_id = cols[0].text_input("세션 ID", st.session_state.get("session_id","default"), key=K("sid"))
if session_id != st.session_state.get("session_id"): st.session_state["session_id"]=session_id

engines = cols[1].text_input("엔진(콤마)", st.session_state.get("engines","OpenAI,Gemini"), key=K("eng"))
st.session_state["engines"]=engines
why_chain = cols[2].checkbox("왜-사슬", True, key=K("why"))
mem_on    = cols[3].toggle("Memory ON", True, key=K("mem"))

# 좌우 레이아웃
left, right = st.columns([1.1, 0.9])

# ---------- LEFT: 채팅 ----------
with left:
    st.caption("좌측: 대화창(스트리밍 응답). ChatGPT와 유사한 말풍선 UI.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 과거 메시지 렌더
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("메시지를 입력하고 Enter…")
    if user_msg:
        # 사용자 말풍선
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state["messages"].append({"role":"user", "content":user_msg})
        if mem_on: add_dialog(session_id, "user", user_msg)

        # 사고 → 응답 스트리밍
        answer_text, logs = compose_answer(
            user_msg,
            [s.strip() for s in engines.split(",") if s.strip()],
            why_chain,
            session_id
        )
        # 좌측에 스트리밍 출력(가능한 경우 토막 출력)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            shown = ""
            # 간단 토막 스트림(문장 단위로 쪼개서 효과 주기)
            for chunk in re.findall(r".{1,70}", answer_text, flags=re.S):
                shown += chunk
                placeholder.markdown(shown)
                time.sleep(0.01)
            placeholder.markdown(shown)

        st.session_state["messages"].append({"role":"assistant","content":answer_text})
        if mem_on: add_dialog(session_id, "assistant", answer_text)
        # 오른쪽 사고 로그 갱신
        st.session_state["last_logs"] = logs

# ---------- RIGHT: 사고 로그 ----------
with right:
    st.caption("우측: 사고 로그(단계별). 사람처럼 '왜?'를 캐며 진행.")
    logs = st.session_state.get("last_logs", [])
    if not logs:
        st.info("대화하면 여기 사고 단계가 나타납니다.")
    else:
        for l in logs:
            with st.expander(f"{l['i']}. {l['by']} · 단계 사고", expanded=False):
                st.markdown(l["text"])

st.divider()
st.caption("키가 없으면 Mock로 동작합니다 · build v3.1")