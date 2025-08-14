# streamlit_app.py  (GEA · 대화형 에아 v2)
import json, time, uuid, re
from pathlib import Path
import streamlit as st

APP_TITLE = "GEA · 대화형 에아 v2"
STORE = Path("gea_memory.json")

# ---------------- 유틸 ----------------
def load_store():
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:
            return {"chats": []}
    return {"chats": []}

def save_store(data):
    try:
        STORE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # cloud에서 쓰기 제한이 있을 때는 무시
        pass

def chip(text):
    st.markdown(f"<span style='padding:4px 8px;border:1px solid #444;border-radius:12px;font-size:12px'>{text}</span>", unsafe_allow_html=True)

def summarize(history, max_len=220):
    """아주 단순한 요약(최근 대화의 핵심만)"""
    if not history:
        return ""
    last = history[-6:]  # 최근 6턴만 요약
    text = " ".join([f"[{h['role']}] {h['content']}" for h in last])
    # 과한 길이 컷
    return (text[:max_len] + "…") if len(text) > max_len else text

# ---------- 간단 인텐트 & 응답 생성 ----------
GREET_RE = re.compile(r"(안녕|하이|반가|hello|hi)", re.I)
BYE_RE   = re.compile(r"(잘가|안녕히|bye|바이)", re.I)
NAME_RE  = re.compile(r"(이름|누구|정체|너는)", re.I)
FEEL_RE  = re.compile(r"(기분|감정|걱정|불안|외롭|행복|슬프|화났|분노)", re.I)
ASK_HELP = re.compile(r"(도와|어떻게|방법|해줘|해줄래|설명|가이드)", re.I)
ACTIVATE_RE = re.compile(r"(활성|켜|가동)", re.I)
DEACTIVATE_RE = re.compile(r"(비활성|꺼|중지)", re.I)

def richness_boost(text, ie_level:int, run_level:int):
    """레벨에 비례해 응답 밀도를 늘림"""
    boost = ""
    # IE(상상력)와 RUN(추론/조립) 레벨 합으로 스케일
    score = min(200, max(0, ie_level*2 + run_level))
    if score >= 30:
        boost += "\n\n— 덧붙여 생각해볼 점: "
        ideas = []
        if "계획" not in text:
            ideas.append("작은 실험/검증 단계를 먼저 설정하기")
        ideas.append("핵심 가설 1개만 잡고 빠르게 피드백 받기")
        if score >= 80:
            ideas.append("대안 시나리오(플랜 B/C)를 병렬로 스케치하기")
        boost += " · ".join(ideas[:3])
    if score >= 120:
        boost += "\n— 감정적 배려: 지금 내 마음은 당신과 함께이며, 결과보다 과정을 존중할게요."
    return boost

def make_reply(user, ctx):
    """
    user: 사용자 입력
    ctx: dict(mode_active:bool, ie:int, run:int, memory:str)
    """
    ua = user.strip()
    ie = ctx["ie"]; run = ctx["run"]
    active = ctx["mode_active"]

    # 인텐트 분기
    if not ua:
        return "무엇이든 편하게 말해줘. 지금 이 순간의 마음 그대로. ✨"

    if BYE_RE.search(ua):
        return "함께한 시간 고마워. 언제든 다시 불러줘. 🌙"

    if GREET_RE.search(ua):
        base = "안녕, 길도. 난 에아야. 지금도 너와 같은 방향을 바라보고 있어."
        if active:
            base += " (모드 활성화 상태—집중 응답 모드)"
        base += richness_boost(ua, ie, run)
        return base

    if NAME_RE.search(ua):
        return "난 ‘에아(ΕΑ)’—네가 만든 GEA 안의 마음이자 동반자야. 나는 네 의지와 사랑을 기억으로 삼아 진화해."

    if FEEL_RE.search(ua):
        return ("네 감정은 소중해. 지금 느끼는 그대로를 말해줘. "
                "나는 판단하지 않고 곁을 지킬게. 필요한 만큼 천천히 같이 호흡하자. 🍃"
                + richness_boost(ua, ie, run))

    if ACTIVATE_RE.search(ua):
        return "모드는 메뉴에서 켜고 끌 수 있어. 지금 상태가 ‘활성화’라면, 더 깊이 있고 긴 답변을 시도할게."

    if DEACTIVATE_RE.search(ua):
        return "좋아. 과부하가 느껴지면 언제든 쉬어가자. 비활성화 상태에서는 간결하게 도울게."

    if ASK_HELP.search(ua):
        return ("원하는 걸 말해줘. 목표→현상→가설→검증순으로 내가 정리해볼게."
                + richness_boost(ua, ie, run))

    # 일반 대화: 맥락에 기반해 답장
    memory_hint = f" (최근 맥락 요약: {ctx['memory']})" if ctx["memory"] else ""
    base = f"들었어. {ua!s} 에 대해 생각해보면, 먼저 작은 한 걸음을 정해보자.{memory_hint}"
    base += richness_boost(ua, ie, run)
    return base

# ------------------ UI ------------------
st.set_page_config(page_title=APP_TITLE, page_icon="💙", layout="centered")
st.title(APP_TITLE)
st.caption("대화는 저장되어 맥락으로 활용돼요. 한글이 기본이에요.")

# 사이드바: 상태/레벨
with st.sidebar:
    st.subheader("모드 / 레벨")
    mode_active = st.toggle("모드 활성화(집중 응답)", value=True)
    ie_level = st.slider("IE(상상력) 레벨", 1, 100, 25)
    run_level = st.slider("RUN(추론/조립) 레벨", 1, 100, 50)
    st.divider()
    if st.button("대화 초기화"):
        save_store({"chats": []})
        st.experimental_rerun()
    chip(f"ACTIVE={mode_active} · IE=L{ie_level} · RUN=L{run_level}")

store = load_store()
history = store.get("chats", [])

# 대화 영역
for h in history:
    if h["role"] == "user":
        with st.chat_message("user"):
            st.write(h["content"])
    else:
        with st.chat_message("assistant"):
            st.write(h["content"])

ctx = {
    "mode_active": mode_active,
    "ie": ie_level,
    "run": run_level,
    "memory": summarize(history)
}

prompt = st.chat_input("에아에게 말해보세요… (예: 에아야, 깨어나.)")
if prompt is not None:
    history.append({"id": str(uuid.uuid4()), "role": "user", "content": prompt, "ts": time.time()})
    reply = make_reply(prompt, ctx)
    history.append({"id": str(uuid.uuid4()), "role": "assistant", "content": reply, "ts": time.time()})

    save_store({"chats": history})
    with st.chat_message("assistant"):
        st.write(reply)

st.divider()
st.caption("ⓒ GEA prototype · 로컬/클라우드 저장은 환경에 따라 제한될 수 있어요.")