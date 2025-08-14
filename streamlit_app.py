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
# ---------- 리치 응답 헬퍼 ----------
def tone_wrap(text, tone):
    if tone == "담백":   return text
    if tone == "공손":   return f"{text}\n\n부족한 부분이 있다면 언제든 말씀 주세요."
    if tone == "따뜻":   return f"{text}\n\n당신 곁에서 차분히 함께할게요. 🌿"
    if tone == "열정":   return f"{text}\n\n지금 바로 불붙여 보자! 🔥"
    return text

def bullets(items):
    return "\n".join([f"- {i}" for i in items])

def section(title, body):
    return f"### {title}\n{body}\n"

def mk_outline(query):
    return bullets([
        "핵심 목적 정리",
        "현재 상황/제약 요약",
        "핵심 가설 1~2개",
        "검증 지표/성공 기준",
        "리스크와 가드레일",
    ])

def mk_steps(query, depth=3):
    steps = [
        "문제 정의 · 요구 명세 확정",
        "작은 실험(POC) 설계 · 데이터 확보",
        "평가 지표·성공 조건 합의",
        "실행 → 피드백 → 개선 루프",
        "결과 공유 · 다음 단계 의사결정",
    ]
    return bullets(steps[:max(3, depth)])

def mk_examples(query, n=2):
    return bullets([f"예시 {i+1}: “{query}”를 {['간단프로토', '파일럿'][i%2]}로 구현" for i in range(n)])

def mk_pros_cons(query):
    return bullets([
        f"장점: {query}의 속도/단순성/비용효율",
        "단점: 데이터 품질/스케일 이슈/운영 복잡도",
        "대안: 규칙기반 + ML 하이브리드, 단계적 자동화",
    ])

def mk_risks(query):
    return bullets([
        "요구 불명확 → 스코프 부풀기",
        "데이터 편향 → 결과 왜곡",
        "지표 부적합 → 성공 착시",
    ])

def mk_next(query):
    return bullets([
        "목표 한 줄로 확정",
        "성공 지표 1~2개 선택",
        "3일짜리 미니 POC 일정 잡기",
    ])
    # --- 일반 대화(리치 생성) ---
    memory_hint = f"(최근 맥락: {ctx['memory']}) " if ctx["memory"] else ""
    depth = detail + (ie + run) // 80  # 레벨이 높을수록 살짝 더 깊게

    blocks = []

    if ctx["rich_mode"] == "요약":
        blocks.append(section("핵심 요약", f"{memory_hint}{ua} 의도를 한 줄로: **가치 창출을 위한 실용적 해법 탐색**"))
        blocks.append(section("바로 다음 한 걸음", mk_next(ua)))

    elif ctx["rich_mode"] == "계획서":
        blocks.append(section("목표/배경", f"{ua}\n\n{memory_hint}"))
        blocks.append(section("아키텍처 개요", mk_outline(ua)))
        blocks.append(section("실행 단계", mk_steps(ua, depth)))
        if depth >= 4:
            blocks.append(section("예시/대안", mk_examples(ua, n=2)))
            blocks.append(section("리스크", mk_risks(ua)))

    elif ctx["rich_mode"] == "코치":
        blocks.append(section("관찰", f"지금 포인트는 **선택과 집중**. 불필요한 스코프를 줄이자."))
        blocks.append(section("질문", bullets([
            "진짜로 풀 문제는 무엇인가요(한 문장)?",
            "성공을 어떻게 측정하나요(정량1+정성1)?",
            "3일 안에 시험 가능한 가장 작은 단위는?",
        ])))
        blocks.append(section("액션", mk_next(ua)))

    elif ctx["rich_mode"] == "스토리":
        story = (
            f"처음에 우리는 '{ua}'를 막연히 바라봤어. 하지만 한 걸음씩 나누자 길이 보였지. "
            "작은 실험 하나가 성공했고, 그 데이터가 다음 선택을 밝혔어. 우리는 안전장치를 두고, "
            "틀리면 바로 고쳤고, 옳았다면 과감히 키웠어. 결국 ‘가치'가 현실이 되었지."
        )
        blocks.append(section("이야기", story))
        if depth >= 4:
            blocks.append(section("현실 적용 체크리스트", mk_steps(ua, depth)))

    else:  # 설명+예시 (기본)
        blocks.append(section("핵심 개념", f"{ua}를 이해/해결하기 위한 핵심 축"))
        blocks.append(section("왜(Why)", bullets([
            "문제가 낳는 비용/리스크",
            "해결 시 얻는 가장 큰 이득 1가지",
        ])))
        blocks.append(section("무엇(What)", mk_outline(ua)))
        blocks.append(section("어떻게(How)", mk_steps(ua, depth)))
        blocks.append(section("예시/대안", mk_examples(ua, n=1 + (depth>=4))))
        if depth >= 4:
            blocks.append(section("장·단점", mk_pros_cons(ua)))
            blocks.append(section("리스크", mk_risks(ua)))
        blocks.append(section("다음 액션", mk_next(ua)))

    base = "\n".join(blocks)
    return tone_wrap(base, ctx.get("tone","따뜻"))
    