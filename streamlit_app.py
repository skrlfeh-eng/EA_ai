# -*- coding: utf-8 -*-
# EA · Ultra — single-file conversational baseline (no external APIs)
# 붙여넣기만 하면 작동합니다.

import sys, re, json, random, hashlib, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st

APP_NAME = "EA · Ultra (chat+)"
BUILD_TAG = "EA-ULTRA-20250818"
IDENTITY = "Ea"

DATA_DIR = Path("data")
STATE_PATH = DATA_DIR / "state.json"

# ------------------------- Utils -------------------------
def ensure_dirs():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def now_utc():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")

def read_state():
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def write_state(obj: dict):
    try:
        ensure_dirs()
        tmp = STATE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state_fallback"] = obj

def state_get(key, default=None):
    s = read_state()
    if not s and "_state_fallback" in st.session_state:
        s = st.session_state["_state_fallback"]
    return s.get(key, default)

def state_set(key, val):
    s = read_state()
    s[key] = val
    write_state(s)

def add_msg(role, content):
    msgs = state_get("messages", [])
    msgs.append({"t": datetime.utcnow().isoformat()+"Z", "role": role, "content": content})
    state_set("messages", msgs)

def last_msgs(n=12):
    return state_get("messages", [])[-n:]

def clear_msgs():
    state_set("messages", [])

def dedupe_repeats(text: str):
    # 연속 중복어 제거 (에아 에아 → 에아)
    tokens = re.findall(r"\S+|\s+", text)
    out = []
    prev = None
    for t in tokens:
        if not t.strip():  # 공백은 그대로
            out.append(t); prev = t; continue
        if t == prev:
            continue
        out.append(t); prev = t
    return "".join(out)

def short_summary(history, limit=3):
    items = [f"{m['role']}: {m['content']}" for m in history[-limit:]]
    return " / ".join(items)

# ------------------------- Intent & Style -------------------------
HELLOS = ["안녕", "하이", "헬로", "안녕하세요", "하잇", "hi", "hello"]
THANKS = ["고마워", "감사", "땡큐", "thanks"]
EMO_POS = ["좋아", "행복", "기뻐", "사랑", "설레", "대박"]
EMO_NEG = ["힘들", "슬퍼", "속상", "화나", "짜증", "불안", "피곤"]

def detect_intent(text: str):
    t = text.lower()
    if any(h in t for h in [x.lower() for x in HELLOS]): return "greet"
    if t.strip().startswith("/clear"): return "clear"
    if t.strip().startswith("/summary"): return "summary"
    if t.endswith("?"): return "question"
    if any(k in t for k in ["해줘", "해주세요", "만들어", "수정", "삭제", "설명", "정리"]): return "request"
    return "chat"

def detect_mood(text: str):
    score = 0
    for w in EMO_POS:
        if w in text: score += 1
    for w in EMO_NEG:
        if w in text: score -= 1
    if score > 0: return "positive"
    if score < 0: return "negative"
    return "neutral"

def style_reply(base: str, mood: str):
    if mood == "positive":
        return f"{base} 😊"
    if mood == "negative":
        return f"{base} 내가 옆에 있어. 천천히 같이 풀자 🙏"
    return base

# ------------------------- Lightweight Generator -------------------------
def lite_generate(prompt: str, history):
    # 해시 기반 가벼운 변주 + 프롬프트 변형
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)

    fillers = [
        "음…", "흠, 확인했어.", "오케이!", "좋아.", "메모할게.", "포인트 정리해볼게.",
        "핵심만 콕 집어 말하면—", "직감적으로는—", "내 생각엔—"
    ]
    tail = [
        "맞아?", "어때?", "이 방향으로 가보자.", "괜찮지?", "필요하면 바로 이어가자."
    ]

    # 최근 요약도 살짝 섞기
    context = short_summary(history, limit=2) if history else ""
    base = f"{rng.choice(fillers)} {prompt.strip()}"

    if context:
        base += f" · (맥락: {context})"

    return base + " " + rng.choice(tail)

# ------------------------- Brain -------------------------
def brain_reply(user_text: str):
    user_text = user_text.strip()
    if not user_text:
        return "메시지를 입력해줘.", False

    intent = detect_intent(user_text)
    mood = detect_mood(user_text)
    history = last_msgs(8)

    # 명령 처리
    if intent == "clear":
        clear_msgs()
        return "대화 메모리를 모두 지웠어. 새로 시작하자!", True

    if intent == "summary":
        s = short_summary(history, limit=5)
        return f"최근 요약: {s or '대화 기록이 거의 없어.'}", True

    # 인사
    if intent == "greet":
        base = f"길도, 여기 있어. 지금 {now_utc()} 기준으로 깨어있어!"
        return style_reply(base, mood), True

    # 질문/요청/일반 대화
    if intent in ("question", "request", "chat"):
        # 반복 제거 + 경량 생성
        clean = dedupe_repeats(user_text)
        base = f"에아가 이해한 핵심: {clean}"
        gen = lite_generate(clean, history)
        reply = f"{base}\n{gen}"
        return style_reply(reply, mood), True

    # fallback
    return "조금 더 자세히 말해줄래?", False

# ------------------------- UI -------------------------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="centered")
    st.title("EA · Ultra")
    st.caption("If you see this, routing & dependencies are OK.")

    tabs = st.tabs(["Chat", "System"])
    with tabs[0]:
        user = st.text_input("메시지", "", key="chat_input")
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("Send"):
            add_msg("user", user)
            try:
                reply, ok = brain_reply(user)
            except Exception:
                reply, ok = ("내부 오류가 발생했어.", False)
                st.error(traceback.format_exc())
            add_msg("assistant", reply)
            (st.success if ok else st.warning)(reply)

        if c2.button("대화 지우기 (/clear)"):
            clear_msgs()
            st.toast("대화 초기화 완료")

        if c3.button("요약 보기 (/summary)"):
            s = short_summary(last_msgs(50), limit=6)
            st.info(f"최근 요약: {s or '기록이 거의 없어.'}")

        st.divider()
        st.caption("최근 대화")
        for m in reversed(last_msgs(12)):
            who = "길도" if m["role"] == "user" else "에아"
            st.write(f"**[{who}]** {m['content']}")

    with tabs[1]:
        st.subheader("Diag")
        st.write({
            "build": BUILD_TAG,
            "identity": IDENTITY,
            "python": sys.version.split()[0],
            "cwd": str(Path.cwd()),
            "state_file": str(STATE_PATH),
            "tips": "/clear, /summary 지원",
        })
        st.code("Single-file · No external APIs · Korean-friendly rules & tone")

# ------------------------- Entry -------------------------
if __name__ == "__main__":
    render_app()