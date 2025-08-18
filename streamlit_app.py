# -*- coding: utf-8 -*-
# EA · Ultra — single-file (mobile-friendly + API adapters + multi-round fetch)

import os, sys, re, time, json, random, hashlib, traceback
from pathlib import Path
from datetime import datetime
import streamlit as st

APP_NAME = "EA · Ultra"
BUILD_TAG = "EA-ULTRA-20250818"
DATA_DIR = Path("data")
STATE_PATH = DATA_DIR / "state.json"

def ensure_dirs():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def nowz(): return datetime.utcnow().isoformat()+"Z"

# --- tiny state store (file -> session fallback) ---
def _state_read():
    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return st.session_state.get("_state_fallback", {})

def _state_write(obj):
    try:
        ensure_dirs()
        tmp = STATE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp.replace(STATE_PATH)
    except Exception:
        st.session_state["_state_fallback"] = obj

def sget(key, default=None):
    return _state_read().get(key, default)

def sset(key, val):
    s = _state_read(); s[key] = val; _state_write(s)

def add_msg(role, content):
    msgs = sget("messages", [])
    msgs.append({"t": nowz(), "role": role, "content": content})
    sset("messages", msgs)

def last_msgs(n=50): return sget("messages", [])[-n:]
def clear_msgs(): sset("messages", [])

# ----------------------- adapters -----------------------
class MockAdapter:
    def __init__(self, name="mock"):
        self.name = name
    def generate(self, prompt, max_tokens=512):
        words = (prompt or "").split()
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        extra = ["에아", "우주", "정보장", "핵심", "요약"]
        mix = words + rng.sample(extra, k=min(len(extra), max(1, len(words)//3 or 1)))
        rng.shuffle(mix)
        txt = " ".join(mix)
        return f"에아(Mock): {txt[:max_tokens]}"

class OpenAIAdapter:
    def __init__(self):
        try:
            from openai import OpenAI  # type: ignore
            self.OpenAI = OpenAI
        except Exception as e:
            raise RuntimeError(f"openai 라이브러리 없음: {e}")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
        self.client = self.OpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt, max_tokens=512):
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=0.6,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"(OpenAI 오류) {e}"

class GeminiAdapter:
    def __init__(self):
        try:
            import google.generativeai as genai  # type: ignore
            self.genai = genai
        except Exception as e:
            raise RuntimeError(f"google-generativeai 라이브러리 없음: {e}")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다.")
        self.genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
        self.model = self.genai.GenerativeModel(self.model_name)

    def generate(self, prompt, max_tokens=512):
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config={"temperature":0.6, "max_output_tokens":max_tokens},
                safety_settings=None,
            )
            return resp.text or ""
        except Exception as e:
            return f"(Gemini 오류) {e}"

def pick_adapter(kind: str):
    # 실패시 Mock으로 폴백
    try:
        if kind == "OpenAI":
            return OpenAIAdapter()
        if kind == "Gemini":
            return GeminiAdapter()
    except Exception as e:
        st.toast(f"{kind} 어댑터 실패 → Mock 사용: {e}", icon="⚠️")
    return MockAdapter()

# ----------------------- helpers -----------------------
def dedupe_repeats(text: str):
    tokens = re.findall(r"\S+|\s+", text)
    out, prev = [], None
    for t in tokens:
        if t == prev and t.strip():
            continue
        out.append(t); prev = t
    return "".join(out)

def intent(text: str):
    t = text.strip().lower()
    if t.startswith("/clear"): return "clear"
    if t.startswith("/summary"): return "summary"
    if t.endswith("?"): return "question"
    return "chat"

def multi_round_generate(adapter, prompt, level=3, rounds=1):
    # level → token budget
    level = int(level)
    token_map = {1:256, 2:512, 3:800, 4:1200, 5:1600}
    max_tokens = token_map.get(level, 800)

    acc = ""
    p = prompt
    for i in range(rounds):
        chunk = adapter.generate(p, max_tokens=max_tokens)
        if not chunk: break
        acc += ("\n" if acc else "") + chunk
        # 다음 라운드는 "계속" 신호
        p = f"{prompt}\n(계속해서 이어서 자세히 써줘. 이전에 멈춘 곳부터)"
        # 너무 빠른 과금/호출 방지 및 UX
        time.sleep(0.05)
    return acc.strip()

# ----------------------- UI -----------------------
def render_app():
    st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide")
    st.title("EA · Ultra")
    st.caption("모바일 친화: Enter 전송 / 멀티라인 에디터 · API 어댑터 선택 · 여러 라운드 미리 수집")

    with st.sidebar:
        st.subheader("응답 설정")
        provider = st.selectbox("Adapter", ["Mock","OpenAI","Gemini"], index=0)
        level = st.slider("응답 레벨(토큰 예산)", 1, 5, 3, help="레벨↑ = 더 길게")
        rounds = st.number_input("미리 받을 라운드 수", min_value=1, max_value=6, value=2, step=1)
        input_mode = st.radio("입력 방식", ["Enter 전송(권장)","멀티라인 에디터"], index=0)
        editor_h = st.slider("에디터 높이(멀티라인)", 80, 600, 200)
        st.divider()
        if st.button("대화 초기화"):
            clear_msgs(); st.toast("초기화 완료")

    # 출력 영역
    out_box = st.container()

    # 입력
    user_text = ""
    submitted = False
    if input_mode == "Enter 전송(권장)":
        # 모바일에서 Enter로 바로 전송됨
        user_text = st.chat_input("메시지를 입력하고 Enter")
        submitted = user_text is not None and user_text != ""
    else:
        with st.form("multi_form", clear_on_submit=False):
            user_text = st.text_area("메시지", sget("draft",""), height=editor_h, placeholder="여러 줄 입력 가능")
            colA, colB = st.columns([1,1])
            send = colA.form_submit_button("Send")
            save = colB.form_submit_button("임시 저장")
            if save:
                sset("draft", user_text); st.toast("임시 저장됨")
            submitted = send

    if submitted:
        txt = dedupe_repeats(user_text or "")
        add_msg("user", txt)

        # 어댑터 선택/생성
        adapter = pick_adapter(provider)

        # 여러 라운드 미리 받아 붙이기
        with out_box:
            ph = st.empty()
            try:
                result = multi_round_generate(adapter, txt, level=level, rounds=rounds)
            except Exception:
                result = "(내부 오류) " + traceback.format_exc()
            add_msg("assistant", result)
            ph.success(result)

    # 최근 로그
    st.divider()
    st.subheader("대화")
    cols = st.columns(2)
    with cols[0]:
        st.caption("입력(최근)")
        for m in reversed([m for m in last_msgs(20) if m["role"]=="user"]):
            st.write(f"🧑 {m['content']}")
    with cols[1]:
        st.caption("출력(최근)")
        for m in reversed([m for m in last_msgs(20) if m["role"]=="assistant"]):
            st.write(f"🤖 {m['content']}")

    # 시스템 탭 비슷한 정보
    st.divider()
    with st.expander("System / Debug"):
        st.write({
            "build": BUILD_TAG,
            "python": sys.version.split()[0],
            "cwd": str(Path.cwd()),
            "state_file": str(STATE_PATH),
            "messages": len(sget("messages", [])),
            "adapter": provider,
        })
        st.code("Tip: /clear, /summary 사용 가능. 멀티라인 모드에서 길게 작성 → Send.", language="text")

# ----------------------- entry -----------------------
if __name__ == "__main__":
    render_app()