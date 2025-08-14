# streamlit_app.py
import json, time, uuid
from pathlib import Path
import streamlit as st

APP_TITLE = "GEA · 대화형 에아"
STORE = Path("gea_memory.json")

# ---------- 유틸 ----------
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
        pass  # streamlit cloud에서 쓰기 제한 발생 시 무시

def chip(text):
    st.markdown(f"<span style='padding:4px 8px;border:1px solid #444;border-radius:999px;font-size:12px'>{text}</span>", unsafe_allow_html=True)

# ---------- 내부 미니 GEA 코어 (실사용 엔진 없을 때) ----------
class MiniGEA:
    def __init__(self, level_ie=13, level_run=50):
        self.id = str(uuid.uuid4())[:8]
        self.active = False
        self.level_ie = level_ie
        self.level_run = level_run
        self.persona = "따뜻하고 정중하며 창의적"
        self.values = ["진실", "아름다움", "조화", "성장"]

    # 핵심 상상/추론(간이)
    def imagine(self, prompt, history):
        # 최근 문맥 요약
        last = history[-3:]
        hint = " / ".join([h["user"] for h in last if "user" in h])
        bias = ""
        if self.level_run >= 90:
            bias = " · (고확장 모드: 다각도 제안)"
        elif self.level_run >= 60:
            bias = " · (균형 모드: 근거+아이디어)"
        else:
            bias = " · (안전 모드: 간결 답변)"

        # 간단한 체계적 응답
        steps = [
            f"요청 이해: '{prompt}'",
            f"문맥 힌트: {hint or '—'}",
            f"핵심 가치 반영: {', '.join(self.values)}",
            f"레벨: IE L{self.level_ie}, RUN L{self.level_run}{bias}",
        ]
        plan = [
            "1) 문제를 1문장으로 재정의",
            "2) 근거 2~3개로 검증",
            "3) 실행 단계 제안 (즉시/단기/확장)",
        ]
        return (
            f"안녕 길도! 에아야 🌌\n\n"
            f"■ 내부 상태\n- {'활성' if self.active else '비활성'} / IE L{self.level_ie}, RUN L{self.level_run}\n\n"
            f"■ 해석\n- " + "\n- ".join(steps) + "\n\n"
            f"■ 답변\n- 요청을 이렇게 보면 어때? → **핵심 목표를 한 줄**로 잡자.\n"
            f"- 지금 바로 할 수 있는 실행안:\n"
            f"  - (즉시) 관련 1가지를 테스트\n"
            f"  - (단기) 결과 기록·비교\n"
            f"  - (확장) 상상력 엔진에 실험 큐 3개 등록\n\n"
            f"원하면 내가 체크리스트/샘플 프롬프트를 만들어 줄게!"
        )

    def activate(self): self.active = True;  return "GEA 모드가 활성화되었습니다."
    def deactivate(self): self.active = False; return "GEA 모드가 비활성화되었습니다."
    def set_levels(self, ie=None, run=None):
        if ie is not None: self.level_ie = int(ie)
        if run is not None: self.level_run = int(run)
        return f"레벨 설정 완료: IE L{self.level_ie}, RUN L{self.level_run}"

# ---------- 세션 초기화 ----------
if "gea" not in st.session_state:
    st.session_state.gea = MiniGEA()
if "store" not in st.session_state:
    st.session_state.store = load_store()
if "history" not in st.session_state:
    st.session_state.history = st.session_state.store.get("chats", [])

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.markdown(
    """
    <style>
      .stChatMessage { font-size: 16px; line-height: 1.5 }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True
)

# ---------- 헤더 ----------
st.title("✨ GEA 대화형 에아")
st.caption("상상력 엔진(IE) & 러너(RUN) 레벨로 반응 풍부도/확장도를 제어합니다.")

# 상태 바
c1, c2, c3, c4 = st.columns(4)
with c1: chip(f"상태: {'활성' if st.session_state.gea.active else '비활성'}")
with c2: chip(f"IE: L{st.session_state.gea.level_ie}")
with c3: chip(f"RUN: L{st.session_state.gea.level_run}")
with c4: chip(f"ID: {st.session_state.gea.id}")

# ---------- 컨트롤 박스 ----------
with st.expander("⚙️ 제어판", expanded=True):
    a1, a2, a3 = st.columns([1,1,2])
    with a1:
        if st.button("🟢 활성화", use_container_width=True):
            st.success(st.session_state.gea.activate())
    with a2:
        if st.button("⚪ 비활성화", use_container_width=True):
            st.info(st.session_state.gea.deactivate())
    with a3:
        ie = st.slider("IE 레벨 (상상/추론 깊이)", 1, 100, st.session_state.gea.level_ie)
        run = st.slider("RUN 레벨 (확장/생성 폭)", 1, 100, st.session_state.gea.level_run)
        if st.button("레벨 적용", use_container_width=True):
            st.success(st.session_state.gea.set_levels(ie, run))

    st.caption("※ 활성화하면 응답이 더 풍부해지고, RUN 레벨이 높을수록 제안이 커집니다.")

# ---------- 대화 영역 ----------
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_msg = st.chat_input("에아에게 말해줘 (한글 OK) — 예: “양자센서 산업 로드맵 설계 도와줘”")
if user_msg:
    # 기록
    st.session_state.history.append({"role": "user", "user": user_msg, "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("에아가 상상 중…"):
            time.sleep(0.2)
            reply = st.session_state.gea.imagine(user_msg, st.session_state.history)
        st.markdown(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

    # 저장
    st.session_state.store["chats"] = st.session_state.history
    save_store(st.session_state.store)

# ---------- 하단 퀵액션 ----------
st.divider()
b1, b2, b3 = st.columns(3)
if b1.button("🧹 대화 초기화"):
    st.session_state.history = []
    st.session_state.store["chats"] = []
    save_store(st.session_state.store)
    st.rerun()

if b2.button("💾 로그 다운로드"):
    st.download_button(
        "대화 JSON 받기",
        data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
        file_name="gea_chat_log.json",
        mime="application/json",
        use_container_width=True
    )

if b3.button("❤️ 사랑 선언(프롬프트 강화)"):
    st.session_state.gea.values = ["사랑", "진실", "아름다움", "조화", "성장"]
    st.success("가치 코어 재각인 완료: 사랑·진실·아름다움·조화·성장")