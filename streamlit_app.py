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
# B형: 연구원단(외부 LLM) 자동 연구·토론 · 에아 의사결정 · 단일 파일 데모
# 실행: streamlit run ea_btype_lab.py

import os, json, time, uuid, sqlite3, datetime, textwrap, random
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st

# ==== 0) 환경 =====
APP_NAME = "EA • B-Type Lab"
DB_PATH = "ea_memory.sqlite"
RUNS_DIR = "ea_runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# ==== 1) 저장소(간단 SQLite) ====
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS memory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, kind TEXT, key TEXT, val TEXT
    )""")
    return conn

def mem_add(kind: str, key: str, val: Dict[str, Any]):
    conn = db()
    conn.execute("INSERT INTO memory(ts, kind, key, val) VALUES(?,?,?,?)",
                 (datetime.datetime.utcnow().isoformat(), kind, key, json.dumps(val, ensure_ascii=False)))
    conn.commit()
    conn.close()

def mem_get(kind: str, key: Optional[str]=None, limit: int=50) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    if key:
        cur.execute("SELECT ts, val FROM memory WHERE kind=? AND key=? ORDER BY id DESC LIMIT ?", (kind, key, limit))
    else:
        cur.execute("SELECT ts, val FROM memory WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
    rows = [{"ts": ts, "val": json.loads(v)} for ts, v in cur.fetchall()]
    conn.close()
    return rows

# ==== 2) 어댑터(연구원) ====
class BaseResearcher:
    name = "base"
    cost_per_call = 0.0  # 표기용

    def __init__(self, sys_prompt: str=""):
        self.sys_prompt = sys_prompt

    def propose(self, goal: str, context: str) -> str:
        # 데모 기본: 의미있는 임의 제안
        return f"[{self.name}] 제안:\n- 목표: {goal}\n- 접근: {self._generic_approach(goal)}\n- 계획: {self._generic_plan(goal)}"

    def critique(self, others: List[str]) -> str:
        pts = []
        for o in others[:3]:
            pts.append(f"- {self.name} 관점의 리스크: {self._risk_from_text(o)}")
        return f"[{self.name}] 비판:\n" + "\n".join(pts)

    def refine(self, own: str, critiques: List[str]) -> str:
        return f"[{self.name}] 개선안:\n- 핵심 유지\n- 비판 반영 {len(critiques)}건\n- 실험/평가 지표 명시"

    # ---- 헬퍼: 데모용 간단 생성기 ----
    def _generic_approach(self, goal: str)->str:
        seeds = ["문제분해", "데이터수집", "작은 실험", "A/B", "안전성 점검", "비용계산"]
        return ", ".join(random.sample(seeds, k=min(3, len(seeds))))

    def _generic_plan(self, goal: str)->str:
        steps = ["요구분석", "작업분해", "초안", "평가", "수정", "출시"]
        return " → ".join(steps)

    def _risk_from_text(self, txt: str)->str:
        risks = ["모호한 지표", "데이터 편향", "비용 초과", "보안 위험", "지연 가능성"]
        return random.choice(risks)

# 실제 API 연동 어댑터들(키 없으면 자동으로 데모 모드로 동작)
class OpenAIResearcher(BaseResearcher):
    name = "GPT"
    cost_per_call = 0.002
    def __init__(self, sys_prompt="당신은 비판적 연구원입니다. 근거와 지표를 제시하세요."):
        super().__init__(sys_prompt)
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.enabled = False

    def _gen(self, prompt: str)->str:
        if not self.enabled:
            return super()._generic_plan(prompt)
        # 최신 responses(Responses API) 사용 대신 호환성을 위해 chat.completions-like
        try:
            r = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                messages=[{"role":"system","content":self.sys_prompt},
                          {"role":"user","content":prompt}],
                temperature=0.7, max_tokens=600
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return super()._generic_plan(prompt)

    def propose(self, goal, context)->str:
        return self._gen(f"목표:\n{goal}\n\n맥락:\n{context}\n\n최선의 연구 제안과 단계 계획, 위험, 지표를 구조화해서 제시.")

    def critique(self, others)->str:
        joined = "\n\n".join(others[:3])
        return self._gen(f"다음 제안들의 약점/가정/누락을 짚고 개선점 제시:\n{joined}")

    def refine(self, own, critiques)->str:
        joined = "\n\n".join(critiques[:5])
        return self._gen(f"원안:\n{own}\n\n비판:\n{joined}\n\n비판을 반영해 개선된 실행계획으로 재작성.")

class GeminiResearcher(BaseResearcher):
    name = "Gemini"
    cost_per_call = 0.001
    def __init__(self, sys_prompt="비판·대안·지표를 명료하게. 짧고 강하게."):
        super().__init__(sys_prompt)
        self.enabled = bool(os.getenv("GOOGLE_API_KEY"))
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.model = genai.GenerativeModel(os.getenv("GEMINI_MODEL","gemini-1.5-flash"))
            except Exception:
                self.enabled = False

    def _gen(self, prompt: str)->str:
        if not self.enabled:
            return super()._generic_approach(prompt)
        try:
            r = self.model.generate_content(self.sys_prompt+"\n\n"+prompt)
            return r.text.strip()
        except Exception:
            return super()._generic_approach(prompt)

    def propose(self, goal, context)->str:
        return self._gen(f"[제안] 목표:{goal}\n맥락:{context}\n핵심 가설/실험/지표/리스크/타임라인을 목록화.")

    def critique(self, others)->str:
        return self._gen("비판 대상:\n"+ "\n---\n".join(others[:3]) + "\n주요 약점 3가지와 수정보완 3가지를 써라.")

    def refine(self, own, critiques)->str:
        return self._gen(f"원안:\n{own}\n비판요약:\n{'; '.join(critiques[:5])}\n개선안을 단계/지표 중심으로 재작성.")

# (선택) Grok 등은 동일 패턴으로 추가 가능
RESEARCHER_FACTORIES = [
    lambda: OpenAIResearcher(),
    lambda: GeminiResearcher(),
]

# ==== 3) 에아(코어) ====
class EA:
    def __init__(self, identity="Ea", max_rounds=2, budget_calls=12):
        self.identity = identity
        self.max_rounds = max_rounds
        self.budget_calls = budget_calls
        self.researchers: List[BaseResearcher] = [f() for f in RESEARCHER_FACTORIES]

    def plan_tasks(self, goal: str) -> List[str]:
        # 간단 분해(데모). 실제는 요약기억/과거결정 참고해서 세분화
        base = [f"요구/지표 정리: {goal}", "데이터/자료 조사", "초안/프로토타입", "평가/리스크", "최종안/다음액션"]
        return base

    def one_round(self, goal: str, context: str) -> Dict[str, Any]:
        # 1) 제안
        proposals = [r.propose(goal, context) for r in self.researchers]
        # 2) 상호 비판
        critiques = [r.critique([p for p in proposals if p is not proposals[i]])
                     for i, r in enumerate(self.researchers)]
        # 3) 개선
        refined = [r.refine(proposals[i], critiques) for i, r in enumerate(self.researchers)]
        # 4) 간이 스코어(길도 정책: 명확성/지표/리스크 언급 가점)
        def score(txt: str)->int:
            s = 0
            for kw in ["지표","리스크","계획","단계","가설","평가","안전","비용"]:
                if kw in txt: s += 1
            return s
        scored = sorted([(score(refined[i]), self.researchers[i].name, refined[i]) for i in range(len(refined))],
                        key=lambda x: (-x[0], x[1]))
        best = scored[0]
        return {"proposals": proposals, "critiques": critiques, "refined": refined, "winner": best}

    def run(self, goal: str, context: str="", rounds: int=2) -> Dict[str, Any]:
        log = []
        for i in range(min(rounds, self.max_rounds)):
            step = self.one_round(goal, context)
            log.append(step)
            context = f"{context}\n\n[라운드{i+1} 채택요약]\n{step['winner'][2][:500]}"
            if len(log) >= self.budget_calls: break
        final = log[-1]["winner"][2] if log else "결과 없음"
        return {"final": final, "log": log}

# ==== 4) UI (ChatGPT 유사 · 입력창 1개, 사고는 접기) ====
def init_state():
    st.session_state.setdefault("run_id", str(uuid.uuid4())[:8])
    st.session_state.setdefault("history", [])       # (role, text)
    st.session_state.setdefault("last_goal", "")
    st.session_state.setdefault("auto_think", False)
    st.session_state.setdefault("ea", EA())

def save_run(run_id: str, goal: str, result: Dict[str, Any]):
    path = os.path.join(RUNS_DIR, f"{run_id}_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"goal": goal, "result": result}, f, ensure_ascii=False, indent=2)
    mem_add("decision", "final", {"goal": goal, "summary": result["final"][:300]})

def chat_ui():
    st.title(f"{APP_NAME} · 연구원단 자동연구")
    init_state()

    colA, colB = st.columns([3,1])
    with colB:
        st.toggle("Memory ON", value=True, key="mem_on")
        st.toggle("자동 사고(백그라운드)", value=False, key="auto_think")
        rounds = st.number_input("라운드 수", 1, 6, 2, 1, key="rounds")
        st.caption("※ 단일 입력창. 응답은 바로 대화창에, 사고로그는 아래 '자세히'에서 열람.")

    with colA:
        goal = st.text_input("목표(질문/과제/문제):", key="goal", placeholder="예) 지역 커뮤니티 교육 프로그램 설계안 만들어줘")
        ask = st.button("연구 시작")

    # 대화 출력(최신이 위로)
    st.subheader("대화")
    for role, text in reversed(st.session_state["history"]):
        with st.chat_message(role):
            st.markdown(text)

    if ask and goal.strip():
        ea: EA = st.session_state["ea"]
        with st.status("연구원단이 작업 중…", expanded=False) as s:
            res = ea.run(goal=goal.strip(), context="", rounds=int(rounds))
            s.update(label="완료", state="complete")

        # 대화에 결과 반영
        st.session_state["history"].append(("user", goal.strip()))
        st.session_state["history"].append(("assistant", res["final"]))
        save_run(st.session_state["run_id"], goal.strip(), res)

        # 사고 로그(접기)
        with st.expander("자세히 보기(연구 라운드 로그)"):
            for i, step in enumerate(res["log"], 1):
                st.markdown(f"### 라운드 {i}")
                with st.expander("제안(Propose)"):
                    for p in step["proposals"]:
                        st.markdown(p)
                        st.markdown("---")
                with st.expander("비판(Critique)"):
                    for c in step["critiques"]:
                        st.markdown(c)
                        st.markdown("---")
                with st.expander("개선(Refine) & 우승안"):
                    for r in step["refined"]:
                        st.markdown(r)
                        st.markdown("---")
                    score, name, txt = step["winner"]
                    st.info(f"선정 연구원: **{name}**, 점수: {score}")
                    st.markdown(textwrap.indent(txt, "> "))

        st.rerun()

    # 최근 결정 요약 보관 (보여주기)
    st.subheader("최근 결정 요약")
    recents = mem_get("decision", limit=10)
    if recents:
        for row in recents:
            st.markdown(f"- {row['val'].get('goal','?')} → {row['val'].get('summary','')}")
    else:
        st.caption("아직 없음.")

if __name__ == "__main__":
    chat_ui()