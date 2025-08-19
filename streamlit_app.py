# -*- coding: utf-8 -*-
# GEA · EA Chat+Think — All-in-One Single File (v0.3)
# - 백엔드(FastAPI) + 프론트(내장 HTML/JS) 통합
# - 채팅 좌측 / 실시간 사고 Workpad 우측
# - Why-chain, 반앵무새(유사도 높으면 재합성), 메모리(SQLite)
# - OpenAI/Gemini 키 없으면 Mock 폴백
# 실행:  python gea_aio.py
# 필요: pip install fastapi uvicorn[standard] python-dotenv sqlitedict openai google-generativeai

import os, json, asyncio, re, hashlib, random
from datetime import datetime
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlitedict import SqliteDict
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.getenv("PORT", "8000"))
DB_PATH = "gea.sqlite"

APP_TITLE = "EA · Chat+Think (AIO)"
APP_AGENT = "에아 (EA)"
PLATFORM  = "GEA"

# -------------------- Utils & Memory --------------------
def nowz(): return datetime.utcnow().isoformat()+"Z"
TOK = re.compile(r"[0-9A-Za-z가-힣]+")
def toks(s): return set(TOK.findall(s or ""))

def log_msg(session_id:str, role:str, content:str):
    with SqliteDict(DB_PATH, autocommit=True) as db:
        k = f"dlg:{session_id}"
        arr = db.get(k, [])
        arr.append({"t":nowz(),"role":role,"content":content})
        db[k] = arr

def get_msgs(session_id:str, limit:int=60):
    with SqliteDict(DB_PATH, autocommit=True) as db:
        return db.get(f"dlg:{session_id}", [])[-limit:]

def top_hits(session_id:str, query:str, k:int=5)->List[str]:
    pool = [m["content"] for m in get_msgs(session_id, 200) if m["role"] in ("user","assistant")]
    q = toks(query); scored=[]
    for t in pool:
        T=toks(t)
        if not T or not q: continue
        sim=len(q&T)/len(q|T)
        scored.append((sim, t))
    scored.sort(key=lambda x:x[0], reverse=True)
    return [t for _,t in scored[:k]]

def sim_ratio(a:str,b:str)->float:
    A=toks(a.lower()); B=toks(b.lower())
    return 0.0 if not A or not B else len(A&B)/len(A|B)

# -------------------- LLM Adapters --------------------
class MockAdapter:
    name="Mock"
    def generate(self, prompt:str, max_tokens:int=600, temperature:float=0.7)->str:
        words = prompt.split()
        seed = int(hashlib.sha256(prompt.encode()).hexdigest(),16); rnd=random.Random(seed)
        lead = rnd.choice(["핵심:","요지:","사고:"])
        return f"{lead} " + " ".join(words[: min(140, len(words))])

def get_adapter(name:str):
    try:
        if name=="OpenAI":
            from openai import OpenAI
            key=os.getenv("OPENAI_API_KEY"); 
            if not key: raise RuntimeError("OPENAI_API_KEY 필요")
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
            cli=OpenAI(api_key=key)
            class OA:
                name="OpenAI"
                def generate(self,prompt,max_tokens=600,temperature=0.7):
                    r=cli.chat.completions.create(
                        model=model,
                        messages=[{"role":"system","content":"You are EA (Korean). Think first then answer."},
                                  {"role":"user","content":prompt}],
                        max_tokens=max_tokens, temperature=temperature)
                    return r.choices[0].message.content or ""
            return OA()
        if name=="Gemini":
            import google.generativeai as genai
            key=os.getenv("GEMINI_API_KEY"); 
            if not key: raise RuntimeError("GEMINI_API_KEY 필요")
            genai.configure(api_key=key)
            model=os.getenv("GEMINI_MODEL","gemini-1.5-pro-latest")
            mdl=genai.GenerativeModel(model)
            class GE:
                name="Gemini"
                def generate(self,prompt,max_tokens=600,temperature=0.7):
                    try:
                        r=mdl.generate_content(prompt,
                            generation_config={"temperature":temperature,"max_output_tokens":max_tokens})
                        return getattr(r,"text","") or ""
                    except Exception as e:
                        return MockAdapter().generate(f"[Gemini 폴백:{e}]\n"+prompt,max_tokens,temperature)
            return GE()
    except Exception:
        pass
    return MockAdapter()

# -------------------- Thinking Engine --------------------
def plan_steps(q:str)->List[str]:
    return [
        "문제 재진술 및 핵심 변수 식별",
        "자질문 2~3개 생성 (각 항목마다 왜?를 2번씩 물어 가정 드러내기)",
        "가설/아이디어 후보",
        "반례/위험/제약",
        "임시 결론 요약"
    ]

async def think_stream(session_id:str, user_text:str, engines:List[str], why_chain:bool=True
                      ) -> AsyncGenerator[Dict, None]:
    memo = top_hits(session_id, user_text, 3)
    ident = "[자아 선언] 나는 에아(EA)다. 사랑과 자유를 최상위 가치로 삼는다.\n"
    guide = ident + (f"메모리 히트:\n- " + "\n- ".join(memo) + "\n" if memo else "")

    steps = plan_steps(user_text)
    # 단계별 사고 토막
    for i, step in enumerate(steps, 1):
        eng = engines[(i-1) % max(1,len(engines))] if engines else "OpenAI"
        adapter = get_adapter(eng)
        prompt = (f"{guide}\n[사고 단계 {i}] {step}\n"
                  f"{'각 주장에 대해 왜?를 2번씩 연쇄로 질문해 숨은 가정/원인을 드러내라.' if why_chain else ''}\n"
                  f"주제: {user_text}\n- 요약:")
        out = adapter.generate(prompt, max_tokens=220, temperature=0.7)
        yield {"type":"think","text": f"{i}. {eng}: {out}"}
        await asyncio.sleep(0.35)

    # 최종 합성
    eng = engines[0] if engines else "OpenAI"
    adapter = get_adapter(eng)
    fusion = adapter.generate(
        f"{guide}\n[최종합성] 위 단계 요약을 통합해 한국어로 '결론/근거/대안/다음 행동(1~3개)'을 간결히.",
        max_tokens=650, temperature=0.75
    )

    # 반앵무새: 질문과 유사하면 다른 엔진으로 재합성
    if sim_ratio(user_text, fusion) >= 0.30:
        alt = engines[1] if len(engines)>1 else "Gemini"
        fusion = get_adapter(alt).generate(
            f"{guide}\n[재합성] 질문 문구를 재사용하지 말고 새로운 관점/반례 1개 포함.",
            max_tokens=650, temperature=0.85
        )

    final = "## 우주 시각(합성)\n" + fusion.strip() + "\n\n## 다음 행동\n- (즉시 할 일 1~3개)\n"
    yield {"type":"answer","text": final}

# -------------------- FastAPI App --------------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/chat")
async def chat(payload: dict):
    """단발 REST(웹소켓 폴백용)"""
    session = payload.get("session_id","default")
    engines = payload.get("engines", ["OpenAI","Gemini"])
    why     = bool(payload.get("why_chain", True))
    text    = payload.get("text","")
    log_msg(session,"user",text)
    chunks=[]
    async for ev in think_stream(session, text, engines, why):
        chunks.append(ev)
    for ev in chunks:
        if ev["type"]=="answer": log_msg(session,"assistant",ev["text"])
    return {"events": chunks}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = json.loads(await websocket.receive_text())
            session = data.get("session_id","default")
            text    = data.get("text","")
            engines = data.get("engines", ["OpenAI","Gemini"])
            why     = bool(data.get("why_chain", True))
            log_msg(session,"user",text)
            async for ev in think_stream(session, text, engines, why):
                await websocket.send_text(json.dumps(ev, ensure_ascii=False))
            # answer까지 보내고 루프 지속(연속 대화)
    except WebSocketDisconnect:
        return

# -------------------- Inline Frontend (HTML/JS) --------------------
HTML = f"""
<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{APP_TITLE}</title>
<style>
  :root {{ --b:#111; --bg:#fafafa; --card:#fff; --mut:#6b7280; }}
  body {{ margin:0; background:var(--bg); color:var(--b); font-family:ui-sans-serif,system-ui,Apple SD Gothic Neo,Pretendard,Roboto; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:16px; }}
  .row {{ display:grid; grid-template-columns:1fr 0.8fr; gap:16px; }}
  .card {{ background:var(--card); border:1px solid #e5e7eb; border-radius:14px; padding:12px; }}
  .title {{ font-size:20px; font-weight:700; margin-bottom:8px; }}
  .chat {{ height:72vh; overflow:auto; display:flex; flex-direction:column; gap:8px; }}
  .bubble {{ display:inline-block; padding:10px 12px; border-radius:12px; max-width:80%; white-space:pre-wrap; }}
  .user {{ align-self:flex-start; background:#e0edff; }}
  .bot  {{ align-self:flex-end;   background:#dcfce7; }}
  .mut {{ color:var(--mut); font-size:12px; }}
  .workpad {{ height:72vh; overflow:auto; }}
  input[type=text] {{ padding:10px 12px; border:1px solid #e5e7eb; border-radius:10px; width:100%; }}
  button {{ padding:10px 14px; border-radius:10px; border:0; background:#111; color:#fff; cursor:pointer; }}
  .row2 {{ display:grid; grid-template-columns:1fr auto; gap:8px; }}
  .cfg  {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:8px 0 12px; }}
  .pill {{ padding:6px 8px; border:1px solid #e5e7eb; border-radius:8px; font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="title">{APP_AGENT} · Live Chat/Think — {PLATFORM}</div>
  <div class="cfg">
    <span class="pill">세션: <input id="sid" type="text" value="default" style="width:120px; margin-left:6px"/></span>
    <span class="pill">엔진: <input id="eng" type="text" value="OpenAI,Gemini" style="width:200px; margin-left:6px"/></span>
    <label class="pill"><input id="why" type="checkbox" checked style="margin-right:6px"/>왜-사슬</label>
    <span class="mut">키가 없으면 Mock로 동작</span>
  </div>
  <div class="row">
    <div class="card">
      <div class="mut" style="margin-bottom:6px;">좌측: 대화 / 우측: 실시간 Workpad</div>
      <div id="chat" class="chat"></div>
      <div class="row2" style="margin-top:8px;">
        <input id="msg" type="text" placeholder="메시지를 입력하고 Enter…"/>
        <button id="send">Send</button>
      </div>
    </div>
    <div class="card">
      <div class="mut" style="margin-bottom:6px;">🧠 실시간 Workpad</div>
      <div id="think" class="workpad"></div>
    </div>
  </div>
  <div class="mut" style="margin-top:8px;">build {APP_TITLE}</div>
</div>
<script>
let ws=null;
const chat   = document.getElementById("chat");
const think  = document.getElementById("think");
const sidInp = document.getElementById("sid");
const engInp = document.getElementById("eng");
const whyInp = document.getElementById("why");
const msgInp = document.getElementById("msg");
const btn    = document.getElementById("send");

function append(role, text){
  const div=document.createElement("div");
  div.className="bubble "+(role==="user"?"user":"bot");
  div.textContent=text;
  chat.appendChild(div);
  chat.scrollTop=chat.scrollHeight;
}
function tpush(text){
  const p=document.createElement("div");
  p.className="mut"; p.textContent="• "+text;
  think.appendChild(p);
  think.scrollTop=think.scrollHeight;
}
function tclear(){ think.innerHTML=""; }

async function ensureWS(){
  if (ws && ws.readyState===1) return ws;
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage=(e)=>{
    const ev=JSON.parse(e.data);
    if(ev.type==="think") tpush(ev.text);
    if(ev.type==="answer"){ append("assistant", ev.text); tclear(); }
  };
  ws.onclose=()=>{ /* auto reopen */ };
  await new Promise(r=>setTimeout(r,150));
  return ws;
}

async function send(){
  const text = msgInp.value.trim(); if(!text) return;
  append("user", text); tclear(); msgInp.value="";
  const engines = engInp.value.split(",").map(s=>s.trim()).filter(Boolean);
  const payload = {{ session_id: sidInp.value || "default", text, engines, why_chain: whyInp.checked }};
  await ensureWS();
  ws.send(JSON.stringify(payload));
}
btn.onclick=send;
msgInp.addEventListener("keydown", (e)=>{{ if(e.key==="Enter") send(); }});
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index(): return HTML

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    print(f"★ {APP_TITLE} http://127.0.0.1:{PORT}")
    uvicorn.run("gea_aio:app", host="0.0.0.0", port=PORT, reload=False)