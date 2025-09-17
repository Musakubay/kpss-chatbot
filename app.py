import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import sqlite3
from contextlib import contextmanager
import httpx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# .env yükle
load_dotenv()

DEEPAGENT_URL = os.getenv("DEEPAGENT_URL")  # örn: https://routellm.abacus.ai/v1/chat/completions
DEEPAGENT_KEY = os.getenv("DEEPAGENT_KEY", "")
TEXT_DEPLOYMENT_ID = os.getenv("TEXT_DEPLOYMENT_ID")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://10.0.2.2:8080,http://localhost:8080,http://127.0.0.1:8080"
)

if not DEEPAGENT_URL:
    raise RuntimeError("DEEPAGENT_URL .env içinde tanımlı değil")
if not DEEPAGENT_KEY:
    raise RuntimeError("DEEPAGENT_KEY .env içinde tanımlı değil")
if not TEXT_DEPLOYMENT_ID:
    raise RuntimeError("TEXT_DEPLOYMENT_ID .env içinde tanımlı değil")

# Windows'ta Tesseract yolu gerekebilir:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\\tesseract.exe"

# -------------------- Pydantic Modelleri --------------------
class ChatContentText(BaseModel):
    type: str = "text"
    text: str

class ChatContentImage(BaseModel):
    type: str = "image_url"
    image_url: Dict[str, str]  # {"url": "https://..."}

ChatContent = Union[ChatContentText, ChatContentImage]

class ChatMessage(BaseModel):
    role: str  # 'system' | 'user' | 'assistant'
    content: List[ChatContent]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    sessionId: Optional[str] = Field(default=None)
    meta: Optional[Dict[str, Any]] = Field(default=None)
    temperature: Optional[float] = None
    stream: Optional[bool] = None

class DefineRequest(BaseModel):
    term: str
    sessionId: Optional[str] = None
    temperature: Optional[float] = 0.2

class QuizQuestionRequest(BaseModel):
    ders: str
    konu: str
    zorluk: str = "orta"  # "kolay" | "orta" | "zor"
    sessionId: Optional[str] = None
    temperature: Optional[float] = 0.2

class QuizEvalRequest(BaseModel):
    question: str
    user_answer: str  # "A" | "B" | "C" | "D" | "E" ya da serbest metin
    ders: Optional[str] = None
    konu: Optional[str] = None
    include_solution: bool = True
    sessionId: Optional[str] = None
    temperature: Optional[float] = 0.2

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="KPSS Uzmanı API (Chat + OCR + Quiz + Define + Karşılıklı Soru)",
    description="KPSS Chat, OCR ile çözüm, Quiz akışı, Tanım ve Karşılıklı Soru servisleri",
    version="6.0.0",
)

origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Yardımcılar --------------------
def json_utf8(content: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=content, status_code=status_code, media_type="application/json; charset=utf-8")

def extract_text_from_abacus(resp: Any) -> str:
    """
    RouteLLM/OpenAI-benzeri response'tan güvenli şekilde düz metin çıkarır.
    """
    if not isinstance(resp, dict):
        try:
            return json.dumps(resp, ensure_ascii=False)
        except Exception:
            return str(resp)

    # choices[0].message.content
    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
            txt = "\n".join([p for p in parts if p])
            if txt:
                return txt

    # fallback
    for key in ("content", "text"):
        val = resp.get(key)
        if isinstance(val, str) and val.strip():
            return val

    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)

def kpss_guardrails_instruction() -> str:
    return (
        "Sen bir KPSS uzmanısın. Sadece KPSS ile ilgili konularda yardımcı ol."
        " Konu dışına çıkma. Uygunsuz içerik üretme."
        " Müfredat: Genel Yetenek (Türkçe, Matematik), Genel Kültür (Tarih, Coğrafya, Vatandaşlık),"
        " Eğitim Bilimleri ve Alan Bilgisi ile sınırlıdır."
        " Yanıtları kısa, net ve adım adım ver; sonunda kesin bir cevap belirt."
    )

def quiz_mode_instruction() -> str:
    return (
        "KPSS Quiz Modu yönergeleri:"
        " 1) Her seferinde yalnız 1 soru sor."
        " 2) Soru kökü ve A,B,C,D (gerekirse E) şıklarını ver."
        " 3) Kullanıcının cevabını bekle."
        " 4) Cevaba göre Doğru/Yanlış ve 1-2 cümle açıklama ver."
        " 5) Ardından yeni bir soru sor."
        " 6) Tamamen KPSS müfredatıyla sınırlı kal."
        " 7) Gereksiz sohbet veya konu dışı bilgi verme."
    )

def definition_instruction() -> str:
    return (
        "Sadece KPSS kapsamındaki kavramların kısa ve öz TANIMINI ver."
        " Tanım 2-6 cümle arasında olsun. Soru sorma, örnek isteme, konu dağıtma."
        " Konu KPSS kapsamı dışında ise 'KPSS kapsamı dışında' deyip kısa kes."
    )

def quiz_generation_instruction(ders: str, konu: str, zorluk: str) -> str:
    return (
        f"KPSS {ders} dersi, '{konu}' konusu için {zorluk} seviyede bir çoktan seçmeli SORU üret.\n"
        "- Sadece 1 soru üret.\n"
        "- Soru kökünü ver ve A, B, C, D (gerekirse E) şıklarını açık ve net yaz.\n"
        "- Şıkları tek satırda değil, her birini yeni satırda ver.\n"
        "- CEVABI HEMEN AÇIKLAMA OLARAK YAZMA; ancak içsel olarak doğru şıkkı belirle.\n"
        "- KPSS müfredatı dışına çıkma. Güncel müfredatla çelişen içerik üretme.\n"
    )

def quiz_eval_instruction() -> str:
    return (
        "Kullanıcının verdiği cevabı değerlendir:\n"
        "- Eğer şıklı bir soru ise, kullanıcının seçtiği şık ile doğru şıkkı karşılaştır.\n"
        "- Doğru/yanlış bilgisini net söyle.\n"
        "- 1-3 cümle kısa ve öğretici açıklama ver.\n"
        "- Kısa ve öz ol.\n"
        "- KPSS kapsamı dışına çıkma."
    )

async def call_deployment(deployment_id: str, payload: ChatRequest) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPAGENT_KEY}",
    }

    forward_body: Dict[str, Any] = {
        "deploymentId": deployment_id,
        "messages": [m.dict() for m in payload.messages],
        "stream": False if payload.stream is None else payload.stream,
    }
    if payload.sessionId:
        forward_body["sessionId"] = payload.sessionId
    if payload.meta:
        forward_body["meta"] = payload.meta
    if payload.temperature is not None:
        forward_body["temperature"] = payload.temperature

    print("➡️ Forwarding body:", json.dumps(forward_body, ensure_ascii=False)[:1000])

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(DEEPAGENT_URL, headers=headers, json=forward_body)

        print("⬅️ Abacus Status:", resp.status_code)
        print("⬅️ Abacus Body (first 1000 chars):", resp.text[:1000])

        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code, "error": resp.text}

        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}

        text = extract_text_from_abacus(data)
        return {"ok": True, "status": resp.status_code, "text": text, "raw": data}

    except httpx.RequestError as e:
        return {"ok": False, "status": 502, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"ok": False, "status": 500, "error": f"Internal error: {str(e)}"}

# -------------------- Endpoint'ler --------------------
@app.get("/health")
async def health():
    return json_utf8({"status": "ok", "deployment": TEXT_DEPLOYMENT_ID, "cors_origins": origins})

@app.post("/kpss-text")
async def kpss_text(payload: ChatRequest):
    """
    Normal chat. KPSS guardrails sistem mesajı en başa eklenir.
    """
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages boş olamaz.")

    system_msg = ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())])
    messages = [system_msg] + payload.messages

    result = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(
            messages=messages,
            sessionId=payload.sessionId,
            meta=payload.meta,
            temperature=payload.temperature,
            stream=payload.stream,
        )
    )
    return json_utf8(result, status_code=result.get("status", 200))

@app.post("/kpss-quiz")
async def kpss_quiz(payload: ChatRequest):
    """
    Quiz modu: Sohbet akışında kullanılacak basit uç.
    İlk açılışta 'start' mesajıyla çağır, sonra kullanıcı cevaplarını gönder.
    """
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages boş olamaz.")

    system_msgs = [
        ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
        ChatMessage(role="system", content=[ChatContentText(text=quiz_mode_instruction())]),
    ]

    messages = system_msgs + payload.messages

    # İlk mesaj "start" tarzıysa, LLM'i açılış sorusuna zorla
    try:
        first = payload.messages[0].content[0]
        if isinstance(first, ChatContentText):
            t = first.text.strip().lower()
            if t in ("start", "start_quiz", "başla", "quiz", "soru"):
                messages.append(
                    ChatMessage(role="user", content=[ChatContentText(text="İlk sorunu sor, şıkları ver.")])
                )
    except Exception:
        pass

    result = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(messages=messages, sessionId=payload.sessionId, temperature=payload.temperature, stream=payload.stream)
    )
    return json_utf8(result, status_code=result.get("status", 200))

@app.post("/kpss-ocr")
async def kpss_ocr(file: UploadFile = File(...)):
    """
    Görselden OCR yapar, KPSS çözümü üretir ve net cevap döner.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Yalnızca görsel dosyalar kabul edilir (jpg, png, jpeg, bmp, tiff).")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Boş dosya gönderildi.")

    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel açılamadı: {str(e)}")

    try:
        extracted_text = pytesseract.image_to_string(image, lang="tur+eng").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR hatası: {str(e)}")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="OCR sonucu boş. Görselde metin yok veya okunamadı.")

    prompt = (
        "Aşağıdaki KPSS soru metnini çöz, adım adım mantığını açıkla ve en sonunda net bir cevap ver:\n\n"
        f"{extracted_text}"
    )
    payload = ChatRequest(
        messages=[
            ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
            ChatMessage(role="user", content=[ChatContentText(text=prompt)]),
        ]
    )

    result = await call_deployment(TEXT_DEPLOYMENT_ID, payload)

    out = {
        "ok": result.get("ok", False),
        "status": result.get("status", 200),
        "text": (result.get("text", "") or "").strip(),
        "raw": result.get("raw"),
        "ocr_extracted_text": extracted_text,
        "file_info": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_bytes),
        },
    }
    return json_utf8(out, status_code=out["status"])

@app.post("/kpss-define")
async def kpss_define(payload: DefineRequest):
    """
    Tanım modu: Sadece kısa ve net tanım döner, soru sormaz.
    """
    term = (payload.term or "").strip()
    if not term:
        raise HTTPException(status_code=400, detail="term boş olamaz.")

    messages = [
        ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
        ChatMessage(role="system", content=[ChatContentText(text=definition_instruction())]),
        ChatMessage(role="user", content=[ChatContentText(text=f"Tanımla: {term}")]),
    ]

    result = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(messages=messages, sessionId=payload.sessionId, temperature=payload.temperature, stream=False)
    )

    out = {
        "ok": result.get("ok", False),
        "status": result.get("status", 200),
        "text": (result.get("text", "") or "").strip(),
    }
    return json_utf8(out, status_code=out["status"])

# -------------------- Karşılıklı Soru Modu Endpoint'leri --------------------
@app.post("/kpss-quiz-question")
async def kpss_quiz_question(req: QuizQuestionRequest):
    """
    Seçilen ders+konu+zorluk için 1 adet çoktan seçmeli soru oluşturur.
    Dönen format:
      {
        ok: true,
        question: "Soru kökü ...",
        options: ["A) ...", "B) ...", "C) ...", "D) ...", "E) ...? (opsiyonel)"],
        hint: "Kısa ipucu" (opsiyonel),
        answer: "B" (opsiyonel - frontend göstermek ZORUNDA değil),
        raw: {...} (LLM ham gövde)
      }
    """
    system_msgs = [
        ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
        ChatMessage(role="system", content=[ChatContentText(text=quiz_generation_instruction(req.ders, req.konu, req.zorluk))]),
    ]
    user_prompt = (
        "Aşağıdaki JSON şemasında dönebileceğin alanları üretmek için içsel bir taslak oluştur; ancak kullanıcıya sadece düz metin ver. "
        "Soru ve şıklar net ve temiz olsun."
    )

    # LLM'den soru ve şıkları metin olarak alırız
    result = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(
            messages=system_msgs + [ChatMessage(role="user", content=[ChatContentText(text=user_prompt)])],
            sessionId=req.sessionId,
            temperature=req.temperature,
            stream=False,
        ),
    )
    if not result.get("ok"):
        return json_utf8({"ok": False, "error": result.get("error", "başarısız"), "status": result.get("status", 500)}, status_code=result.get("status", 500))

    raw_text = (result.get("text") or "").strip()

    # Basit çıkarım: metinden soru kökü ve şıkları ayır
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    question_lines: List[str] = []
    options: List[str] = []
    started_options = False
    for ln in lines:
        u = ln.upper()
        if u.startswith(("A)", "B)", "C)", "D)", "E)")):
            started_options = True
            options.append(ln)
        else:
            if not started_options:
                question_lines.append(ln)

    question = " ".join(question_lines).strip()
    if not question and lines:
        question = lines[0]

    # İpucu/cevap için ek çağrı (deterministik)
    hint_answer = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(
            messages=[
                ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
                ChatMessage(role="system", content=[ChatContentText(text="Aşağıdaki çoktan seçmeli sorunun doğru şıkkını tek harf (A/B/C/D/E) olarak ve 1 kısa ipucu olarak döndür.")]),
                ChatMessage(role="user", content=[ChatContentText(text=f"Soru:\n{question}\nŞıklar:\n" + "\n".join(options))]),
            ],
            sessionId=req.sessionId,
            temperature=0.0,
            stream=False,
        ),
    )

    hint_text: Optional[str] = None
    answer_letter: Optional[str] = None
    if hint_answer.get("ok"):
        ha = (hint_answer.get("text") or "").strip()
        lower = ha.lower()
        for ch in ("A", "B", "C", "D", "E"):
            if f" {ch.lower()}" in lower or f": {ch}" in ha or f" {ch})" in ha or f"{ch} " in ha:
                answer_letter = ch
                break
        hint_text = ha

    out = {
        "ok": True,
        "question": question,
        "options": options,
        "hint": hint_text,
        "answer": answer_letter,
        "raw": result.get("raw"),
    }
    return json_utf8(out)

@app.post("/kpss-quiz-eval")
async def kpss_quiz_eval(req: QuizEvalRequest):
    """
    Kullanıcının cevabını değerlendirir.
    Dönen format:
      {
        ok: true,
        correct: true/false,
        feedback: "Kısa açıklama...",
        model_answer: "B" (opsiyonel)
      }
    """
    system_msgs = [
        ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
        ChatMessage(role="system", content=[ChatContentText(text=quiz_eval_instruction())]),
    ]

    prompt = (
        f"Soru:\n{req.question}\n\n"
        f"Kullanıcının cevabı: {req.user_answer}\n\n"
        "Eğer sorunun doğru şıkkını belirleyebiliyorsan, tek harf (A/B/C/D/E) olarak içsel belirle."
        " Ardından kısa bir açıklama ile Doğru/Yanlış sonucunu net şekilde yaz."
        " Yanıtını kısa tut."
    )

    result = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(
            messages=system_msgs + [ChatMessage(role="user", content=[ChatContentText(text=prompt)])],
            sessionId=req.sessionId,
            temperature=req.temperature,
            stream=False,
        ),
    )
    if not result.get("ok"):
        return json_utf8({"ok": False, "error": result.get("error", "başarısız"), "status": result.get("status", 500)}, status_code=result.get("status", 500))

    text = (result.get("text") or "").strip()

    # Basit correct tespiti: "Doğru" / "Yanlış"
    lt = text.lower()
    correct = ("doğru" in lt and "yanlış" not in lt) or ("correct" in lt and "incorrect" not in lt)

    # Model cevabı harf olarak bulmaya çalış
    model_answer = None
    for ch in ("A", "B", "C", "D", "E"):
        if f" {ch})" in text or f" {ch} " in text or f"Cevap: {ch}" in text or f"Doğru: {ch}" in text:
            model_answer = ch
            break

    out = {
        "ok": True,
        "correct": bool(correct),
        "feedback": text,
        "model_answer": model_answer if req.include_solution else None,
    }
    return json_utf8(out)
# ==================== KPSS Tercih Robotu (SQLite Minimal) ====================


# 1) SQLite bağlantısı
SQLITE_PATH = os.getenv("SQLITE_PATH", "./kpss.db")

def _connect():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# 2) Tablo oluşturma
def init_sqlite():
    with db() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS kpss_positions(
            id TEXT PRIMARY KEY,
            term TEXT NOT NULL,
            institution TEXT NOT NULL,
            title TEXT NOT NULL,
            province TEXT NOT NULL,
            district TEXT,
            score_type TEXT NOT NULL,
            base_score REAL,
            quota INTEGER NOT NULL DEFAULT 1,
            edu_level TEXT NOT NULL,
            requirement_codes TEXT NOT NULL, -- JSON string (["3001","BILGISAYAR_SERT"])
            notes TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            list_json TEXT NOT NULL,  -- JSON string of position_id[]
            notes_json TEXT,          -- JSON string of {posId: note}
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""")

init_sqlite()

# 3) Tercih Robotu Pydantic modelleri
class Position(BaseModel):
    id: str
    term: str
    institution: str
    title: str
    province: str
    district: Optional[str] = None
    score_type: str
    base_score: Optional[float] = None
    quota: int = 1
    edu_level: str
    requirement_codes: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class UserProfile(BaseModel):
    score_type: str
    score: float
    edu_level: str
    majors: List[str] = Field(default_factory=list)
    documents: List[str] = Field(default_factory=list)
    preferred_provinces: List[str] = Field(default_factory=list)
    excluded_provinces: List[str] = Field(default_factory=list)
    special: Dict[str, Any] = Field(default_factory=dict)

class MatchResult(BaseModel):
    position_id: str
    status: str           # "eligible" | "ineligible"
    match_score: float    # 0-100
    risk: str             # "düşük" | "orta" | "yüksek"
    reasons: List[str] = Field(default_factory=list)

class MatchRequest(BaseModel):
    profile: UserProfile
    positions: Optional[List[Position]] = None
    filters: Optional[Dict[str, Any]] = None

class MatchResponse(BaseModel):
    ok: bool
    results: List[MatchResult]
    used_count: int

# 4) Yardımcı JSON
def _to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _from_json(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

# 5) Repository fonksiyonları (sqlite3 ile)
def repo_insert_positions(positions: List[Position]):
    with db() as conn:
        c = conn.cursor()
        for p in positions:
            c.execute("""
            INSERT OR REPLACE INTO kpss_positions
            (id, term, institution, title, province, district, score_type, base_score, quota, edu_level, requirement_codes, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.id, p.term, p.institution, p.title, p.province, p.district, p.score_type,
                p.base_score, p.quota, p.edu_level, _to_json(p.requirement_codes), p.notes
            ))

def repo_list_positions(filters: Optional[Dict[str, Any]] = None) -> List[Position]:
    with db() as conn:
        c = conn.cursor()
        base = "SELECT * FROM kpss_positions WHERE 1=1"
        params: List[Any] = []
        if filters:
            if "score_type" in filters and filters["score_type"]:
                base += " AND UPPER(score_type)=UPPER(?)"
                params.append(filters["score_type"])
            if "edu_level" in filters and filters["edu_level"]:
                base += " AND LOWER(edu_level)=LOWER(?)"
                params.append(filters["edu_level"])
            if "province" in filters and filters["province"]:
                base += " AND LOWER(province)=LOWER(?)"
                params.append(filters["province"])
            if "term" in filters and filters["term"]:
                base += " AND LOWER(term)=LOWER(?)"
                params.append(filters["term"])
            if "institution" in filters and filters["institution"]:
                base += " AND LOWER(institution) LIKE LOWER(?)"
                params.append(f"%{filters['institution']}%")
            if "title" in filters and filters["title"]:
                base += " AND LOWER(title) LIKE LOWER(?)"
                params.append(f"%{filters['title']}%")
        c.execute(base, params)
        rows = c.fetchall()

    out: List[Position] = []
    for r in rows:
        out.append(Position(
            id=r["id"],
            term=r["term"],
            institution=r["institution"],
            title=r["title"],
            province=r["province"],
            district=r["district"],
            score_type=r["score_type"],
            base_score=r["base_score"],
            quota=r["quota"],
            edu_level=r["edu_level"],
            requirement_codes=_from_json(r["requirement_codes"], []),
            notes=r["notes"],
        ))
    return out

def repo_get_positions_by_ids(ids: List[str]) -> List[Position]:
    if not ids:
        return []
    q = ",".join(["?"] * len(ids))
    with db() as conn:
        c = conn.cursor()
        c.execute(f"SELECT * FROM kpss_positions WHERE id IN ({q})", ids)
        rows = c.fetchall()
    return [
        Position(
            id=r["id"],
            term=r["term"],
            institution=r["institution"],
            title=r["title"],
            province=r["province"],
            district=r["district"],
            score_type=r["score_type"],
            base_score=r["base_score"],
            quota=r["quota"],
            edu_level=r["edu_level"],
            requirement_codes=_from_json(r["requirement_codes"], []),
            notes=r["notes"],
        ) for r in rows
    ]

def repo_save_preferences(user_id: str, position_ids: List[str], notes: Optional[Dict[str, str]] = None):
    with db() as conn:
        c = conn.cursor()
        c.execute("""
        INSERT INTO user_preferences(user_id, list_json, notes_json)
        VALUES (?, ?, ?)
        """, (user_id, _to_json(position_ids), _to_json(notes) if notes else None))

def repo_list_preferences(user_id: str) -> List[Dict[str, Any]]:
    with db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM user_preferences WHERE user_id = ? ORDER BY id DESC", (user_id,))
        rows = c.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "user_id": r["user_id"],
            "positions": _from_json(r["list_json"], []),
            "notes": _from_json(r["notes_json"], None),
            "created_at": r["created_at"],
        })
    return out

# 6) Basit kural motoru
def _rule_3001(profile: UserProfile, pos: Position):
    ok = profile.edu_level.strip().lower() == "lisans"
    return ok, ("3001 uygun" if ok else "3001: Lisans mezunu olmalı")

def _rule_bilgisayar_sert(profile: UserProfile, pos: Position):
    dl = [d.lower() for d in profile.documents]
    ok = "bilgisayar_sert" in dl
    return ok, ("Bilgisayar sertifikası ✓" if ok else "Bilgisayar sertifikası gerekli")

def _rule_ziraat(profile: UserProfile, pos: Position):
    ml = [m.lower() for m in profile.majors]
    ok = any(k in ml for k in ["ziraat", "ziraat müh", "bitki koruma"])
    return ok, ("Ziraat müh. uygun" if ok else "Ziraat mühendisliği mezunu olmalı")

RULES = {
    "3001": (True, _rule_3001),
    "BILGISAYAR_SERT": (True, _rule_bilgisayar_sert),
    "46XX_ZIRAAT": (True, _rule_ziraat),
}

def evaluate_position(profile: UserProfile, pos: Position) -> MatchResult:
    reasons: List[str] = []
    eligible = True

    for code in pos.requirement_codes:
        mandatory, func = RULES.get(code, (True, lambda p, q: (True, f"{code}: tanımsız (varsayılan geçerli)")))
        ok, reason = func(profile, pos)
        reasons.append(reason)
        if mandatory and not ok:
            eligible = False

    if profile.excluded_provinces and pos.province in profile.excluded_provinces:
        reasons.append(f"Kullanıcı bu ili istemiyor: {pos.province}")
    elif profile.preferred_provinces and pos.province in profile.preferred_provinces:
        reasons.append(f"Tercih edilen il: {pos.province}")

    match = 60.0 if eligible else 25.0
    if profile.preferred_provinces and pos.province in profile.preferred_provinces:
        match += 10.0
    if pos.base_score is not None:
        diff = profile.score - pos.base_score
        match += max(-10.0, min(15.0, diff))
    if pos.quota >= 10:
        match += 3.0
    match = float(max(0.0, min(100.0, match)))

    base = pos.base_score if pos.base_score is not None else profile.score
    diff = profile.score - base
    risk = "düşük" if diff >= 2.0 else ("yüksek" if diff <= -2.0 else "orta")

    return MatchResult(
        position_id=pos.id,
        status="eligible" if eligible else "ineligible",
        match_score=match,
        risk=risk,
        reasons=reasons
    )

# 7) Endpoint'ler
@app.post("/kpss/match", response_model=MatchResponse)
async def kpss_match(req: MatchRequest):
    if not req.profile.score_type.strip() or not req.profile.edu_level.strip():
        raise HTTPException(status_code=400, detail="profile.score_type ve profile.edu_level zorunludur")

    if req.positions is not None:
        positions = req.positions
    else:
        positions = repo_list_positions(req.filters or {})

    results = [evaluate_position(req.profile, p) for p in positions]
    return MatchResponse(ok=True, results=results, used_count=len(positions))

@app.get("/kpss/positions")
async def kpss_list_positions(score_type: Optional[str] = None, edu_level: Optional[str] = None, province: Optional[str] = None, term: Optional[str] = None, institution: Optional[str] = None, title: Optional[str] = None):
    filters: Dict[str, Any] = {}
    if score_type: filters["score_type"] = score_type
    if edu_level: filters["edu_level"] = edu_level
    if province: filters["province"] = province
    if term: filters["term"] = term
    if institution: filters["institution"] = institution
    if title: filters["title"] = title
    data = repo_list_positions(filters)
    return json_utf8({"ok": True, "count": len(data), "positions": [p.dict() for p in data]})

class SavePreferenceRequest(BaseModel):
    user_id: str
    preferences: List[str]
    notes: Optional[Dict[str, str]] = None

@app.post("/kpss/preferences/save")
async def kpss_preferences_save(req: SavePreferenceRequest):
    # Geçersiz id varsa bile kaydediyoruz; istersen repo_get_positions_by_ids ile doğrula
    repo_save_preferences(req.user_id, req.preferences, req.notes)
    return json_utf8({"ok": True, "saved_count": len(req.preferences)})

@app.get("/kpss/preferences/list")
async def kpss_preferences_list(user_id: str):
    rows = repo_list_preferences(user_id)
    return json_utf8({"ok": True, "count": len(rows), "items": rows})

# 8) Mock veri doldurma (isteğe bağlı)
@app.post("/kpss/positions/import-mock")
async def import_mock_positions():
    sample = [
        Position(
            id="1", term="2024/2", institution="Gelir İdaresi", title="Memur",
            province="Ankara", score_type="P3", base_score=83.6, quota=12,
            edu_level="Lisans", requirement_codes=["3001", "BILGISAYAR_SERT"]
        ),
        Position(
            id="2", term="2024/2", institution="NVİ", title="VHKİ",
            province="İstanbul", score_type="P3", base_score=86.9, quota=5,
            edu_level="Lisans", requirement_codes=["3001"]
        ),
        Position(
            id="3", term="2024/2", institution="Tarım ve Orman", title="Mühendis",
            province="İzmir", score_type="P3", base_score=88.2, quota=3,
            edu_level="Lisans", requirement_codes=["46XX_ZIRAAT", "BILGISAYAR_SERT"],
            notes="Ziraat Müh. (Bitki Koruma) tercih edilir."
        ),
    ]
    repo_insert_positions(sample)
    return json_utf8({"ok": True, "inserted": len(sample)})
# ==================== KPSS Tercih Robotu - AI Öneri Katmanı ====================

from enum import Enum

class AiGoal(str, Enum):
    safe = "safe"        # garanti odaklı
    balanced = "balanced"
    bold = "bold"        # iddialı

class AiRecommendRequest(BaseModel):
    profile: UserProfile
    filters: Optional[Dict[str, Any]] = None
    goal: AiGoal = AiGoal.balanced
    top_n: int = 30
    use_llm: bool = False           # true ise LLM rerank + açıklama
    llm_explain_top_k: int = 20     # açıklama üretilecek ilk K kayıt
    llm_rerank_top_k: int = 50      # rerank için ilk K kayıt

class AiSuggestion(BaseModel):
    position_id: str
    institution: str
    title: str
    province: str
    base_score: Optional[float] = None
    quota: int
    match_score: float
    preference_score: float
    bucket: str
    risk: str
    reasons: List[str] = Field(default_factory=list)
    ai_note: Optional[str] = None   # LLM açıklaması (opsiyonel)

class AiRecommendResponse(BaseModel):
    ok: bool
    suggestions: List[AiSuggestion]
    strategy: Optional[str] = None

def _city_boost(profile: UserProfile, pos: Position) -> float:
    pr = (pos.province or "").strip().lower()
    prefers = [s.strip().lower() for s in profile.preferred_provinces]
    excluded = [s.strip().lower() for s in profile.excluded_provinces]
    if pr in excluded:
        return -20.0
    if pr in prefers:
        return 10.0
    return 0.0

def _quota_effect(pos: Position) -> float:
    import math
    q = pos.quota or 1
    return min(10.0, math.log1p(q) * 3.0)

def _base_gap_effect(profile: UserProfile, pos: Position) -> float:
    if pos.base_score is None:
        return 0.0
    gap = profile.score - float(pos.base_score)
    return max(-10.0, min(10.0, gap / 2.0))

def _preference_score(profile: UserProfile, pos: Position, match_score: float) -> float:
    # Heuristik harman
    ps = (
        0.6 * match_score +
        0.2 * _city_boost(profile, pos) +
        0.1 * _quota_effect(pos) +
        0.1 * _base_gap_effect(profile, pos)
    )
    return float(max(0.0, min(100.0, ps)))

def _bucket_of(goal: AiGoal, pref_score: float, risk: str) -> str:
    r = (risk or "").lower()
    if pref_score >= 75 and r in ("düşük", "dusuk", "low"):
        return "safe"
    if pref_score >= 55:
        return "balanced"
    return "bold"

def _simple_strategy_text(goal: AiGoal, suggestions: List[AiSuggestion]) -> str:
    safe_count = sum(1 for s in suggestions if s.bucket == "safe")
    bal_count = sum(1 for s in suggestions if s.bucket == "balanced")
    bold_count = sum(1 for s in suggestions if s.bucket == "bold")
    if goal == AiGoal.safe:
        return f"Önceliğiniz güvenli tercihler. Dağılım: {safe_count} güvenli, {bal_count} dengeli, {bold_count} iddialı."
    if goal == AiGoal.bold:
        return f"Önceliğiniz iddialı tercihler. Dağılım: {safe_count} güvenli, {bal_count} dengeli, {bold_count} iddialı."
    return f"Dengeli bir liste önerildi. Dağılım: {safe_count} güvenli, {bal_count} dengeli, {bold_count} iddialı."

async def _llm_rerank(profile: UserProfile, items: List[Dict[str, Any]]) -> List[float]:
    """
    items: [{pos: Position, match_score: float, pref_score: float, reasons: [...]}, ...] (ilk K öğe)
    Dönen: her item için 0-100 arası rerank_score listesi (aynı sırada).
    """
    # Kısa, maliyet dostu bir prompt; JSON benzeri skor dizisi istiyoruz.
    # LLM’den deterministik bir sayı seti almaya çalışıyoruz (temperature=0).
    lines = []
    for i, it in enumerate(items, start=1):
        p: Position = it["pos"]
        lines.append(
            f"{i}. id={p.id}, kurum={p.institution}, unvan={p.title}, il={p.province}, "
            f"taban={p.base_score}, kontenjan={p.quota}, match={it['match_score']:.1f}, "
            f"pref={it['pref_score']:.1f}"
        )
    prompt = (
        "Aşağıda KPSS ilan listesi veriliyor. Her satır bir ilan. "
        "0-100 arası 'rerank_score' üret ve sadece sayıları virgülle ayırarak sırayla döndür. "
        "Detay yazma, sadece sayılar.\n\n" +
        "\n".join(lines)
    )

    payload = ChatRequest(
        messages=[
            ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
            ChatMessage(role="user", content=[ChatContentText(text=prompt)]),
        ],
        temperature=0.0,
        stream=False,
    )
    res = await call_deployment(TEXT_DEPLOYMENT_ID, payload)
    if not res.get("ok"):
        # Hata olursa rerank yapmayız, 0 döneriz (nötr)
        return [0.0] * len(items)

    text = (res.get("text") or "").strip()
    # "12, 54, 66, 70, 10" gibi bir dize bekliyoruz
    nums = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip().replace("%", "")
        try:
            v = float(part)
            v = max(0.0, min(100.0, v))
            nums.append(v)
        except:
            continue

    if len(nums) < len(items):
        # Eksikse kalanları 0 yap
        nums += [0.0] * (len(items) - len(nums))
    return nums[:len(items)]

async def _llm_explain(profile: UserProfile, items: List[Dict[str, Any]]) -> List[str]:
    """
    items: [{pos: Position, match_score: float, pref_score: float, reasons: [...]}, ...] (ilk K)
    Dönen: Kısa ai_note listesi (aynı sırada).
    """
    notes: List[str] = []
    for it in items:
        p: Position = it["pos"]
        reasons = it.get("reasons", [])
        rp = (
            f"Profil: puan={profile.score} ({profile.score_type}), eğitim={profile.edu_level}, "
            f"tercih iller={profile.preferred_provinces}, hariç iller={profile.excluded_provinces}.\n"
            f"İlan: {p.institution} - {p.title} ({p.province}), taban={p.base_score}, kontenjan={p.quota}.\n"
            f"Nedenler: {', '.join(reasons)}\n"
            "Bu ilan için 1-2 cümle net ve kısa bir öneri notu yaz."
        )
        payload = ChatRequest(
            messages=[
                ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
                ChatMessage(role="user", content=[ChatContentText(text=rp)]),
            ],
            temperature=0.2,
            stream=False,
        )
        res = await call_deployment(TEXT_DEPLOYMENT_ID, payload)
        note = (res.get("text") or "").strip() if res.get("ok") else None
        # Çok uzun ise kısalt
        if note and len(note) > 240:
            note = note[:237] + "..."
        notes.append(note or "")
    return notes

@app.post("/kpss/ai/recommendations", response_model=AiRecommendResponse)
async def kpss_ai_recommendations(req: AiRecommendRequest):
    # 1) Pozisyonları al
    positions = repo_list_positions(req.filters or {})
    if not positions:
        return AiRecommendResponse(ok=True, suggestions=[], strategy="Uygun ilan bulunamadı.")

    # 2) Mevcut matcher ile skorlar
    match_results = [evaluate_position(req.profile, p) for p in positions]
    match_by_id = {m.position_id: m for m in match_results}

    # 3) Eligible filtre + heuristik skor
    items = []
    for p in positions:
        m = match_by_id.get(p.id)
        if not m or m.status != "eligible":
            continue
        ms = float(m.match_score)
        ps = _preference_score(req.profile, p, ms)
        items.append({"pos": p, "match_score": ms, "pref_score": ps, "risk": m.risk, "reasons": m.reasons})

    if not items:
        return AiRecommendResponse(ok=True, suggestions=[], strategy="Uygun ilan bulunamadı (eligibility).")

    # 4) Heuristik sıralama
    items.sort(key=lambda x: x["pref_score"], reverse=True)

    # 5) Opsiyonel LLM rerank (ilk K için)
    if req.use_llm and req.llm_rerank_top_k > 0:
        topk = items[:req.llm_rerank_top_k]
        rerank_scores = await _llm_rerank(req.profile, topk)
        # Harmanlama: yeni_pref = 0.7 * pref + 0.3 * rerank
        for i, sc in enumerate(rerank_scores):
            topk[i]["pref_score"] = 0.7 * topk[i]["pref_score"] + 0.3 * float(sc)
        # Hepsini tekrar sırala
        items.sort(key=lambda x: x["pref_score"], reverse=True)

    # 6) Öneri listesi oluştur
    trimmed = items[:req.top_n]
    ai_notes = []
    if req.use_llm and req.llm_explain_top_k > 0:
        explain_pack = trimmed[:req.llm_explain_top_k]
        ai_notes = await _llm_explain(req.profile, explain_pack)
        # Explain sayısı kadar doldur, geri kalanı boş
        if len(ai_notes) < len(trimmed):
            ai_notes += [""] * (len(trimmed) - len(ai_notes))
    else:
        ai_notes = [""] * len(trimmed)

    suggestions: List[AiSuggestion] = []
    for idx, it in enumerate(trimmed):
        p: Position = it["pos"]
        bucket = _bucket_of(req.goal, it["pref_score"], it["risk"])
        suggestions.append(AiSuggestion(
            position_id=p.id,
            institution=p.institution,
            title=p.title,
            province=p.province,
            base_score=p.base_score,
            quota=p.quota,
            match_score=round(it["match_score"], 1),
            preference_score=round(it["pref_score"], 1),
            bucket=bucket,
            risk=it["risk"],
            reasons=it["reasons"],
            ai_note=ai_notes[idx] if idx < len(ai_notes) else None
        ))

    # 7) Strateji metni
    strategy = _simple_strategy_text(req.goal, suggestions)
    return AiRecommendResponse(ok=True, suggestions=suggestions, strategy=strategy)

class ExplainRequest(BaseModel):
    profile: UserProfile
    position_id: str

@app.post("/kpss/ai/explain")
async def kpss_ai_explain(req: ExplainRequest):
    pos_list = repo_get_positions_by_ids([req.position_id])
    if not pos_list:
        raise HTTPException(status_code=404, detail="İlan bulunamadı")
    p = pos_list[0]
    m = evaluate_position(req.profile, p)

    prompt = (
        f"Kullanıcı profili: puan={req.profile.score} ({req.profile.score_type}), eğitim={req.profile.edu_level}, "
        f"tercih iller={req.profile.preferred_provinces}, hariç iller={req.profile.excluded_provinces}.\n"
        f"İlan: {p.institution} - {p.title} ({p.province}), taban={p.base_score}, kontenjan={p.quota}.\n"
        f"Eşleşme skoru: {m.match_score:.1f}, risk: {m.risk}.\n"
        f"Nedenler: {', '.join(m.reasons)}\n"
        "Bu ilan için 1-3 cümle açıklayıcı, kısa ve net bir öneri yaz."
    )

    res = await call_deployment(
        TEXT_DEPLOYMENT_ID,
        ChatRequest(messages=[
            ChatMessage(role="system", content=[ChatContentText(text=kpss_guardrails_instruction())]),
            ChatMessage(role="user", content=[ChatContentText(text=prompt)]),
        ], temperature=0.2, stream=False)
    )
    if not res.get("ok"):
        return json_utf8({"ok": False, "error": res.get("error", "LLM hatası")}, status_code=res.get("status", 500))
    return json_utf8({"ok": True, "note": (res.get("text") or "").strip()})

class RerankRequest(BaseModel):
    profile: UserProfile
    position_ids: List[str]
    use_llm: bool = True

@app.post("/kpss/ai/rerank")
async def kpss_ai_rerank(req: RerankRequest):
    positions = repo_get_positions_by_ids(req.position_ids)
    if not positions:
        return json_utf8({"ok": True, "items": []})

    items = []
    for p in positions:
        m = evaluate_position(req.profile, p)
        if m.status != "eligible":
            continue
        pref = _preference_score(req.profile, p, float(m.match_score))
        items.append({"pos": p, "match_score": float(m.match_score), "pref_score": pref, "reasons": m.reasons})

    if not items:
        return json_utf8({"ok": True, "items": []})

    items.sort(key=lambda x: x["pref_score"], reverse=True)

    if req.use_llm:
        topk = items[: min(50, len(items))]
        rerank_scores = await _llm_rerank(req.profile, topk)
        for i, sc in enumerate(rerank_scores):
            topk[i]["pref_score"] = 0.7 * topk[i]["pref_score"] + 0.3 * float(sc)
        items.sort(key=lambda x: x["pref_score"], reverse=True)

    out = [{
        "position_id": it["pos"].id,
        "institution": it["pos"].institution,
        "title": it["pos"].title,
        "province": it["pos"].province,
        "preference_score": round(it["pref_score"], 1),
        "match_score": round(it["match_score"], 1),
    } for it in items]

    return json_utf8({"ok": True, "items": out})