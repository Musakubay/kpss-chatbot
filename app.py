import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

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