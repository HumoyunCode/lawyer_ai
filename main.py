"""
main.py - FastAPI server
Lawyer AI (yuridik yordamchi): savollarga indekslangan qonun matnlari asosida javob beradi.
Hozirgi MVP: tadbirkorlik va biznesga tegishli qoidalar (docs/ dagi hujjatlar).
Reja: Konstitutsiya va boshqa sohalardagi normativ-huquqiy aktlarni qamrab olish.
Ishlatish: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag import ask
import os

load_dotenv()

app = FastAPI(
    title="Lawyer AI — biznes qonunchiligi",
    description="RAG orqali foydalanuvchi savollariga qonun matnlariga tayangan javoblar. "
    "Hozirgi qamrov: indekslangan biznes/tadbirkorlik hujjatlari. "
    "Kelajakda: Konstitutsiya va kengaytirilgan qonunchilik bazasi.",
)

# CORS (web UI dan murojaat uchun)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static fayllar (index.html)
app.mount("/static", StaticFiles(directory="."), name="static")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    """
    Savol qabul qilib, qonun asosida javob qaytaradi.
    
    Misol:
    POST /ask
    { "question": "MChJ ochish uchun nima kerak?" }
    """
    result = ask(req.question)
    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"]
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Lawyer AI ishlayapti!",
        "scope": "biznes va tadbirkorlik (indekslangan hujjatlar)",
        "roadmap": "Konstitutsiya va boshqa qonunlar bazasini kengaytirish",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
