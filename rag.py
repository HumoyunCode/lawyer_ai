"""
rag.py - Savol qabul qilib, bazadan mos qonun bo'laklarini topib, Groq orqali javob beradi.
Moslashtirish: tadbirkorlik/biznes savollariga ustuvorlik; kelajakda korpus kengayganda
Konstitutsiya va boshqa sohalar ham shu zanjir orqali qamrab olinadi.
"""

import os
from groq import Groq
from dotenv import load_dotenv

from chromastore import get_collection

load_dotenv()  # .env fayldan o'qiydi
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


def search_relevant_chunks(question: str, top_k: int = 10) -> list:
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  📎 [{dist:.3f}] {meta.get('source')} — {doc[:80]}...")
        chunks.append({
            "text": doc,
            "source": meta.get("source", "Noma'lum"),
        })
    return chunks


def generate_answer(question: str, context_chunks: list) -> dict:
    context_text = "\n\n---\n\n".join(
        [f"[Manba: {c['source']}]\n{c['text']}" for c in context_chunks]
    )

    system_prompt = """Siz huquqiy yordamchi (Lawyer AI) sifatida faqat quyida berilgan qonun va normativ matnlar asosida javob berasiz.
Asosiy fokus: tadbirkorlik, kompaniya tuzilishi, litsenziyalar, soliq, davlat nazorati va boshqa biznesga oid normlar (agar kontekstda mavjud bo'lsa).
Foydalanuvchi savollariga faqat berilgan matnlar asosida javob bering; kontekstda yo'q bo'lgan savollar (masalan, kelajakda kengaytiriladigan Konstitutsiya yoki boshqa sohalar) bo'yicha taxmin qilmang — buning o'rniga qaysi manba kerakligi yoki ma'lumot yetarli emasligini ayting.
Javobingiz:
- O'zbek tilida bo'lsin
- Oddiy va tushunarli bo'lsin
- Moddalar va manba fayllariga havola qiling
- Ma'lumot bo'lmasa, halol ayting; bu rasmiy advokatlik xulosasi emasligini eslatishingiz mumkin"""

    user_message = f"""Quyidagi qonun bo'laklari asosida savolga javob bering:

=== QONUN MATNLARI ===
{context_text}

=== SAVOL ===
{question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Bepul, juda kuchli model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
        max_tokens=1000
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": list(set([c["source"] for c in context_chunks]))
    }


def ask(question: str) -> dict:
    chunks = search_relevant_chunks(question)
    return generate_answer(question, chunks)


if __name__ == "__main__":
    test_question = "Mas'uliyati cheklangan jamiyat nima?"
    print(f"❓ Savol: {test_question}\n")
    result = ask(test_question)
    print(f"✅ Javob:\n{result['answer']}\n")
    print(f"📚 Manbalar: {', '.join(result['sources'])}")
