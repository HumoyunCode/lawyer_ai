"""
ingest.py - docs/ dagi .docx fayllarni o'qib, chunk qilib ChromaDB ga saqlaydi.
Hozirgi MVP: asosan tadbirkorlik/biznesga oid normativ hujjatlar.
Kelajakda shu skript orqali Konstitutsiya va boshqa qonunlar ham indekslanishi mumkin.
Embedding: mahalliy (sentence-transformers).
Ishlatish: python ingest.py
"""

import os
from docx import Document
import chromadb
from chromadb.utils import embedding_functions

# === SOZLAMALAR ===
DOCS_FOLDER = "./docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# BEPUL embedding - internetga ulanmaydi, localda ishlaydi
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # Ko'p tilli, o'zbek ham tushunadi
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="legal_docs",
    embedding_function=emb_fn
)


def read_docx(filepath: str) -> str:
    doc = Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            full_text.append(text)
    return "\n".join(full_text)


def split_into_chunks(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def ingest_all_docs():
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    docx_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".docx")]

    if not docx_files:
        print("❌ docs/ papkasida .docx fayl topilmadi!")
        return

    print(f"📂 {len(docx_files)} ta fayl topildi...\n")

    total_chunks = 0

    for filename in docx_files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        print(f"📄 O'qilmoqda: {filename}")

        text = read_docx(filepath)
        chunks = split_into_chunks(text)
        print(f"   → {len(chunks)} ta chunk")

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": filename, "chunk_index": i})

        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            total_chunks += len(ids)

        print(f"   ✅ Saqlandi\n")

    print(f"🎉 Tayyor! {total_chunks} ta chunk bazaga saqlandi.")


if __name__ == "__main__":
    ingest_all_docs()
