from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg
import os
import requests
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app, origins=["*"], allow_headers="*", supports_credentials=True)
load_dotenv()

# ─────────────────────────────────────────────
# CONEXÃO COM BANCO
# ─────────────────────────────────────────────

def get_conn():
    db_host = os.getenv("DB_HOST")
    sslmode = "disable" if db_host in ("localhost", "127.0.0.1") else "require"
    return psycopg.connect(
        host=db_host,
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=sslmode
    )

# ─────────────────────────────────────────────
# SCHEMA DO BANCO
# ─────────────────────────────────────────────

VECTOR_DIM = 384

def init_vector_table():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({VECTOR_DIM}),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            conn.commit()

# ─────────────────────────────────────────────
# MODELOS (carregados uma única vez no startup)
# ─────────────────────────────────────────────

print("[STARTUP] Carregando SentenceTransformer...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("[STARTUP] SentenceTransformer OK")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
print(f"[STARTUP] LLM via OpenRouter: {OPENROUTER_MODEL}")
print("[STARTUP] LLM OK")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_embedding(text):
    return np.array(embedder.encode([text])[0])

def split_chunks(text, size=500, overlap=50):
    import re
    blocks = re.split(r'(?m)(?=^Produto:|^RESUMO DE POLÍTICAS|^POLÍTICA DE SERVIÇOS|^POLÍTICA DE MANUTENÇÃO|^AVISO IMPORTANTE|^MARCAS AUTORIZADAS|^LISTA DE LOJAS|^IceMan \w+\.)', text)
    chunks = [b.strip() for b in blocks if len(b.strip()) > 30]
    return chunks

KNOWN_ENTITIES = [
    "Rolex", "Audemars Piguet", "Omega", "TAG Heuer", "Panerai",
    "Submariner", "Daytona", "GMT-Master", "Datejust", "Sky-Dweller",
    "Royal Oak", "Seamaster", "Speedmaster", "Planet Ocean",
    "Monaco", "Aquaracer", "Formula 1", "Luminor",
    "Ibirapuera", "Paulista", "Leblon", "Barra", "Campinas",
    "gravação", "garantia", "Pix", "parcela", "manutenção",
    "devolução", "encomenda", "transferência", "revisão"
]

def search_context(question, top_k=5):
    q_emb = get_embedding(question)
    vector_str = "[" + ",".join(str(x) for x in q_emb.tolist()) + "]"
    question_lower = question.lower()
    keyword_filter = next(
        (e for e in KNOWN_ENTITIES if e.lower() in question_lower),
        None
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            if keyword_filter:
                cur.execute(
                    """
                    SELECT filename, content, 1 - (embedding <=> %s::vector) AS distance
                    FROM documents
                    WHERE content ILIKE %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                    """,
                    (vector_str, f"%{keyword_filter}%", vector_str, top_k)
                )
                results = cur.fetchall()
                if results:
                    return results
            cur.execute(
                """
                SELECT filename, content, 1 - (embedding <=> %s::vector) AS distance
                FROM documents
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (vector_str, vector_str, top_k)
            )
            return cur.fetchall()

def generate_response(prompt, max_new_tokens=300):
    """Chama o LLM via OpenRouter API (compatível com OpenAI)."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/orugian/GS-CloudComputing",
                "X-Title": "IceMan RAG Assistant",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.1,
            },
            timeout=30
        )
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"[OpenRouter] Resposta inesperada: {data}")
            return "I could not find a specific answer in the knowledge base."
    except Exception as e:
        print(f"[OpenRouter] Erro: {e}")
        return "I could not find a specific answer in the knowledge base."


# ─────────────────────────────────────────────

# ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify({"message": "Hello, World!"})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename.lower()

    # Extração de texto
    if filename.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            content = "\n".join(
                page.extract_text() or "" for page in pdf_reader.pages
            )
        except Exception as e:
            return jsonify({"error": f"Erro ao ler PDF: {e}"}), 400
    else:
        try:
            content = file.read().decode("utf-8")
        except Exception as e:
            return jsonify({"error": f"Erro ao ler arquivo: {e}"}), 400

    if not content.strip():
        return jsonify({"error": "Arquivo vazio ou sem texto extraível"}), 400

    # Chunking
    chunks = split_chunks(content, size=500, overlap=50)

    # Indexação
    indexed = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                if not chunk.strip():
                    continue
                emb = get_embedding(chunk)
                cur.execute(
                    "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s)",
                    (file.filename, chunk, emb.tolist())
                )
                indexed += 1
            conn.commit()

    return jsonify({
        "message": "File uploaded and embedded successfully",
        "filename": file.filename,
        "chunks_indexed": indexed
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, filename, created_at FROM documents ORDER BY id DESC"
            )
            docs = [
                {"id": r[0], "filename": r[1], "created_at": str(r[2])}
                for r in cur.fetchall()
            ]
    return jsonify({"documents": docs})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    doc_id = data.get("doc_id")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not doc_id:
        return jsonify({"error": "No document selected"}), 400

    q_emb = get_embedding(question)
    vector_str = "[" + ",".join(str(x) for x in q_emb.tolist()) + "]"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, content, 1 - (embedding <=> %s::vector) AS distance
                FROM documents
                WHERE id = %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1
                """,
                (vector_str, doc_id, vector_str)
            )
            results = cur.fetchall()

    docs = [
        {"filename": r[0], "content": r[1], "distance": float(r[2])}
        for r in results
    ]
    return jsonify({"matches": docs})


@app.route("/ask_all", methods=["POST"])
def ask_all():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    results = search_context(question, top_k=5)
    docs = [
        {"filename": r[0], "content": r[1], "distance": float(r[2])}
        for r in results
    ]
    return jsonify({"matches": docs})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1. Buscar chunks relevantes
    results = search_context(question, top_k=5)
    if not results:
        return jsonify({
            "answer": "I could not find relevant information in the knowledge base.",
            "sources": []
        })

    # 2. Montar contexto numerado (ajuda o LLM a ancorar trechos)
    context_parts = []
    for idx, r in enumerate(results, 1):
        context_parts.append(f"[{idx}] {r[1].strip()}")
    context = "\n\n".join(context_parts)

    # 3. Sources com distância (útil para debug e frontend)
    sources = [
        {"filename": r[0], "distance": round(float(r[2]), 4)}
        for r in results
    ]

    # 4. System prompt — direto, restrito, anti-alucinação
    system = (
        "Você é o assistente virtual da IceMan, especializada em relógios de luxo. "
        "Responda SEMPRE em português, de forma clara e direta. "
        "Use APENAS as informações do CONTEXTO fornecido para responder. "
        "Nunca invente preços, estoques, telefones, URLs ou qualquer dado. "
        "Se a resposta não estiver no CONTEXTO, diga: "
        "Não encontrei essa informação em nosso catálogo."
    )

    # 5. Prompt no formato chat template do SmolLM2-Instruct
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"CONTEXT:\n{context}\n\n"
        f"Based only on the CONTEXT, answer this question in one sentence. "
        f"If the answer is not in the CONTEXT, say you could not find it.\n\n"
        f"QUESTION: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 6. Gerar resposta
    answer = generate_response(prompt, max_new_tokens=60)

    return jsonify({
        "answer": answer,
        "sources": sources
    })

# ─────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────

init_vector_table()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
