from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg
import os
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
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[STARTUP] SentenceTransformer OK")

LLM_MODEL = os.getenv("LLM_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
print(f"[STARTUP] Carregando LLM: {LLM_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu"
)
llm_model.eval()
print("[STARTUP] LLM OK")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_embedding(text):
    return np.array(embedder.encode([text])[0])

def split_chunks(text, size=180, overlap=20):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def search_context(question, top_k=3):
    q_emb = get_embedding(question)
    vector_str = "[" + ",".join(str(x) for x in q_emb.tolist()) + "]"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, content, embedding <-> %s::vector AS distance
                FROM documents
                ORDER BY embedding <-> %s::vector ASC
                LIMIT %s
                """,
                (vector_str, vector_str, top_k)
            )
            return cur.fetchall()

def generate_response(prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                    # GREEDY decoding — determinístico
            num_beams=1,                        # sem beam search (mais rápido)
            repetition_penalty=1.5,             # penaliza repetição agressivamente
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][input_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for artifact in ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "assistant\n", "user\n"]:
        decoded = decoded.replace(artifact, "").strip()
    # Truncar na primeira quebra dupla — se modelo começar a divagar, corta
    if "\n\n" in decoded:
        decoded = decoded.split("\n\n")[0].strip()
    return decoded if decoded else "I could not find a specific answer in the knowledge base."
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
    chunks = split_chunks(content, size=180, overlap=20)

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
                SELECT filename, content, embedding <-> %s::vector AS distance
                FROM documents
                WHERE id = %s
                ORDER BY embedding <-> %s::vector ASC
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

    results = search_context(question, top_k=3)
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
    results = search_context(question, top_k=3)
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
        "You are the IceMan watch store assistant. "
        "Reply with ONE short sentence containing only facts from the CONTEXT. "
        "Never invent prices, phone numbers, websites, URLs or model details. "
        "If the answer is not in the CONTEXT, reply exactly: "
        "I could not find this information in our catalog. "
        "Do not add any extra information after answering."
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
