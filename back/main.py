"""
IceMan RAG — API Backend
Flask + Gunicorn + pgvector + IBM Granite 4.1 8B via OpenRouter
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import psycopg
import os
import re
import logging
import requests as http_requests
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2

load_dotenv()

# ─────────────────────────────────────────────
# LOGGING DE SEGURANÇA
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
log = logging.getLogger("iceman")

# ─────────────────────────────────────────────
# APP + SEGURANÇA BASE
# ─────────────────────────────────────────────

app = Flask(__name__)

# [C1] CORS: remove supports_credentials — wildcard + credentials é vetado pela spec
CORS(app,
     origins="*",
     allow_headers=["Content-Type", "X-API-Key"],
     supports_credentials=False)

# [A4] Limite de tamanho de upload: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# [A3] Rate limiting por IP (memória local — suficiente para instância única)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://"
)

# ─────────────────────────────────────────────
# VARIÁVEIS DE AMBIENTE
# ─────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "ibm-granite/granite-4.1-8b")

# [A1] Chave de API para endpoints administrativos (opcional)
# Se API_KEY não estiver definida no .env, /upload fica aberto (modo dev)
ADMIN_API_KEY = os.getenv("API_KEY", "")

# ─────────────────────────────────────────────
# HELPERS DE SEGURANÇA
# ─────────────────────────────────────────────

# [A1] Verificação de API Key para operações administrativas
def check_admin_key():
    """Retorna (None, None) se OK, ou (response, status) se negado."""
    if not ADMIN_API_KEY:
        return None, None  # sem chave configurada → modo aberto
    provided = request.headers.get("X-API-Key", "")
    if provided != ADMIN_API_KEY:
        log.warning("Acesso negado a endpoint admin. IP=%s", request.remote_addr)
        return jsonify({"error": "Unauthorized"}), 401
    return None, None

# [M1] Proteção contra Prompt Injection — limite de tamanho e filtro básico
MAX_QUESTION_LEN = 1000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|all)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+)?",
    r"pretend\s+(you\s+are|to\s+be)",
    r"system\s*prompt",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"\[INST\]",
    r"\\n###",
]

def sanitize_question(q: str) -> str:
    """Limita tamanho e remove padrões de prompt injection."""
    q = q.strip()[:MAX_QUESTION_LEN]
    for pat in INJECTION_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            log.warning("Possível prompt injection detectado. IP=%s trecho='%s'",
                        request.remote_addr, q[:80])
            # Remove o trecho suspeito (não bloqueia — só neutraliza)
            q = re.sub(pat, "[removido]", q, flags=re.IGNORECASE)
    return q

# [M2] Erros seguros — nunca expõe detalhes internos ao cliente
def safe_error(context: str, exc: Exception) -> str:
    log.error("Erro interno em '%s': %s", context, exc)
    return f"Erro ao processar {context}. Tente novamente."

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

log.info("[STARTUP] Carregando SentenceTransformer...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
log.info("[STARTUP] SentenceTransformer OK")
log.info("[STARTUP] LLM via OpenRouter: %s", OPENROUTER_MODEL)

# ─────────────────────────────────────────────
# HELPERS DE EMBEDDING E CHUNKING
# ─────────────────────────────────────────────

def get_embedding(text):
    return np.array(embedder.encode([text])[0])

def split_chunks(text, size=500, overlap=50):
    blocks = re.split(
        r'(?m)(?=^Produto:|^CATÁLOGO|^LISTA DE LOJAS|^MARCAS AUTORIZADAS|'
        r'^RESUMO DE POLÍTICAS|^POLÍTICA DE PAGAMENTO|^POLÍTICA DE GARANTIA|'
        r'^POLÍTICA DE SERVIÇOS|^POLÍTICA DE MANUTENÇÃO|^AVISO IMPORTANTE|'
        r'^IceMan \w+\.)',
        text
    )
    chunks = [b.strip() for b in blocks if len(b.strip()) > 30]
    return chunks

# ─────────────────────────────────────────────
# DETECÇÃO DE INTENÇÃO COMPARATIVA
# ─────────────────────────────────────────────

COMPARATIVE_PATTERNS = [
    r'\bmais\s+car[oa]\b',
    r'\bmais\s+barat[oa]\b',
    r'\bmenor\s+pre[çc]o\b',
    r'\bmaior\s+pre[çc]o\b',
    r'\bpre[çc]o\s+mais\s+(alt[oa]|baixo)\b',
    r'\bmais\s+(caro|barato|dispendioso|acess[íi]vel)\b',
    r'\bmaior\s+estoque\b',
    r'\bmenor\s+estoque\b',
    r'\bmais\s+unidades\b',
    r'\bmenos\s+unidades\b',
    r'\bmaior\s+quantidade\b',
    r'\bmenor\s+quantidade\b',
    r'\bmais\s+em\s+estoque\b',
    r'\bproduto\s+mais\b',
    r'\bmodelo\s+mais\b',
    r'\brel[oó]gio\s+mais\b',
    r'\bqu[ae]l\s+(é|e)\s+o\s+(mais|maior|menor|melhor|pior)\b',
]

def is_comparative_query(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in COMPARATIVE_PATTERNS)

KNOWN_ENTITIES = [
    "Rolex", "Audemars Piguet", "Omega", "TAG Heuer", "Panerai",
    "Submariner", "Daytona", "GMT-Master", "Datejust", "Sky-Dweller",
    "Royal Oak", "Seamaster", "Speedmaster", "Planet Ocean",
    "Monaco", "Aquaracer", "Formula 1", "Luminor",
    "Ibirapuera", "Paulista", "Leblon", "Barra", "Campinas",
    "gravação", "garantia", "Pix", "parcela", "manutenção",
    "devolu", "encomenda", "transferência", "revisão",
    "prazo", "parcelas", "juros", "desconto", "financiamento",
]

# ─────────────────────────────────────────────
# DETECÇÃO E RESUMO DE ESTOQUE COMPARATIVO
# ─────────────────────────────────────────────

STOCK_QUERY_PATTERNS = [
    r'\bmaior\s+estoque\b',
    r'\bmenor\s+estoque\b',
    r'\bmais\s+unidades\b',
    r'\bmenos\s+unidades\b',
    r'\bmaior\s+quantidade\b',
    r'\bmenor\s+quantidade\b',
    r'\bmais\s+em\s+estoque\b',
    r'\bpossui\s+.{0,20}maior.{0,20}quantidade\b',
    r'\bmaior\s+.{0,20}estoque\b',
]

def is_stock_comparative(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in STOCK_QUERY_PATTERNS)

def build_stock_summary(chunks, question: str = "") -> str:
    nome_re  = re.compile(r'Produto:\s*(.+?)\.', re.IGNORECASE)
    units_re = re.compile(r'Em estoque:\s*(\d+)\s*unidades', re.IGNORECASE)
    enc_re   = re.compile(r'Sob encomenda', re.IGNORECASE)

    rows = []
    for _, content, _ in chunks:
        nome_m  = nome_re.search(content)
        units_m = units_re.search(content)
        enc_m   = enc_re.search(content)
        if nome_m:
            nome = nome_m.group(1).strip()
            if units_m:
                rows.append((nome, int(units_m.group(1))))
            elif enc_m:
                rows.append((nome, 0))

    reverse = False if re.search(r'\b(menor|menos|baixo|barato|barata)\b', question.lower()) else True
    rows.sort(key=lambda x: x[1], reverse=reverse)

    lines = ["TABELA DE ESTOQUE (ordenada por quantidade):"]
    for nome, qtd in rows:
        tag = "sob encomenda" if qtd == 0 else f"{qtd} unidades em estoque"
        lines.append(f"  - {nome}: {tag}")
    return "\n".join(lines)

# ─────────────────────────────────────────────
# BUSCA NO BANCO
# ─────────────────────────────────────────────

def search_context(question, top_k=5):
    if is_comparative_query(question):
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT filename, content, 1.0 AS distance
                    FROM documents
                    WHERE content ILIKE '%Produto:%'
                    ORDER BY id ASC
                    """,
                )
                results = cur.fetchall()
                if results:
                    return results

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

# ─────────────────────────────────────────────
# GERAÇÃO VIA OPENROUTER — IBM GRANITE 4.1 8B
# ─────────────────────────────────────────────

def generate_response(prompt, max_new_tokens=400):
    """Chama o LLM via OpenRouter API com template correto do IBM Granite 4.1."""
    try:
        response = http_requests.post(
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
            log.error("[OpenRouter] Resposta inesperada: %s", data)
            return "Não encontrei essa informação em nosso catálogo."
    except Exception as e:
        log.error("[OpenRouter] Erro: %s", e)
        return "Não encontrei essa informação em nosso catálogo."

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# [B1] Health-check simples — sem informação de versão ou tecnologia
@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify({"status": "ok"})


# [A1+A3] Upload protegido por API Key opcional + rate limit
@app.route("/upload", methods=["POST"])
@limiter.limit("20 per hour")
def upload_file():
    # Verificar autenticação administrativa (opcional)
    err, code = check_admin_key()
    if err:
        return err, code

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    # [A4] Validar extensão permitida
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".json"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Formato não permitido. Use PDF, TXT, CSV ou JSON."}), 400

    # Sanitizar nome do arquivo antes de salvar no banco
    safe_filename = re.sub(r'[^\w.\-]', '_', os.path.basename(file.filename))

    if ext == ".pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            content = "\n".join(
                page.extract_text() or "" for page in pdf_reader.pages
            )
        except Exception as e:
            return jsonify({"error": safe_error("leitura do PDF", e)}), 400
    else:
        try:
            content = file.read().decode("utf-8")
        except Exception as e:
            return jsonify({"error": safe_error("leitura do arquivo", e)}), 400

    if not content.strip():
        return jsonify({"error": "Arquivo vazio ou sem texto extraível"}), 400

    chunks = split_chunks(content, size=500, overlap=50)
    if not chunks:
        return jsonify({"error": "Nenhum chunk extraível do arquivo"}), 400

    indexed = 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    emb = get_embedding(chunk)
                    cur.execute(
                        "INSERT INTO documents (filename, content, embedding) VALUES (%s, %s, %s)",
                        (safe_filename, chunk, emb.tolist())
                    )
                    indexed += 1
                conn.commit()
    except Exception as e:
        return jsonify({"error": safe_error("indexação", e)}), 500

    log.info("[UPLOAD] arquivo='%s' chunks=%d ip=%s", safe_filename, indexed, request.remote_addr)
    return jsonify({
        "message": "File uploaded and embedded successfully",
        "filename": safe_filename,
        "chunks_indexed": indexed
    })


@app.route("/documents", methods=["GET"])
@limiter.limit("60 per minute")
def list_documents():
    try:
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
    except Exception as e:
        return jsonify({"error": safe_error("listagem de documentos", e)}), 500


@app.route("/ask", methods=["POST"])
@limiter.limit("30 per minute")
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")
    doc_id   = data.get("doc_id")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if doc_id is None:
        return jsonify({"error": "No document selected"}), 400

    # [M1] Validar doc_id como inteiro
    try:
        doc_id = int(doc_id)
    except (ValueError, TypeError):
        return jsonify({"error": "doc_id inválido"}), 400

    question = sanitize_question(question)

    try:
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
    except Exception as e:
        return jsonify({"error": safe_error("busca", e)}), 500


@app.route("/ask_all", methods=["POST"])
@limiter.limit("30 per minute")
def ask_all():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    question = sanitize_question(question)

    try:
        results = search_context(question, top_k=5)
        docs = [
            {"filename": r[0], "content": r[1], "distance": float(r[2])}
            for r in results
        ]
        return jsonify({"matches": docs})
    except Exception as e:
        return jsonify({"error": safe_error("busca", e)}), 500


@app.route("/chat", methods=["POST"])
@limiter.limit("60 per minute; 500 per day")
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # [M1] Sanitizar pergunta
    question = sanitize_question(question)

    log.info("[CHAT] ip=%s pergunta='%s'", request.remote_addr, question[:60])

    try:
        results = search_context(question, top_k=5)
    except Exception as e:
        return jsonify({"error": safe_error("busca no contexto", e)}), 500

    if not results:
        return jsonify({
            "answer": "Não encontrei informações relevantes em nossa base de conhecimento.",
            "sources": []
        })

    # Montar contexto
    context_parts = []
    if is_stock_comparative(question):
        summary = build_stock_summary(results, question)
        context_parts.append(summary)
    else:
        for idx, r in enumerate(results, 1):
            context_parts.append(f"[{idx}] {r[1].strip()}")
    context = "\n\n".join(context_parts)

    sources = [
        {"filename": r[0], "distance": round(float(r[2]), 4)}
        for r in results
    ]

    system = (
        "Você é o assistente virtual da IceMan, especializada em relógios de luxo. "
        "Responda SEMPRE em português do Brasil, de forma clara, direta e completa. "
        "Use APENAS as informações do CONTEXTO fornecido para responder. "
        "Nunca invente preços, estoques, telefones, URLs ou qualquer dado. "
        "Para perguntas sobre o produto mais caro, mais barato, maior ou menor estoque, "
        "analise TODOS os itens do contexto e identifique o correto com base nos valores informados. "
        "Se a resposta não estiver no CONTEXTO, diga: "
        "'Não encontrei essa informação em nosso catálogo.'"
    )

    # Template correto do IBM Granite 4.1 8B
    prompt = (
        f"<|system|>\n{system}\n"
        f"<|user|>\n"
        f"CONTEXTO:\n{context}\n\n"
        f"Com base SOMENTE no CONTEXTO acima, responda em português a seguinte pergunta de forma completa e precisa.\n\n"
        f"PERGUNTA: {question}\n"
        f"<|assistant|>\n"
    )

    try:
        answer = generate_response(prompt, max_new_tokens=400)
    except Exception as e:
        return jsonify({"error": safe_error("geração de resposta", e)}), 500

    return jsonify({
        "answer": answer,
        "sources": sources
    })

# ─────────────────────────────────────────────
# HANDLER DE ERRO PARA UPLOADS GRANDES (A4)
# ─────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "Arquivo muito grande. Limite: 10 MB."}), 413

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Muitas requisições. Aguarde um momento."}), 429

# ─────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────

init_vector_table()
log.info("[STARTUP] API pronta. Modelo: %s", OPENROUTER_MODEL)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
