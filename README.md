# ��� IceMan Intelligent Catalog — Assistente IA com RAG na AWS

> Trabalho Final de Semestre — FIAP Graduação  
> Disciplina: Arquitetura de Soluções na AWS  
> Prof. Itamar | 1º Semestre 2026

---

## ��� Visão Geral

Sistema de **Assistente de IA com Retrieval-Augmented Generation (RAG)** deployado integralmente na AWS, desenvolvido como trabalho final de semestre.

O sistema permite fazer upload de documentos (PDF/TXT), indexá-los em um banco vetorial e realizar consultas em linguagem natural com respostas geradas por um LLM local rodando na EC2.

**Caso de uso:** catálogo inteligente da loja fictícia **IceMan Luxury Watches** — consultas sobre preços, estoque e localização de produtos.

---

## ���️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                         USUÁRIO                             │
│                      (Navegador Web)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS
┌──────────────────────────▼──────────────────────────────────┐
│                    FRONTEND (S3)                            │
│              Static Website Hosting                         │
│                    index.html                               │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP :8000
┌──────────────────────────▼──────────────────────────────────┐
│                   API + LLM (EC2)                           │
│         Flask + Gunicorn + SmolLM2-1.7B-Instruct            │
│         SentenceTransformers (all-MiniLM-L6-v2)            │
└──────────────────────────┬──────────────────────────────────┘
                           │ PostgreSQL :5432
┌──────────────────────────▼──────────────────────────────────┐
│               BANCO VETORIAL (Aurora RDS)                   │
│            PostgreSQL 15 + pgvector extension               │
│           Armazena chunks + embeddings 384 dims             │
└─────────────────────────────────────────────────────────────┘
```

### Fluxo RAG

```
1. Upload documento (PDF/TXT)
        ↓
2. Extração de texto
        ↓
3. Chunking (size=180, overlap=20)
        ↓
4. Geração de embeddings (all-MiniLM-L6-v2, 384 dims)
        ↓
5. Armazenamento no pgvector (Aurora PostgreSQL)
        ↓
─── QUERY ───────────────────────────────────────────
        ↓
6. Usuário faz pergunta
        ↓
7. Embedding da pergunta
        ↓
8. Busca por similaridade coseno no pgvector (top-3)
        ↓
9. Montagem do contexto + prompt
        ↓
10. LLM local gera resposta (SmolLM2-1.7B-Instruct)
        ↓
11. Retorno: resposta + fontes ao usuário
```

---

## ���️ Stack Tecnológica

| Componente | Tecnologia | Versão |
|---|---|---|
| **Cloud** | AWS EC2 + Aurora RDS + S3 | — |
| **API** | Flask + Gunicorn | Flask 3.x |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 (384 dims) |
| **LLM Local** | SmolLM2-Instruct | 1.7B parâmetros |
| **Banco Vetorial** | PostgreSQL + pgvector | PostgreSQL 15 |
| **Container** | Docker | — |
| **Frontend** | HTML/CSS/JS estático | S3 Static Hosting |
| **Runtime** | Python 3.11 | Ubuntu 24.04 |

---

## ☁️ Infraestrutura AWS

| Serviço | Configuração | Função |
|---|---|---|
| **EC2** | t3.medium (2 vCPU, 4GB RAM) — Ubuntu 24.04 | API + LLM local |
| **Aurora PostgreSQL** | db.t3.medium — PostgreSQL 15 + pgvector | Banco vetorial |
| **S3** | Static Website Hosting habilitado | Frontend |
| **Security Groups** | Porta 22 (SSH) + 8000 (API) | Segurança de rede |

---

## ��� Endpoints da API

| Método | Endpoint | Função |
|---|---|---|
| `GET` | `/api/data` | Health check |
| `POST` | `/upload` | Upload PDF/TXT → chunking → embeddings → pgvector |
| `GET` | `/documents` | Lista documentos indexados |
| `POST` | `/ask` | Busca semântica em documento específico |
| `POST` | `/ask_all` | Busca semântica em todos os documentos |
| `POST` | `/chat` | RAG completo: retrieval + LLM → resposta |

### Exemplo de uso

```bash
# Health check
curl http://<EC2_IP>:8000/api/data

# Upload de documento
curl -X POST http://<EC2_IP>:8000/upload \
     -F "file=@estoque_iceman.txt"

# Chat com o agente
curl -X POST http://<EC2_IP>:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the price of the Rolex Submariner?"}'
```

---

## ��� Deploy — Passo a Passo

### Pré-requisitos

- Conta AWS ativa
- Par de chaves EC2 (.pem)
- Python 3.11+
- Git

### 1. EC2 — Configuração inicial

```bash
# Conectar na EC2
ssh -i chave.pem ubuntu@<EC2_IP>

# Atualizar sistema
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clonar repositório
git clone https://github.com/SEU_USUARIO/SEU_REPO
cd SEU_REPO/back

# Ambiente virtual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Aurora PostgreSQL — Configurar pgvector

```sql
-- Conectar no banco
psql -h <RDS_ENDPOINT> -U postgres -d postgres

-- Habilitar extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Verificar
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Variáveis de ambiente

```bash
cp .env.sample .env
nano .env
```

```env
DB_HOST=<RDS_ENDPOINT>
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=<sua_senha>
LLM_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
```

### 4. Rodar a API

```bash
# Desenvolvimento
python main.py

# Produção (Gunicorn)
gunicorn main:app -w 1 -b 0.0.0.0:8000 --timeout 300 --preload
```

### 5. Configurar como serviço systemd

```bash
sudo nano /etc/systemd/system/rag-api.service
```

```ini
[Unit]
Description=RAG API - Assistente IA com Flask + SmolLM2
After=network.target

[Service]
Type=notify
User=ubuntu
WorkingDirectory=/home/ubuntu/SEU_REPO/back
EnvironmentFile=/home/ubuntu/SEU_REPO/back/.env
ExecStart=/home/ubuntu/SEU_REPO/back/venv/bin/gunicorn \
    main:app -w 1 -b 0.0.0.0:8000 --timeout 300 --preload
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl start rag-api
sudo systemctl status rag-api
```

### 6. Frontend no S3

```bash
# No Console AWS
# 1. Criar bucket S3
# 2. Habilitar Static Website Hosting
# 3. Configurar Bucket Policy (público)
# 4. Editar index.html → atualizar API_URL com IP da EC2
# 5. Upload de index.html + assets
# 6. Acessar via endpoint do S3
```

---

## ��� Configurações de Chunking

O projeto passou por **múltiplas iterações** para otimizar a qualidade do RAG:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `chunk_size` | 180 chars | Cada produto ocupa ~1 chunk isolado |
| `overlap` | 20 chars | Mantém contexto entre chunks sem duplicação excessiva |
| `top_k` | 3 | Retorna 3 chunks mais relevantes por query |
| `embeddings` | 384 dims | Equilíbrio entre qualidade e performance em CPU |

---

## �� Variáveis de Ambiente

| Variável | Descrição | Exemplo |
|---|---|---|
| `DB_HOST` | Endpoint do Aurora RDS | `mydb.xxx.rds.amazonaws.com` |
| `DB_PORT` | Porta PostgreSQL | `5432` |
| `DB_NAME` | Nome do banco | `postgres` |
| `DB_USER` | Usuário do banco | `postgres` |
| `DB_PASSWORD` | Senha do banco | `sua_senha` |
| `LLM_MODEL` | Modelo HuggingFace | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |

> ⚠️ **NUNCA** commitar o arquivo `.env` no GitHub. Ele está no `.gitignore`.

---

## ��� Troubleshooting

### API não sobe após restart

```bash
sudo journalctl -u rag-api -f
# Aguardar ~90 segundos — modelo LLM demora para carregar
```

### Worker Timeout no startup

```bash
# Aumentar timeout no comando gunicorn
gunicorn main:app -w 1 -b 0.0.0.0:8000 --timeout 300 --preload
```

### Erro de conexão com RDS

```bash
# Verificar Security Group do RDS
# Regra de entrada: PostgreSQL (5432) liberado para o IP da EC2
psql -h <RDS_ENDPOINT> -U postgres -d postgres -p 5432 -c "SELECT 1;"
```

### pgvector não encontrado

```sql
-- Conectar no banco e rodar:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Memória insuficiente para o LLM

```bash
# Verificar uso de memória
free -h

# Verificar swap ativo
swapon --show

# Ativar swap se necessário
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### IP da EC2 mudou entre sessões (AWS Academy)

O IP público da EC2 muda a cada nova sessão no AWS Academy. Após iniciar a sessão:

```bash
# 1. Verificar IP atual no Console AWS → EC2 → Instances
# 2. Reiniciar o serviço se necessário
sudo systemctl restart rag-api

# 3. Atualizar o campo API URL no frontend (interface do chat)
```

---

## ��� Estrutura do Projeto

```
.
├── back/
│   ├── main.py              # API Flask + RAG + LLM
│   ├── requirements.txt     # Dependências Python
│   ├── Dockerfile           # Container da API
│   ├── .env.sample          # Template de variáveis
│   └── .gitignore
├── front/
│   ├── index.html           # Interface do chat
│   └── assets/              # Logo e recursos estáticos
├── docs/
│   └── estoque_iceman.txt   # Base de conhecimento (exemplo)
└── README.md
```

---

## ��� Lições Aprendidas

### Problemas encontrados e soluções

| Problema | Causa | Solução |
|---|---|---|
| Worker Timeout no startup | LLM demora ~90s para carregar | `--timeout 300 --preload` no Gunicorn |
| LLM consome muita memória | Modelo grande + float32 | `torch_dtype=float16` + 1 worker |
| RAG retornava chunks misturados | `chunk_size=500` colocava 4+ produtos por chunk | Reduzir para `chunk_size=180` |
| Alucinação de preços/dados | SmolLM2-360M limitado para extração numérica | Upgrade para SmolLM2-1.7B-Instruct |
| IP da EC2 muda entre sessões | AWS Academy reinicia instâncias | Campo configurável de API URL no frontend |
| `.env` com credenciais | Risco de vazar no GitHub | `.gitignore` defensivo |
| Distâncias altas no retrieval (0.92+) | Documento com muitos campos por produto | Documento enxuto foco em nome/preço/estoque/loja |

### Decisões de arquitetura

**Por que SmolLM2 local em vez de API externa?**
Para demonstrar um pipeline RAG completo e auto-suficiente. Em produção, um modelo via API (Claude, GPT-4) seria preferível para maior precisão.

**Por que chunk_size=180?**
Com produtos de ~120-150 chars, chunk_size=180 garante que cada produto ocupa majoritariamente um chunk isolado, melhorando a precisão do retrieval semântico.

**Por que pgvector em vez de Pinecone/Chroma?**
Reduz a infraestrutura — um único serviço RDS já cobre banco relacional + banco vetorial. Menos pontos de falha, menor custo.

---

## ��� Equipe

| Nome | Papel |
|---|---|
| [Nome 1] | Infra AWS (EC2, RDS, Security Groups, S3) |
| [Nome 2] | Backend / API (Flask, RAG, LLM) |
| [Nome 3] | Frontend (Interface, integração com API) |
| [Nome 4] | Docs e Testes (README, validação, apresentação) |

---

## ��� Links

- **Repositório:** [github.com/orugian/GS-CloudComputing](https://github.com/orugian/GS-CloudComputing.git)
- **Repositório base:** [github.com/arquitetoitamar/aula-2-api](https://github.com/arquitetoitamar/aula-2-api)
- **Documentação do projeto:** [arquitetoitamar.github.io/aula-2-api](https://arquitetoitamar.github.io/aula-2-api/)
- **Frontend (S3):** [URL do bucket S3]

---

## ��� Licença

Projeto acadêmico — FIAP 2026
