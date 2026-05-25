# IceMan Intelligent Catalog — Assistente IA com RAG na AWS


<img width="1981" height="794" alt="Adobe Express - file" src="https://github.com/user-attachments/assets/4c1ef540-1d69-413b-a614-b33349ee80ac" />

---

## Visão Geral

Sistema de **Assistente de IA com Retrieval-Augmented Generation (RAG)** deployado integralmente na AWS, desenvolvido como trabalho final de semestre.

O sistema permite fazer upload de documentos (PDF/TXT), indexá-los em um banco vetorial e realizar consultas em linguagem natural com respostas geradas por um LLM via API externa.

**Caso de uso:** catálogo inteligente da loja fictícia **IceMan Luxury Watches** — consultas sobre preços, estoque e localização de produtos.

---

## 🥷 Arquitetura

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
│                   API (EC2)                                 │
│         Flask + Gunicorn                                    │
│         paraphrase-multilingual-MiniLM-L12-v2 (local)      │
│         IBM Granite 4.1 8B via OpenRouter API               │
└──────────┬───────────────────────────┬──────────────────────┘
           │ PostgreSQL :5432          │ HTTPS
┌──────────▼──────────┐    ┌───────────▼──────────────────────┐
│  BANCO VETORIAL     │    │       OPENROUTER API             │
│  Aurora PostgreSQL  │    │  ibm-granite/granite-4.1-8b      │
│  pgvector           │    │  Geração de respostas            │
│  384 dims           │    └──────────────────────────────────┘
└─────────────────────┘
```

### Fluxo RAG

```
1. Upload documento (PDF/TXT)
        ↓
2. Extração de texto
        ↓
3. Chunking semântico por seção/produto
        ↓
4. Geração de embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384 dims)
        ↓
5. Armazenamento no pgvector (Aurora PostgreSQL)
        ↓
─── QUERY ───────────────────────────────────────────
        ↓
6. Usuário faz pergunta em português
        ↓
7. Embedding da pergunta (mesmo modelo multilingual)
        ↓
8. Busca híbrida: keyword filter + similaridade cosseno (top-5)
        ↓
9. Montagem do contexto + prompt em português
        ↓
10. IBM Granite 4.1 8B (OpenRouter) gera resposta
        ↓
11. Retorno: resposta + fontes ao usuário
```

---

## Stack Tecnológica

| Componente         | Tecnologia                          | Detalhe                              |
|--------------------|-------------------------------------|--------------------------------------|
| **Cloud**          | AWS EC2 + Aurora RDS + S3           | —                                    |
| **API**            | Flask + Gunicorn                    | Flask 3.x, 1 worker, timeout 300s    |
| **Embeddings**     | paraphrase-multilingual-MiniLM-L12-v2 | 384 dims, 50+ idiomas, local       |
| **LLM**            | IBM Granite 4.1 8B                  | Via OpenRouter API (zero RAM da EC2) |
| **Busca**          | Busca híbrida keyword + cosseno     | pgvector operador `<=>`, top-5       |
| **Banco Vetorial** | PostgreSQL + pgvector               | Aurora PostgreSQL 15                 |
| **Frontend**       | HTML/CSS/JS estático                | S3 Static Website Hosting            |
| **Runtime**        | Python 3.11                         | Ubuntu 24.04                         |

---

## ☁️ Infraestrutura AWS

| Serviço               | Configuração                              | Função            |
|-----------------------|-------------------------------------------|-------------------|
| **EC2**               | t3.medium — Ubuntu 24.04                  | API + embedder    |
| **Aurora PostgreSQL** | db.t3.medium — PostgreSQL 15 + pgvector   | Banco vetorial    |
| **S3**                | Static Website Hosting habilitado         | Frontend          |
| **Security Groups**   | Porta 22 (SSH) + 8000 (API)               | Segurança de rede |

---

## Endpoints da API

| Método | Endpoint     | Função                                            |
|--------|--------------|---------------------------------------------------|
| `GET`  | `/api/data`  | Health check                                      |
| `POST` | `/upload`    | Upload PDF/TXT → chunking → embeddings → pgvector |
| `GET`  | `/documents` | Lista documentos indexados                        |
| `POST` | `/ask`       | Busca semântica em documento específico           |
| `POST` | `/ask_all`   | Busca semântica em todos os documentos            |
| `POST` | `/chat`      | RAG completo: retrieval + LLM → resposta          |

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
     -d '{"question": "Qual o preço do Rolex Submariner?"}'
```

---

## Deploy — Passo a Passo

### Pré-requisitos

- Conta AWS ativa
- Par de chaves EC2 (.pem)
- Python 3.11+
- Git
- Chave OpenRouter (https://openrouter.ai)

### 1. EC2 — Configuração inicial

```bash
# Conectar na EC2
ssh -i chave.pem ubuntu@<EC2_IP>

# Atualizar sistema
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clonar repositório
git clone https://github.com/orugian/GS-CloudComputing
cd GS-CloudComputing/back

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
OPENROUTER_API_KEY=<sua_chave_openrouter>
OPENROUTER_MODEL=ibm-granite/granite-4.1-8b
```

### 4. Configurar como serviço systemd

```bash
sudo nano /etc/systemd/system/rag-api.service
```

```ini
[Unit]
Description=RAG API - Assistente IA IceMan
After=network.target

[Service]
Type=notify
User=ubuntu
WorkingDirectory=/home/ubuntu/GS-CloudComputing/back
EnvironmentFile=/home/ubuntu/GS-CloudComputing/back/.env
ExecStart=/home/ubuntu/GS-CloudComputing/back/venv/bin/gunicorn \
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

### 5. Upload do documento e validação

```bash
# Aguardar startup completo (~2-3 min para baixar embedder)
sudo journalctl -u rag-api -f --no-pager
# Aguardar: "[STARTUP] SentenceTransformer OK" → Ctrl+C

# Indexar documento
curl -X POST http://localhost:8000/upload \
     -F "file=@/home/ubuntu/estoque_iceman.txt"

# Testar
curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Qual o preço do Rolex Submariner?"}' \
     | python3 -m json.tool
```

### 6. Frontend no S3

```bash
# No Console AWS:
# 1. Criar bucket S3
# 2. Habilitar Static Website Hosting
# 3. Configurar Bucket Policy (público)
# 4. Upload de index.html + Logo.png
# 5. Acessar via endpoint do S3
# 6. Configurar o IP da EC2 no campo API URL da interface
```
---

##  Interface da Aplicação

<img width="2514" height="924" alt="image" src="https://github.com/user-attachments/assets/bf5900a8-100d-4d06-b359-b2b339978821" />


---


## Configurações de RAG

| Parâmetro         | Valor                                      | Justificativa                                      |
|-------------------|--------------------------------------------|----------------------------------------------------|
| `embedder`        | paraphrase-multilingual-MiniLM-L12-v2      | Suporte nativo a português (50+ idiomas)           |
| `chunk_strategy`  | Semântico por seção/produto                | Garante que nome + preço + estoque ficam no mesmo chunk |
| `top_k`           | 5                                          | Aumenta recall sem degradar o contexto do LLM      |
| `metric`          | Cosseno (`<=>`)                            | Modelo multilingual requer cosseno, não Euclidiana |
| `search_strategy` | Híbrida: keyword filter + similaridade     | Resolve queries com entidades nomeadas (marcas, lojas) |
| `dims`            | 384                                        | Equilíbrio entre qualidade e performance em CPU    |

---

## Variáveis de Ambiente

| Variável              | Descrição                  | Exemplo                          |
|-----------------------|----------------------------|----------------------------------|
| `DB_HOST`             | Endpoint do Aurora RDS     | `mydb.xxx.rds.amazonaws.com`     |
| `DB_PORT`             | Porta PostgreSQL            | `5432`                           |
| `DB_NAME`             | Nome do banco              | `postgres`                       |
| `DB_USER`             | Usuário do banco           | `postgres`                       |
| `DB_PASSWORD`         | Senha do banco             | `sua_senha`                      |
| `OPENROUTER_API_KEY`  | Chave da API OpenRouter    | `sk-or-v1-...`                   |
| `OPENROUTER_MODEL`    | Modelo LLM via OpenRouter  | `ibm-granite/granite-4.1-8b`     |

> ⚠️ **NUNCA** commitar o arquivo `.env` no GitHub. Ele está no `.gitignore`.

---

## Troubleshooting

### API não sobe após restart

```bash
sudo journalctl -u rag-api -f --no-pager
# Aguardar ~2-3 min — embedder multilingual (~420MB) demora para baixar na 1ª vez
```

### Worker Timeout no startup

```bash
# Garantir --timeout 300 --preload no Gunicorn
# O preload carrega o modelo antes de aceitar requests
```

### Erro de conexão com RDS

```bash
# Verificar Security Group do RDS
# Regra de entrada: PostgreSQL (5432) liberado para o IP da EC2
psql -h <RDS_ENDPOINT> -U postgres -d postgres -p 5432 -c "SELECT 1;"
```

### pgvector não encontrado

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Retrieval retornando chunks incorretos

```bash
# Verificar se o banco foi limpo antes de reindexar com novo embedder
psql -h <RDS_ENDPOINT> -U postgres -d postgres -p 5432 \
     -c "SELECT filename, COUNT(*) FROM documents GROUP BY filename;"

# Se necessário, limpar e reindexar
psql -h <RDS_ENDPOINT> -U postgres -d postgres -p 5432 \
     -c "DELETE FROM documents;"

curl -X POST http://localhost:8000/upload \
     -F "file=@/home/ubuntu/estoque_iceman.txt"
```

### IP da EC2 mudou entre sessões (AWS Academy)

```bash
# O IP público muda a cada nova sessão
# 1. Verificar IP atual: Console AWS → EC2 → Instances
# 2. Reiniciar serviço se necessário:
sudo systemctl restart rag-api
# 3. Atualizar o campo API URL no frontend (interface do chat)
```

### Resposta da OpenRouter com erro

```bash
# Verificar chave e modelo no .env
grep "OPENROUTER" ~/GS-CloudComputing/back/.env

# Testar diretamente
curl -s -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "teste"}' | python3 -m json.tool
```

---

## Lições Aprendidas

| Problema | Causa | Solução |
|----------|-------|---------|
| Distance alta no retrieval (~1.06) | Embedder inglês para queries em português | Trocar para `paraphrase-multilingual-MiniLM-L12-v2` |
| Preço errado na resposta | Chunk cortando produto no meio | Chunking semântico por seção/produto |
| Entidades nomeadas sem retrieval | Score cosseno insuficiente para marcas específicas | Busca híbrida: keyword filter + semântica |
| Métrica errada causando distances altas | `<->` (Euclidiana) incompatível com modelo multilingual | Trocar para `<=>` (Cosseno) |
| Worker Timeout no startup | Embedder demora para carregar | `--timeout 300 --preload` no Gunicorn |
| IP da EC2 muda entre sessões | AWS Academy reinicia instâncias | Campo configurável de API URL no frontend |
| `.env` com credenciais | Risco de vazar no GitHub | `.gitignore` defensivo |

### Decisões de arquitetura

**Por que LLM via API (OpenRouter) em vez de local?** O SmolLM2 local consumia toda a RAM da EC2 t3.medium, deixando pouco espaço para o embedder e o processo Flask. Com OpenRouter, a EC2 roda apenas o embedder local (~420MB) e delega a geração ao IBM Granite 4.1 8B na nuvem — melhor qualidade de resposta, zero custo de RAM.

**Por que chunking semântico em vez de tamanho fixo?** Com tamanho fixo (180 ou 500 chars), o nome do produto e seu preço ficavam em chunks diferentes, causando alucinações. O chunking por seção garante que todos os dados de um produto (nome, preço, estoque, lojas) ficam no mesmo vetor.

**Por que busca híbrida?** Para entidades nomeadas (marcas, lojas), a similaridade cosseno pura pode ranquear chunks irrelevantes mais alto que os corretos. O filtro keyword garante que a busca semântica aconteça apenas no subconjunto relevante.

**Por que pgvector em vez de Pinecone/Chroma?** Reduz a infraestrutura — um único serviço RDS já cobre banco relacional + banco vetorial. Menos pontos de falha, menor custo operacional.


---

## 🔗 Links

- **Repositório:** [github.com/orugian/GS-CloudComputing](https://github.com/orugian/GS-CloudComputing)
- **Repositório base:** [github.com/arquitetoitamar/aula-2-api](https://github.com/arquitetoitamar/aula-2-api)
- **Frontend (S3):** http://rag-frontend-2026.s3-website-us-east-1.amazonaws.com/

---

## 📄 Licença

Projeto acadêmico — FIAP 2026
