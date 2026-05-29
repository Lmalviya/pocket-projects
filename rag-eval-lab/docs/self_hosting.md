# 🐳 Self-Hosting Guide: Qdrant, Redis, and Langfuse Architecture

This guide provides a detailed engineering analysis of the self-hosted infrastructure stack required for the **RAG Eval Lab**. It covers service dependencies, database engines, caching layer benefits, prompt registry setup, and step-by-step instructions to spin up the entire cluster using Docker Compose.

---

## 1. Systems Architecture Overview

To run a fully production-grade, local-first RAG evaluation environment, we orchestrate four primary containers on a unified Docker virtual network.

```mermaid
graph TD
    subgraph Local Client
        CLI[Python main.py] -->|Retrieval / Embed 1024-dim| Ollama(Ollama Embeddings API: 11434)
        CLI -->|LLM Generation| NV[NVIDIA API Catalog]
    end

    subgraph Docker Virtual Network rag_network
        CLI -->|Hybrid Search| Qdrant[(Qdrant Vector DB: 6333)]
        CLI -->|Telemetry Traces + Prompt Registry| LF[Langfuse Web Server: 3000]
        
        subgraph Langfuse Dependencies
            LF -->|State / Telemetry Tables| PG[(PostgreSQL DB: 5432)]
            LF -->|Caching & Task Queuing| RD[(Redis Cache: 6379)]
        end
    end
    
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    class PG,Qdrant,RD storage;
```

---

## 2. Deep-Dive: Langfuse Infrastructure & Dependencies

Langfuse is a robust, full-featured observability platform. Unlike lightweight script tracers, Langfuse runs an asynchronous web server and worker system. To host it successfully, you must satisfy three core dependencies:

### A. The Database Layer: PostgreSQL (Structured State)
* **What it does**: PostgreSQL is the primary transactional database for Langfuse. It stores users, projects, prompts, API keys, metadata tables, and the raw structured trace/span timelines.
* **Why PostgreSQL?**: Observability data requires complex relational linking (e.g. mapping a parent Trace to many nested Spans, linking Spans to specific Prompt versions, and associating scores with Trace IDs). PostgreSQL's ACID compliance and powerful JSONB querying make it the perfect engine for structured trace analytics.
* **System Version Requirement**: PostgreSQL version 12 or higher is required. We use PostgreSQL 16 on alpine Linux for optimal performance and a small memory footprint.

### B. The Caching & Message Queue Layer: Redis (Speed & Async Work)
* **What it does**: Redis acts as an in-memory key-value cache and a message broker.
* **Why Redis is a critical Langfuse dependency**:
  1. **Asynchronous Background Processing**: When your Python pipeline fires 10 concurrent tracing spans, we don't want the LLM generation to wait for Langfuse to write to PostgreSQL. The Langfuse SDK buffers these logs and pushes them in bulk. Langfuse uses Redis as a high-speed message queue (via background workers) to ingest trace events instantly in memory and commit them to PostgreSQL asynchronously in the background.
  2. **API Rate Limiting**: If you benchmark 50 evaluation tasks concurrently, the API traffic spikes. Langfuse uses Redis's atomic increments to manage token bucket rate-limiting and protect the web servers.
  3. **Session Caching**: Caches user sessions, project configurations, and active prompt versions to prevent hammering the PostgreSQL database.

### C. The Authentication & Security Layer: NextAuth Cryptography
Langfuse is built on Next.js and uses NextAuth for user authentication. It requires two cryptographic seeds:
* **`NEXTAUTH_SECRET`**: A private cryptographic seed used to encrypt and sign session cookies, preventing session hijacking or user spoofing.
* **`SALT`**: A private cryptographic salt used to hash API keys (both Public and Secret keys) before committing them to the PostgreSQL database. If your database is ever compromised, the attacker cannot read or use the raw API keys because they are cryptographically hashed using this salt.

---

## 3. Qdrant & Redis Hosting Specs

### A. Qdrant Vector Database
* **Configuration**: Runs on port `6333` (REST API) and port `6334` (gRPC).
* **Storage Preservation**: Vectors and payloads are highly optimized on-disk structures. We bind a persistent Docker volume (`qdrant_storage`) to `/qdrant/storage` so your stage-isolated collections survive container restarts.
* **Collection Naming**: Each experiment combination creates an isolated collection named:
  ```
  rag_eval_lab_stage_{N}_{chunking}_{sparse_strategy}
  ```
  For example: `rag_eval_lab_stage_1_fixed_splade`

### B. Redis Cache
* **Configuration**: Runs on standard port `6379`.
* **Storage Preservation**: Configured to run on Redis 7 Alpine, storing cache snapshots in a persistent volume (`redis_data`) under `/data`.

---

## 4. The Unified Docker Compose Configuration

To launch this complete cluster, we define a unified `docker-compose.yml` file. This creates an isolated virtual bridge network (`rag_network`) and maps all volume persistence points.

### Environment Secrets Template
Before booting, Langfuse expects these variables. They are injected directly in the `docker-compose.yml` or read from a local `.env` file:

| Variable | Value |
|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@postgres:5432/langfuse` |
| `REDIS_URL` | `redis://redis:6379` |
| `NEXTAUTH_URL` | `http://localhost:3000` |
| `NEXTAUTH_SECRET` | A 32-character hex key |
| `SALT` | A 32-character hex key |

Generate secrets with:
```bash
openssl rand -hex 32
```

---

## 5. Deployment & Verification Steps

### Step 1: Boot the Infrastructure
From the root of `rag-eval-lab/` (where `docker-compose.yml` is located), run:
```bash
docker compose up -d
```
*The `-d` flag runs all containers in "detached" daemon mode in the background.*

### Step 2: Verify Service Status
To check if all containers are healthy, run:
```bash
docker compose ps
```

You should see four healthy services:
1. **`postgres`** (Port 5432) — Operational
2. **`redis`** (Port 6379) — Operational
3. **`qdrant`** (Port 6333) — Operational
4. **`langfuse`** (Port 3000) — Operational

### Step 3: Access the Dashboards
* **Langfuse Observer Dashboard**: Open `http://localhost:3000` in your web browser.
  * Click **Sign Up** to create your first local admin account.
  * Create a new project named `"RAG Eval Lab"`.
  * Go to **Settings** → **API Credentials** → **Create API Keys**.
  * Copy the **Public Key**, **Secret Key**, and **Host** (`http://localhost:3000`) and paste them into your project's `.env` file.
* **Qdrant Admin Panel**: Open `http://localhost:6333/dashboard` to view, check, and delete your stage-isolated collections visually.

---

## 6. Langfuse Prompt Registry Setup

The application fetches versioned prompts from the Langfuse **Prompt Registry** at runtime. This allows you to change the system prompt without redeploying the application. If Langfuse is unreachable, the app falls back to local YAML files in `app/generation/prompts/`.

> [!IMPORTANT]
> Prompts must be created **and labeled as `production`** in the Langfuse UI or the SDK fetch will return a 404. The application calls `client.get_prompt("<name>", label="production")`.

### Creating the Required Prompts

Navigate to **`http://localhost:3000`** → your project → **Prompts** → **New Prompt**.

#### Prompt 1: `rag-qna-prompt` (Single-Turn Q&A)
* **Type**: Chat
* **Name**: `rag-qna-prompt`
* **System message**:
  ```
  You are a precise factual assistant. Answer the user's question using ONLY the context provided below.
  If the context does not contain enough information, say "I don't know based on the provided context."
  Do not fabricate answers.

  Context:
  {context}
  ```
* **User message**:
  ```
  {question}
  ```

#### Prompt 2: `rag-conversation-prompt` (Multi-Turn Chat)
* **Type**: Text
* **Name**: `rag-conversation-prompt`
* **Content**:
  ```
  You are a helpful, conversational factual assistant. You have access to a retrieved context below.
  Use it to answer questions accurately. When the user references earlier messages, use the conversation history.
  If uncertain, say so rather than guessing.

  Context:
  {context}
  ```

### Labeling as `production`
After saving each prompt version:
1. Open the prompt → click on your saved version.
2. Find **Labels** → click **Add Label** → type `production` → confirm.

The label `production` is what the SDK resolves when no explicit label override is given.

---

## 7. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `LangfuseNotFoundError: Prompt not found 'rag-qna-prompt' with label 'production'` | Prompt exists but not labeled `production` | Go to Langfuse UI → Prompts → your prompt version → Add Label → `production` |
| `Context error: No active span in current context` | Span created but not entered as context manager | Fixed in `app/pipeline/single_turn.py` and `multi_turn.py` — upgrade to latest code |
| Qdrant collection created but empty | `build_cache()` was skipped, leaving missing Wikipedia articles | Bootstrap detects `points_count == 0` and triggers a full rebuild automatically |
| `Failed to export span batch code: 404` | OTel exporter hitting wrong Langfuse endpoint | Ensure `LANGFUSE_HOST=http://localhost:3000` in `.env` (not a sub-path) |
| SPLADE model download fails / connection reset | HuggingFace blocked by network | `HF_ENDPOINT` is set to `https://hf-mirror.com` in `main.py` automatically |
| `max_completion_tokens` 400 error from NVIDIA API | NVIDIA NIM does not support newer OpenAI param name | `CompatibleChatOpenAI` in `generation/single_turn.py` strips this automatically |
