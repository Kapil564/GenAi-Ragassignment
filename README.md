# 📄 DocuMind — RAG-Powered Document Q&A

> Upload any PDF or text document and have an intelligent conversation with it.  
> Answers are grounded in your document's actual content — not AI hallucinations.

![Node.js](https://img.shields.io/badge/Node.js-18+-green)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red)
![LangChain](https://img.shields.io/badge/LangChain-JS-yellow)

---

## 🎯 What It Does

DocuMind is a full **Retrieval-Augmented Generation (RAG)** application that lets you:

1. **Upload** a PDF or plain text document
2. **Process** it automatically — chunking, embedding, and indexing
3. **Ask questions** in natural language
4. **Get grounded answers** sourced directly from your document with page references

---

## 🏗️ Architecture — RAG Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌──────────┐
│   Upload     │ ──▶ │   Chunking   │ ──▶ │   Embedding          │ ──▶ │  Qdrant  │
│  (PDF/TXT)   │     │ Recursive    │     │ text-embedding-004   │     │ Vector DB│
└─────────────┘     │ CharSplitter │     └──────────────────────┘     └──────────┘
                     └──────────────┘                                       │
                                                                            ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌──────────┐
│   Answer     │ ◀── │  Generation  │ ◀── │   Retrieval          │ ◀── │  Query   │
│  + Sources   │     │ Gemini Flash │     │ Top-5 similar chunks │     │  Embed   │
└─────────────┘     └──────────────┘     └──────────────────────┘     └──────────┘
```

### Pipeline Steps

| Step | Technology | Description |
|------|-----------|-------------|
| **Ingestion** | `@langchain/community` PDFLoader | Extracts text from PDF pages; also supports plain `.txt` files |
| **Chunking** | `RecursiveCharacterTextSplitter` | Splits text using `["\n\n", "\n", " ", ""]` separators. Chunk size: 1000 chars, overlap: 200 chars. Preserves paragraph & sentence boundaries. |
| **Embedding** | `text-embedding-004` (Google) | Converts each chunk into a 768-dim vector embedding |
| **Storage** | Qdrant Vector Database | Indexes embeddings for cosine-similarity search |
| **Retrieval** | LangChain Retriever (k=5) | Embeds the user query and finds the 5 most relevant chunks |
| **Generation** | Gemini 2.0 Flash (Google) | Generates an answer strictly from retrieved context, with page refs |

---

## 📦 Chunking Strategy — Explained

We use **`RecursiveCharacterTextSplitter`** from LangChain, which is the recommended general-purpose text splitter.

### How it works:

1. **Hierarchical splitting** — It tries a list of separators in order:
   - `"\n\n"` (double newline — paragraph boundaries)
   - `"\n"` (single newline — line breaks)
   - `" "` (space — word boundaries)
   - `""` (character-level — last resort)

2. It splits on the **largest possible unit** first. If a paragraph fits within the chunk size (1000 chars), it keeps the entire paragraph intact.

3. **Overlap of 200 characters** ensures that context at chunk boundaries is not lost — if a sentence is split across two chunks, the overlapping portion appears in both.

### Why this strategy?

- **Preserves semantic coherence** — paragraphs and sentences stay together
- **Better retrieval quality** — whole paragraphs give the LLM more context
- **Handles diverse documents** — works well on reports, manuals, articles, textbooks

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** ≥ 18
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/apikey)
- **Qdrant** — Either:
  - Local: `docker run -p 6333:6333 qdrant/qdrant`
  - Cloud: [Free tier at cloud.qdrant.io](https://cloud.qdrant.io)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/RAG-Assignment.git
cd RAG-Assignment

# Install dependencies
npm install --legacy-peer-deps

# Create your .env file
cp .env.example .env
# Edit .env and add your API keys
```

### Configuration

Edit `.env`:

```env
GEMINI_API_KEY=your-gemini-key-here
QDRANT_URL=http://localhost:6333      # or your Qdrant Cloud URL
QDRANT_API_KEY=                        # required for Qdrant Cloud
PORT=3000
```

### Run Locally

```bash
# Development mode (auto-restart on file changes)
npm run dev

# Production mode
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🖥️ How to Use

1. Click **"Upload & Start Chatting"** on the landing page
2. Drag & drop (or click to browse) a **PDF** or **TXT** file
3. Wait for processing — the system will chunk, embed, and index your document
4. Type a question in the chat input
5. Get an answer with **source references** showing which pages the information came from

---

## 📁 Project Structure

```
RAG-Assignment/
├── server.js           # Express backend — RAG pipeline endpoints
├── public/
│   ├── index.html      # Web UI — landing page + chat interface
│   ├── style.css       # Design system — dark theme, animations
│   └── app.js          # Frontend logic — upload, chat, rendering
├── .env.example        # Environment variable template
├── .gitignore
├── package.json
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Express.js (Node.js) |
| Frontend | Vanilla HTML/CSS/JS |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | Google text-embedding-004 |
| Vector DB | Qdrant |
| Document Loading | LangChain PDFLoader |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| File Upload | Multer |

---

## 🌐 Deployment

### Deploy to Render (recommended)

1. Push your code to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `npm install --legacy-peer-deps`
   - **Start Command:** `npm start`
5. Add environment variables (`GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`)
6. Use [Qdrant Cloud](https://cloud.qdrant.io) (free tier) as your vector DB

---

## 📝 License

ISC
