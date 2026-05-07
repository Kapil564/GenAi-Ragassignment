import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import OpenAI from "openai";

// -------------------------------------------------------------------
// GitHub Models API — OpenAI-compatible endpoint
// Uses GITHUB_TOKEN for both embeddings and LLM generation
// -------------------------------------------------------------------
const GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com";

// -------------------------------------------------------------------
// Setup
// -------------------------------------------------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer storage config for file uploads
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, `${unique}-${file.originalname}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 }, // 20 MB max
  fileFilter: (_req, file, cb) => {
    const allowed = [".pdf", ".txt"];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Only PDF and TXT files are supported."));
    }
  },
});

// -------------------------------------------------------------------
// Shared helpers
// -------------------------------------------------------------------
const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const QDRANT_API_KEY = process.env.QDRANT_API_KEY || undefined;

function getEmbeddings() {
  return new OpenAIEmbeddings({
    openAIApiKey: process.env.GITHUB_TOKEN,
    model: "text-embedding-3-small",
    batchSize: 256,
    configuration: {
      baseURL: GITHUB_MODELS_BASE_URL,
    },
  });
}

/**
 * Derive a safe Qdrant collection name from the uploaded filename.
 * Qdrant collection names must be <= 255 chars, no slashes, etc.
 */
function collectionName(filename) {
  return filename
    .replace(/[^a-zA-Z0-9_-]/g, "_")
    .replace(/_+/g, "_")
    .substring(0, 200);
}

// -------------------------------------------------------------------
// In-memory registry of uploaded documents
// -------------------------------------------------------------------
const documentRegistry = new Map(); // collectionId → { name, pages, chunks, uploadedAt }

// -------------------------------------------------------------------
// Routes
// -------------------------------------------------------------------

/**
 * POST /api/upload
 * Accepts a single file (PDF or TXT), chunks it, embeds it, and stores
 * it in Qdrant. Returns a collectionId the frontend uses for chatting.
 *
 * Chunking strategy: RecursiveCharacterTextSplitter
 * -------------------------------------------------
 * We use LangChain's RecursiveCharacterTextSplitter which splits text
 * hierarchically using a list of separators: ["\n\n", "\n", " ", ""].
 * It first tries to split on double-newlines (paragraphs), then single
 * newlines, then spaces, and only as a last resort splits mid-word.
 * This preserves semantic coherence within each chunk far better than
 * a naive fixed-size split. We use a chunk size of 1000 characters
 * with 200 characters of overlap so that context at chunk boundaries
 * is not lost.
 */
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded." });
    }

    const filePath = req.file.path;
    const originalName = req.file.originalname;
    const ext = path.extname(originalName).toLowerCase();

    // ---- 1. Load document ----
    let rawDocs;
    if (ext === ".pdf") {
      const loader = new PDFLoader(filePath);
      rawDocs = await loader.load();
    } else {
      // Plain text
      const text = fs.readFileSync(filePath, "utf-8");
      rawDocs = [{ pageContent: text, metadata: { source: originalName, page: 1 } }];
    }

    // ---- 2. Chunk with RecursiveCharacterTextSplitter ----
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const docs = await splitter.splitDocuments(rawDocs);

    // Tag every chunk with the original filename for traceability
    docs.forEach((doc, idx) => {
      doc.metadata.source = originalName;
      doc.metadata.chunkIndex = idx;
    });

    // ---- 3. Embed & store in Qdrant ----
    const colName = collectionName(originalName);
    const embeddings = getEmbeddings();

    const qdrantConfig = {
      url: QDRANT_URL,
      collectionName: colName,
    };
    if (QDRANT_API_KEY) qdrantConfig.apiKey = QDRANT_API_KEY;

    await QdrantVectorStore.fromDocuments(docs, embeddings, qdrantConfig);

    // ---- 4. Register the document ----
    documentRegistry.set(colName, {
      name: originalName,
      pages: rawDocs.length,
      chunks: docs.length,
      uploadedAt: new Date().toISOString(),
    });

    // Clean up the uploaded file after indexing
    fs.unlinkSync(filePath);

    return res.json({
      success: true,
      collectionId: colName,
      documentName: originalName,
      pages: rawDocs.length,
      chunks: docs.length,
      message: `Document indexed successfully — ${docs.length} chunks stored.`,
    });
  } catch (err) {
    console.error("Upload error:", err);
    return res.status(500).json({ error: err.message || "Upload failed." });
  }
});

/**
 * POST /api/chat
 * Body: { collectionId, question, history? }
 * Retrieves the most relevant chunks from Qdrant and passes them to
 * the LLM to generate a grounded answer.
 */
app.post("/api/chat", async (req, res) => {
  try {
    const { collectionId, question, history } = req.body;

    if (!collectionId || !question) {
      return res.status(400).json({ error: "collectionId and question are required." });
    }

    // ---- 1. Retrieve relevant chunks ----
    const embeddings = getEmbeddings();

    const qdrantConfig = {
      url: QDRANT_URL,
      collectionName: collectionId,
    };
    if (QDRANT_API_KEY) qdrantConfig.apiKey = QDRANT_API_KEY;

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      qdrantConfig
    );

    const retriever = vectorStore.asRetriever({ k: 5 });
    const relevantChunks = await retriever.invoke(question);

    // ---- 2. Build context string ----
    const context = relevantChunks
      .map((chunk, i) => {
        const page = chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A";
        return `[Chunk ${i + 1} | Page ${page}]\n${chunk.pageContent}`;
      })
      .join("\n\n---\n\n");

    // ---- 3. Generate answer using LLM via GitHub Models (with model fallback) ----
    const client = new OpenAI({
      baseURL: GITHUB_MODELS_BASE_URL,
      apiKey: process.env.GITHUB_TOKEN,
    });

    const modelsToTry = [
      "gpt-4o-mini",
      "gpt-4o",
      "Mistral-small",
      "Meta-Llama-3.1-8B-Instruct",
    ];

    const systemPrompt = `You are an intelligent document assistant. Your job is to answer the user's question using ONLY the context retrieved from their uploaded document.

RULES:
1. Answer ONLY based on the provided context from the document.
2. If the answer is not found in the context, say: "I couldn't find information about that in the uploaded document."
3. When possible, reference the page number where the information was found.
4. Be concise, clear, and accurate.
5. Use markdown formatting for better readability (headings, lists, bold text, code blocks as appropriate).
6. Do NOT make up information or use your general knowledge.

CONTEXT FROM DOCUMENT:
${context}`;

    // Build messages array with optional conversation history
    const messages = [{ role: "system", content: systemPrompt }];

    if (history && Array.isArray(history)) {
      const recentHistory = history.slice(-12);
      messages.push(...recentHistory);
    }

    messages.push({ role: "user", content: question });

    // Try each model in order until one succeeds
    let answer = null;
    for (const modelName of modelsToTry) {
      try {
        const completion = await client.chat.completions.create({
          model: modelName,
          messages,
          temperature: 0.3,
        });

        answer = completion.choices[0].message.content;
        console.log(`✅ Response generated using model: ${modelName}`);
        break; // success — stop trying
      } catch (modelErr) {
        console.warn(`⚠️  Model "${modelName}" failed: ${modelErr.message}`);
        // continue to next model
      }
    }

    if (!answer) {
      throw new Error("All models failed. Please try again later.");
    }

    // ---- 4. Return answer with source references ----
    const sources = relevantChunks.map((chunk, i) => ({
      chunk: i + 1,
      page: chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A",
      preview: chunk.pageContent.substring(0, 150) + "…",
    }));

    return res.json({
      answer,
      sources,
    });
  } catch (err) {
    console.error("Chat error:", err);
    return res.status(500).json({ error: err.message || "Chat failed." });
  }
});

/**
 * GET /api/documents
 * Returns a list of documents that have been uploaded & indexed.
 */
app.get("/api/documents", (_req, res) => {
  const docs = [];
  for (const [id, info] of documentRegistry) {
    docs.push({ collectionId: id, ...info });
  }
  return res.json(docs);
});

// SPA fallback — Express 5 uses {*path} syntax for catch-all
app.get("/{*path}", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// -------------------------------------------------------------------
// Start
// -------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`\n🚀  RAG server running at  http://localhost:${PORT}\n`);
});
