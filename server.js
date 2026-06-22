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

// -------------------------------------------------------------------
// Corrective RAG (CRAG) Nodes
// -------------------------------------------------------------------

/**
 * Grade the relevance of retrieved chunks to the question using the LLM.
 * Returns array of { chunk: number, relevant: boolean }
 */
async function gradeChunks(question, chunks, GITHUB_TOKEN) {
  const client = new OpenAI({
    baseURL: GITHUB_MODELS_BASE_URL,
    apiKey: GITHUB_TOKEN,
  });

  const chunksText = chunks
    .map((c, i) => `[Chunk ${i + 1}]\n${c.pageContent}`)
    .join("\n\n---\n\n");

  const systemPrompt = `You are an intelligent document relevance evaluator. Your job is to grade the relevance of retrieved document chunks to a user's question.
For each chunk, determine if it contains information directly relevant to answering the user's question.
You must output a JSON object with a single key "grades", which maps to an array of objects. Each object must have "chunk" (the chunk number, 1-indexed) and "relevant" (boolean: true/false).
Respond ONLY with the raw JSON object. Do not wrap it in markdown or add explanations.

Example Output Format:
{
  "grades": [
    { "chunk": 1, "relevant": true },
    { "chunk": 2, "relevant": false }
  ]
}`;

  const userPrompt = `Question: ${question}

Chunks to evaluate:
${chunksText}`;

  const modelsToTry = [
    "gpt-4o-mini",
    "gpt-4o",
    "Mistral-small",
    "Meta-Llama-3.1-8B-Instruct",
  ];

  for (const modelName of modelsToTry) {
    try {
      const completion = await client.chat.completions.create({
        model: modelName,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        temperature: 0.0,
      });

      let content = completion.choices[0].message.content.trim();
      if (content.startsWith("```")) {
        content = content.replace(/^```json\s*/i, "").replace(/```$/, "").trim();
      }

      const parsed = JSON.parse(content);
      if (parsed && Array.isArray(parsed.grades)) {
        console.log(`✅ Chunks graded successfully using: ${modelName}`);
        return parsed.grades;
      }
    } catch (err) {
      console.warn(`⚠️ Relevance grading failed with model "${modelName}": ${err.message}`);
    }
  }

  console.warn("⚠️ All relevance grading attempts failed. Defaulting to keeping all chunks.");
  return chunks.map((_, i) => ({ chunk: i + 1, relevant: true }));
}

/**
 * Perform a web search to fetch supplemental context.
 * Cascades from Tavily (key required) -> Wikipedia (free API) -> DuckDuckGo HTML scraping.
 */
async function webSearch(query) {
  const steps = [];
  
  // 1. Tavily Search
  if (process.env.TAVILY_API_KEY) {
    try {
      console.log("🔍 Triggering Tavily web search...");
      steps.push("Querying Tavily Web Search API...");
      const res = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: process.env.TAVILY_API_KEY,
          query: query,
          search_depth: "basic",
          max_results: 3,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.results && data.results.length > 0) {
          const results = data.results.map((r) => ({
            title: r.title,
            url: r.url,
            content: r.content,
          }));
          return { results, sourceUsed: "Tavily Search API", steps };
        }
      }
    } catch (err) {
      console.warn("⚠️ Tavily Search failed:", err.message);
    }
  }

  // 2. Wikipedia API Fallback
  try {
    console.log("🔍 Triggering Wikipedia Search fallback...");
    steps.push("Querying Wikipedia Search API...");
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(
      query
    )}&format=json&origin=*`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      const searchResults = data.query?.search || [];
      if (searchResults.length > 0) {
        const results = searchResults.slice(0, 3).map((r) => ({
          title: r.title,
          url: `https://en.wikipedia.org/wiki/${encodeURIComponent(r.title)}`,
          content: r.snippet.replace(/<\/?[^>]+(>|$)/g, ""), // strip HTML tags
        }));
        return { results, sourceUsed: "Wikipedia Search API", steps };
      }
    }
  } catch (err) {
    console.warn("⚠️ Wikipedia Search failed:", err.message);
  }

  // 3. DuckDuckGo HTML Scraper Fallback
  try {
    console.log("🔍 Triggering DuckDuckGo HTML Search fallback...");
    steps.push("Querying DuckDuckGo (HTML search interface)...");
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const res = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
    });

    if (res.ok) {
      const html = await res.text();
      const results = [];
      const snippetRegex = /<a class="result__snippet"[\s\S]*?>([\s\S]*?)<\/a>/g;
      
      let match;
      while ((match = snippetRegex.exec(html)) !== null && results.length < 3) {
        const content = match[1]
          .trim()
          .replace(/<\/?[^>]+(>|$)/g, "")
          .replace(/\s+/g, " ");
        results.push({
          title: `Web Result ${results.length + 1}`,
          url: "https://duckduckgo.com",
          content,
        });
      }

      if (results.length > 0) {
        return { results, sourceUsed: "DuckDuckGo Web Search", steps };
      }
    }
  } catch (err) {
    console.warn("⚠️ DuckDuckGo HTML Search failed:", err.message);
  }

  return { results: [], sourceUsed: "None (All web search attempts failed)", steps };
}

/**
 * POST /api/chat
 * Body: { collectionId, question, history? }
 * Retrieves the most relevant chunks from Qdrant, evaluates relevance
 * using Corrective RAG (CRAG), triggers search if necessary, and compiles context.
 */
app.post("/api/chat", async (req, res) => {
  try {
    const { collectionId, question, history } = req.body;

    if (!collectionId || !question) {
      return res.status(400).json({ error: "collectionId and question are required." });
    }

    const cragSteps = ["Retrieved top-5 matching document chunks."];

    // ---- 1. Retrieve relevant chunks ----
    let relevantChunks = [];
    try {
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
      relevantChunks = await retriever.invoke(question);
    } catch (dbErr) {
      console.warn("⚠️ Vector DB offline or connection failed. Falling back to search:", dbErr.message);
      cragSteps.push(`Vector DB connection issue: ${dbErr.message}`);
    }

    // ---- 2. Grade retrieved chunks relevance (CRAG Grader) ----
    let gradedRelevantChunks = [];
    let cragPath = "Correct";
    let searchResult = null;

    if (relevantChunks.length > 0) {
      const grades = await gradeChunks(question, relevantChunks, process.env.GITHUB_TOKEN);
      
      gradedRelevantChunks = relevantChunks.filter((_, idx) => {
        const grade = grades.find((g) => g.chunk === idx + 1);
        return grade ? grade.relevant : false;
      });

      const relevantCount = gradedRelevantChunks.length;
      cragSteps.push(`Evaluated relevance: ${relevantCount} of ${relevantChunks.length} chunks graded as relevant.`);

      if (relevantCount === relevantChunks.length) {
        // Correct path: Keep all documents, no web search
        cragPath = "Correct";
        cragSteps.push("Relevance check complete (100% confidence). No web search required.");
      } else if (relevantCount > 0) {
        // Ambiguous path: Keep relevant documents, supplement with search
        cragPath = "Ambiguous";
        cragSteps.push("Ambiguous relevance context. Triggering web search to supplement document content.");
        searchResult = await webSearch(question);
        cragSteps.push(...searchResult.steps);
        cragSteps.push(`Supplemented context using results from: ${searchResult.sourceUsed}`);
      } else {
        // Incorrect path: Discard all documents, replace with search
        cragPath = "Incorrect";
        cragSteps.push("All document chunks graded as irrelevant. Rejecting local context and replacing with web search.");
        searchResult = await webSearch(question);
        cragSteps.push(...searchResult.steps);
        cragSteps.push(`Using web search results from: ${searchResult.sourceUsed}`);
      }
    } else {
      // Empty DB / No chunks retrieved
      cragPath = "Incorrect";
      cragSteps.push("No chunks retrieved from the document database. Defaulting to web search.");
      searchResult = await webSearch(question);
      cragSteps.push(...searchResult.steps);
      cragSteps.push(`Using web search results from: ${searchResult.sourceUsed}`);
    }

    // ---- 3. Build context and dynamic prompt ----
    const context = gradedRelevantChunks
      .map((chunk, i) => {
        const page = chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A";
        return `[Chunk ${i + 1} | Page ${page}]\n${chunk.pageContent}`;
      })
      .join("\n\n---\n\n");

    let systemPrompt;
    if (searchResult && searchResult.results.length > 0) {
      const searchContext = searchResult.results
        .map((r, i) => `[Web Search Result ${i + 1} | Source: ${r.title} (${r.url})]\n${r.content}`)
        .join("\n\n---\n\n");

      systemPrompt = `You are an intelligent document assistant. Your job is to answer the user's question using the retrieved document context and/or the web search results provided.
      
RULES:
1. Synthesize the provided document context and web search results to answer the question.
2. If using web search results, explicitly state or cite that the information comes from web search (e.g. "[Web Search: Source Name]").
3. If using local document context, reference the page number.
4. Be concise, clear, and accurate.
5. Use markdown formatting for better readability (headings, lists, bold text, code blocks as appropriate).
6. Do NOT make up information or use general knowledge not present in the context.

LOCAL DOCUMENT CONTEXT:
${context || "No relevant content found in local document."}

SUPPLEMENTAL WEB SEARCH RESULTS:
${searchContext}`;
    } else {
      systemPrompt = `You are an intelligent document assistant. Your job is to answer the user's question using ONLY the context retrieved from their uploaded document.

RULES:
1. Answer ONLY based on the provided context from the document.
2. If the answer is not found in the context, say: "I couldn't find information about that in the uploaded document."
3. When possible, reference the page number where the information was found.
4. Be concise, clear, and accurate.
5. Use markdown formatting for better readability.
6. Do NOT make up information or use your general knowledge.

CONTEXT FROM DOCUMENT:
${context}`;
    }

    // ---- 4. Generate answer using LLM ----
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

    const messages = [{ role: "system", content: systemPrompt }];

    if (history && Array.isArray(history)) {
      const recentHistory = history.slice(-12);
      messages.push(...recentHistory);
    }

    messages.push({ role: "user", content: question });

    let answer = null;
    for (const modelName of modelsToTry) {
      try {
        const completion = await client.chat.completions.create({
          model: modelName,
          messages,
          temperature: 0.3,
        });

        answer = completion.choices[0].message.content;
        console.log(`✅ Response generated using model: ${modelName} (CRAG Path: ${cragPath})`);
        break;
      } catch (modelErr) {
        console.warn(`⚠️ Model "${modelName}" failed: ${modelErr.message}`);
      }
    }

    if (!answer) {
      throw new Error("All LLM models failed. Please try again later.");
    }

    // ---- 5. Compile and return results ----
    const sources = gradedRelevantChunks.map((chunk, i) => ({
      chunk: i + 1,
      page: chunk.metadata?.loc?.pageNumber ?? chunk.metadata?.page ?? "N/A",
      preview: chunk.pageContent.substring(0, 150) + "…",
    }));

    return res.json({
      answer,
      sources,
      searchSources: searchResult ? searchResult.results : [],
      cragSteps,
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
