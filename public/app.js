// ============================================================
// DocuMind — Frontend Application Logic
// ============================================================

const API_BASE = "";

// ------- State -------
let activeCollectionId = null;
let chatHistory = []; // {role, content}[]
const documents = []; // {collectionId, name, pages, chunks}[]

// ------- DOM references -------
const heroSection = document.getElementById("hero-section");
const featuresSection = document.getElementById("features");
const howSection = document.getElementById("how-it-works");
const appSection = document.getElementById("app-section");
const navbar = document.getElementById("navbar");

const getStartedBtn = document.getElementById("get-started-btn");
const backToLandingBtn = document.getElementById("back-to-landing");
const newUploadBtn = document.getElementById("new-upload-btn");

const uploadPanel = document.getElementById("upload-panel");
const chatPanel = document.getElementById("chat-panel");

const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadProgress = document.getElementById("upload-progress");
const progressFilename = document.getElementById("progress-filename");
const progressStatus = document.getElementById("progress-status");
const progressFill = document.getElementById("progress-fill");

const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const chatDocName = document.getElementById("chat-doc-name");
const chatDocMeta = document.getElementById("chat-doc-meta-text");
const documentsList = document.getElementById("documents-list");

// ------- Navigation -------
getStartedBtn.addEventListener("click", () => {
  heroSection.classList.add("hidden");
  featuresSection.classList.add("hidden");
  howSection.classList.add("hidden");
  navbar.classList.add("hidden");
  appSection.classList.remove("hidden");
});

backToLandingBtn.addEventListener("click", () => {
  appSection.classList.add("hidden");
  heroSection.classList.remove("hidden");
  featuresSection.classList.remove("hidden");
  howSection.classList.remove("hidden");
  navbar.classList.remove("hidden");
});

newUploadBtn.addEventListener("click", () => {
  showUploadPanel();
});

// ------- File Upload -------
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleFileUpload(file);
  fileInput.value = "";
});

async function handleFileUpload(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "txt"].includes(ext)) {
    alert("Only PDF and TXT files are supported.");
    return;
  }

  if (file.size > 20 * 1024 * 1024) {
    alert("File size exceeds 20 MB limit.");
    return;
  }

  // Show progress
  uploadProgress.classList.remove("hidden");
  progressFilename.textContent = file.name;
  progressStatus.textContent = "Uploading…";
  progressFill.style.width = "10%";

  const formData = new FormData();
  formData.append("file", file);

  try {
    progressFill.style.width = "30%";
    progressStatus.textContent = "Processing & chunking…";

    const res = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData,
    });

    progressFill.style.width = "80%";
    progressStatus.textContent = "Embedding & indexing…";

    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Upload failed");

    progressFill.style.width = "100%";
    progressStatus.textContent = "Done ✓";

    // Register document
    const doc = {
      collectionId: data.collectionId,
      name: data.documentName,
      pages: data.pages,
      chunks: data.chunks,
    };
    documents.push(doc);
    renderDocumentsList();

    // Switch to chat
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      progressFill.style.width = "0%";
      switchToDocument(doc.collectionId);
    }, 600);
  } catch (err) {
    progressStatus.textContent = `Error: ${err.message}`;
    progressFill.style.background = "var(--error)";
    setTimeout(() => {
      uploadProgress.classList.add("hidden");
      progressFill.style.width = "0%";
      progressFill.style.background = "";
    }, 3000);
  }
}

// ------- Documents sidebar -------
function renderDocumentsList() {
  if (documents.length === 0) {
    documentsList.innerHTML = '<p class="sidebar-empty">No documents yet.</p>';
    return;
  }

  documentsList.innerHTML = documents
    .map(
      (doc) => `
    <div
      class="doc-item ${doc.collectionId === activeCollectionId ? "active" : ""}"
      data-id="${doc.collectionId}"
      title="${doc.name}"
    >
      <span class="doc-item-icon">📄</span>
      <span class="doc-item-name">${doc.name}</span>
    </div>
  `
    )
    .join("");

  // Attach click listeners
  documentsList.querySelectorAll(".doc-item").forEach((el) => {
    el.addEventListener("click", () => {
      switchToDocument(el.dataset.id);
    });
  });
}

function switchToDocument(collectionId) {
  activeCollectionId = collectionId;
  chatHistory = [];

  const doc = documents.find((d) => d.collectionId === collectionId);
  if (!doc) return;

  // Update header
  chatDocName.textContent = doc.name;
  chatDocMeta.textContent = `${doc.pages} page${doc.pages > 1 ? "s" : ""} · ${doc.chunks} chunks`;

  // Reset messages
  chatMessages.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">💬</div>
      <h3>Document ready!</h3>
      <p>Ask any question about <strong>${doc.name}</strong> and I'll find the answer from its content.</p>
    </div>
  `;

  // Show chat panel
  uploadPanel.classList.add("hidden");
  chatPanel.classList.remove("hidden");

  renderDocumentsList();
  chatInput.focus();
}

function showUploadPanel() {
  activeCollectionId = null;
  uploadPanel.classList.remove("hidden");
  chatPanel.classList.add("hidden");
  renderDocumentsList();
}

// ------- Chat -------
chatInput.addEventListener("input", () => {
  // Auto-resize
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";

  sendBtn.disabled = chatInput.value.trim().length === 0;
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (chatInput.value.trim()) sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || !activeCollectionId) return;

  // Clear welcome message if present
  const welcome = chatMessages.querySelector(".welcome-message");
  if (welcome) welcome.remove();

  // Add user message
  appendMessage("user", question);
  chatHistory.push({ role: "user", content: question });

  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;

  // Show typing indicator
  const typingEl = showTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        collectionId: activeCollectionId,
        question,
        history: chatHistory.slice(-12),
      }),
    });

    const data = await res.json();
    removeTypingIndicator(typingEl);

    if (!res.ok) throw new Error(data.error || "Chat failed");

    appendMessage("assistant", data.answer, data.sources, data.searchSources, data.cragSteps);
    chatHistory.push({ role: "assistant", content: data.answer });
  } catch (err) {
    removeTypingIndicator(typingEl);
    appendMessage("assistant", `⚠️ Error: ${err.message}`);
  }
}

function appendMessage(role, content, sources, searchSources, cragSteps) {
  const msgEl = document.createElement("div");
  msgEl.className = `message ${role}`;

  const avatarText = role === "user" ? "Y" : "🤖";

  let cragTraceHtml = "";
  if (role === "assistant" && cragSteps && cragSteps.length > 0) {
    const stepItems = cragSteps.map(step => `<li>${escapeHtml(step)}</li>`).join("");
    cragTraceHtml = `
      <div class="crag-trace-container">
        <div class="crag-trace-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.classList.toggle('active')">
          <span class="crag-icon">⚙️</span> CRAG Pipeline Trace <span class="chevron">▼</span>
        </div>
        <ul class="crag-trace-list">${stepItems}</ul>
      </div>
    `;
  }

  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    const sourceItems = sources
      .map(
        (s) =>
          `<div class="source-item"><span class="source-badge">Page ${s.page}</span>${escapeHtml(s.preview)}</div>`
      )
      .join("");

    sourcesHtml = `
      <div class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
        📎 ${sources.length} document source${sources.length > 1 ? "s" : ""} referenced
      </div>
      <div class="sources-list">${sourceItems}</div>
    `;
  }

  let searchSourcesHtml = "";
  if (searchSources && searchSources.length > 0) {
    const searchItems = searchSources
      .map((s) => {
        let domain = "Web Search";
        try {
          if (s.url && s.url !== "https://duckduckgo.com") {
            domain = new URL(s.url).hostname;
          }
        } catch (_) {}
        return `
          <div class="source-item">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
              <span class="source-badge search-badge">Web: ${escapeHtml(domain)}</span>
              <a href="${escapeHtml(s.url)}" target="_blank" class="source-link" style="color: var(--accent-2); font-weight: 500; font-size: 0.8rem; text-decoration: underline;">${escapeHtml(s.title)}</a>
            </div>
            <div class="source-preview" style="font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4;">${escapeHtml(s.content)}</div>
          </div>
        `;
      })
      .join("");

    searchSourcesHtml = `
      <div class="sources-toggle search-sources-toggle" onclick="this.nextElementSibling.classList.toggle('open')">
        🌐 ${searchSources.length} web search source${searchSources.length > 1 ? "s" : ""} referenced
      </div>
      <div class="sources-list">${searchItems}</div>
    `;
  }

  const formattedContent = role === "assistant" ? renderMarkdown(content) : escapeHtml(content);

  msgEl.innerHTML = `
    <div class="message-avatar">${avatarText}</div>
    <div class="message-content">
      ${cragTraceHtml}
      <div class="message-text-body">${formattedContent}</div>
      ${sourcesHtml}
      ${searchSourcesHtml}
    </div>
  `;

  chatMessages.appendChild(msgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function removeTypingIndicator(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ------- Helpers -------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Very lightweight markdown renderer for assistant messages.
 * Handles: headers, bold, italic, inline code, code blocks, lists, links, line breaks.
 */
function renderMarkdown(text) {
  let html = escapeHtml(text);

  // Code blocks (```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, lang, code) => {
    return `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Headers
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Unordered lists
  html = html.replace(/^[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Line breaks — convert double newlines to paragraphs, single to <br>
  html = html
    .split("\n\n")
    .map((p) => {
      p = p.trim();
      if (!p) return "";
      // Don't wrap already-wrapped elements
      if (
        p.startsWith("<h") ||
        p.startsWith("<ul") ||
        p.startsWith("<ol") ||
        p.startsWith("<pre") ||
        p.startsWith("<li")
      )
        return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");

  return html;
}
