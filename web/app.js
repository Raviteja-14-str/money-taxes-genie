const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const API_URL = window.MONEY_GENIE_API_URL || (isLocal ? "/api/chat" : "REPLACE_WITH_LAMBDA_FUNCTION_URL");
const form = document.querySelector("#composer");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const status = document.querySelector("#status");
const conversationHistory = [];

function addMessage(kind, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${kind}`;
  article.innerHTML = `<div class="avatar">${kind === "genie" ? "✦" : "You"}</div><div class="content"><p></p></div>`;
  article.querySelector("p").textContent = text;
  if (sources.length) {
    const sourceLine = document.createElement("small");
    sourceLine.append("Sources: ");
    sources.forEach((source, index) => {
      if (index) sourceLine.append(" · ");
      const label = source.doc_title || "Knowledge source";
      const url = Array.isArray(source.source_urls) ? source.source_urls[0] : source.source_url;
      if (url) {
        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = label;
        sourceLine.append(link);
      } else {
        sourceLine.append(label);
      }
    });
    article.querySelector(".content").append(sourceLine);
  }
  messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

async function ask(question) {
  const trimmed = question.trim();
  if (!trimmed) return;
  addMessage("user", trimmed);
  input.value = "";
  input.style.height = "auto";
  send.disabled = true;
  status.textContent = "Thinking…";
  try {
    if (API_URL.startsWith("REPLACE_")) throw new Error("The Lambda URL is not configured yet.");
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed, history: conversationHistory.slice(-6) })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The assistant could not respond.");
    addMessage("genie", data.answer, data.sources || []);
    conversationHistory.push({ role: "user", content: trimmed });
    conversationHistory.push({ role: "assistant", content: data.answer });
    if (conversationHistory.length > 12) conversationHistory.splice(0, conversationHistory.length - 12);
    status.textContent = "Ready for a question";
  } catch (error) {
    addMessage("genie error", `I couldn’t answer just now: ${error.message}`);
    status.textContent = "Needs configuration";
  } finally {
    send.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  ask(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => ask(button.dataset.question));
});
