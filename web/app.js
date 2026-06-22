// Leipzig-Experte — front-end: sends questions and renders the streamed
// reasoning steps (status, sources) and the answer token by token.

const chat = document.getElementById("chat");
const form = document.getElementById("composer-form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");
const welcome = document.getElementById("welcome");
const srLive = document.getElementById("sr-live");

let busy = false;

// Announce a message once to screen readers (status changes, final answer).
function announce(text) {
  srLive.textContent = text;
}

// --- Backend info in the header -----------------------------------------
fetch("/api/info")
  .then((r) => r.json())
  .then((info) => {
    const badge = document.getElementById("backend-badge");
    badge.textContent =
      info.model && info.backend ? `${info.model} · ${info.backend}` : "offline";
    const count = document.getElementById("doc-count");
    if (count) count.textContent = info.documents;
  })
  .catch(() => {
    document.getElementById("backend-badge").textContent = "offline";
  });

// --- Helpers -------------------------------------------------------------
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  chat.scrollTop = chat.scrollHeight;
}

// Minimal, XSS-safe markdown: HTML is escaped first, then only **bold**,
// bullet lists and paragraphs are turned into tags. Applied once at the end so
// the raw text can still stream in token by token during generation.
function renderAnswer(text) {
  const esc = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  let html = "";
  let para = [];
  let inList = false;
  const flushPara = () => {
    if (para.length) {
      html += "<p>" + para.join("<br>") + "</p>";
      para = [];
    }
  };
  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };
  for (const raw of esc.split("\n")) {
    const line = raw.trim();
    if (!line) {
      flushPara();
      closeList();
      continue;
    }
    const bullet = line.match(/^[*-]\s+(.*)$/);
    if (bullet) {
      flushPara();
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += "<li>" + bullet[1] + "</li>";
    } else {
      closeList();
      para.push(line);
    }
  }
  flushPara();
  closeList();
  return html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function addUserMessage(text) {
  const msg = el("div", "msg user");
  msg.appendChild(el("div", "bubble", text));
  chat.appendChild(msg);
  scrollDown();
}

// Builds the bot turn skeleton and returns handles to update as events arrive.
function createBotTurn() {
  const msg = el("div", "msg bot");

  const status = el("div", "status");
  const statusText = el("span", null, "Durchsuche die Wissensbasis …");
  const dots = el("span", "dots");
  dots.append(el("i"), el("i"), el("i"));
  status.append(dots, statusText);

  const bubble = el("div", "bubble");
  bubble.style.display = "none";
  const caret = el("span", "caret");

  const sources = el("details", "sources");
  sources.style.display = "none";

  msg.append(status, bubble, sources);
  chat.appendChild(msg);
  scrollDown();

  return {
    msg, status, statusText, dots, bubble, caret, sources,
    answer: "", answerNode: null, finished: false,
  };
}

function renderSources(turn, list) {
  if (!list || list.length === 0) return;
  const summary = el(
    "summary",
    null,
    `${list.length} Quelle${list.length > 1 ? "n" : ""} aus der Wissensbasis`
  );
  const wrap = el("div", "source-list");

  list.forEach((src) => {
    const card = el("div", "source-card");
    const head = el("div", "source-head");

    const title = el("div", "source-title");
    // Only turn the source into a link if it is a real http(s) URL. Anything
    // else (e.g. a javascript: scheme in a hand-edited data file) becomes
    // plain text, so an untrusted file can never inject a clickable script.
    let safeUrl = null;
    if (src.source) {
      try {
        const parsed = new URL(src.source);
        if (parsed.protocol === "http:" || parsed.protocol === "https:") {
          safeUrl = parsed.href;
        }
      } catch {
        safeUrl = null;
      }
    }
    if (safeUrl) {
      const link = el("a", null, src.title);
      link.href = safeUrl;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      title.appendChild(link);
    } else {
      title.textContent = src.title;
    }

    head.append(title, el("span", "score", `Relevanz ${src.score}`));
    card.append(head, el("p", "source-snippet", src.snippet));
    wrap.appendChild(card);
  });

  turn.sources.innerHTML = "";
  turn.sources.append(summary, wrap);
  turn.sources.style.display = "";
  turn.sources.open = true; // show what was found while the answer streams
  scrollDown();
}

// --- Core: send one question and stream the response --------------------
async function send(message) {
  if (busy || !message) return;
  busy = true;
  sendBtn.disabled = true;
  input.disabled = true;
  if (welcome) welcome.style.display = "none";

  addUserMessage(message);
  const turn = createBotTurn();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (!raw.startsWith("data:")) continue;
        try {
          handleEvent(turn, JSON.parse(raw.slice(5).trim()));
        } catch {
          /* skip a single malformed frame, keep streaming */
        }
      }
    }
  } catch (err) {
    showError(turn, `Verbindungsfehler: ${err.message}`);
  } finally {
    finishTurn(turn);
    busy = false;
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function handleEvent(turn, data) {
  switch (data.type) {
    case "status":
      turn.statusText.textContent = data.text;
      announce(data.text);
      break;
    case "sources":
      renderSources(turn, data.sources);
      break;
    case "token":
      if (turn.bubble.style.display === "none") {
        // First token: reveal the answer, drop the thinking strip.
        turn.bubble.style.display = "";
        turn.status.style.display = "none";
        turn.answerNode = document.createTextNode("");
        turn.bubble.append(turn.answerNode, turn.caret);
      }
      turn.answer += data.text;
      // Append only the new token instead of rewriting the whole bubble, so a
      // long answer is not re-rendered character by character.
      turn.answerNode.data += data.text;
      scrollDown();
      break;
    case "error":
      showError(turn, data.text);
      break;
    case "done":
      finishTurn(turn);
      break;
  }
}

// Shows an error without destroying the status node's structure. If part of the
// answer already streamed, the error is appended as a separate notice so the
// partial text and its incompleteness are both visible.
function showError(turn, message) {
  turn.caret.remove();
  turn.finished = true;
  if (turn.answer) {
    if (!turn.msg.querySelector(".error-notice")) {
      turn.msg.appendChild(el("div", "error-notice", message));
    }
    if (turn.status) turn.status.style.display = "none";
  } else {
    turn.dots.style.display = "none";
    turn.statusText.textContent = message;
    turn.status.style.display = "";
  }
  announce(message);
  scrollDown();
}

// Idempotent: called from the SSE "done" event, from showError and again from
// send()'s finally block. The turn.finished guard makes the later calls no-ops.
function finishTurn(turn) {
  if (turn.finished) return;
  turn.finished = true;
  turn.caret.remove();
  if (turn.answer) {
    if (turn.status) turn.status.style.display = "none";
    // Re-render the streamed plain text as light formatted HTML.
    turn.bubble.innerHTML = renderAnswer(turn.answer);
    announce(turn.answer);
  } else if (turn.status) {
    // Stream ended with no answer: stop the spinner and show a fallback.
    turn.dots.style.display = "none";
    turn.statusText.textContent = "Keine Antwort erhalten.";
    turn.status.style.display = "";
    announce("Keine Antwort erhalten.");
  }
  // Collapse sources into a one-click toggle for a tidy transcript.
  if (turn.sources && turn.sources.style.display !== "none") {
    turn.sources.open = false;
  }
}

// --- Input handling ------------------------------------------------------
function autoresize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

input.addEventListener("input", autoresize);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  autoresize();
  send(message);
});

document.getElementById("suggestions").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) send(btn.textContent.trim());
});

resetBtn.addEventListener("click", async () => {
  if (busy) return;
  await fetch("/api/reset", { method: "POST" }).catch(() => {});
  chat.querySelectorAll(".msg").forEach((m) => m.remove());
  if (welcome) welcome.style.display = "";
  input.focus();
});

input.focus();
