const state = {
  socket: null,
  connected: false,
  ended: false,         // true once the session ended — the closing bubbles
                        // keep playing and the socket-close handler must not
                        // tear the UI down underneath them
  sessionId: "",        // graph session id (context manager key) shown in the sidebar
  paused: false,
  profile: "",
  lastStatus: null,
  thinkingTimer: null,
  chunking: false,      // true while a reply is being played back as bubbles
  chunkStreams: 0,      // active bubble streams (guards chunking_start/end)
  displayToken: 0,      // bumped to cancel an in-flight bubble playback
  methodWarned: false,  // console warning already emitted for a bad debounce method
};

// Loaded from GET /api/config (backend/config.yaml -> chunking block).
// Enabled by default so bubbles work even before/without the config
// fetch; the fetched values override these.
let chunkConfig = {
  enabled: true,
  showTypingIndicator: true,
  maxChars: 220,
  minCps: 4,
  maxCps: 12,
  punctuationPauseMs: 160,
  maxTypingMs: 6000,
  bubbleGapMs: 700,
  firstBubbleDelayMs: 700,
  reflectionVanishMs: 900,
};

const elements = {
  profileSelect: document.getElementById("profile-select"),
  connectBtn: document.getElementById("connect-btn"),
  pauseBtn: document.getElementById("pause-btn"),
  connectionState: document.getElementById("connection-state"),
  profileState: document.getElementById("profile-state"),
  sessionId: document.getElementById("session-id"),
  timerState: document.getElementById("timer-state"),
  assistMode: document.getElementById("assist-mode"),
  description: document.getElementById("description"),
  chatTitle: document.getElementById("chat-title"),
  liveBadge: document.getElementById("live-badge"),
  chatLog: document.getElementById("chat-log"),
  thinkingIndicator: document.getElementById("thinking-indicator"),
  composer: document.getElementById("composer"),
  messageInput: document.getElementById("message-input"),
  sendBtn: document.getElementById("send-btn"),
  savePdfBtn: document.getElementById("savepdf-btn"),
  sidebar: document.getElementById("sidebar"),
  navToggle: document.getElementById("nav-toggle"),
  navClose: document.getElementById("nav-close"),
  scrim: document.getElementById("sidebar-scrim"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function addMessage(kind, title, content, meta = "", extraClass = "") {
  const item = document.createElement("article");
  item.className = `message ${kind} ${extraClass}`.trim();
  item.innerHTML = `
    <div class="message-body">${escapeHtml(content || "")}</div>
    <div class="message-time">${escapeHtml(meta)}</div>
  `;
  elements.chatLog.insertBefore(item, elements.thinkingIndicator);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

// WhatsApp-style session divider: a small grey pill centred in the log that
// marks when the client entered the chat and when the connection dropped.
// It is not a bubble from either speaker, so it carries no speaker styling.
function addSessionNotice(text) {
  const item = document.createElement("div");
  item.className = "session-note";
  item.textContent = text;
  elements.chatLog.insertBefore(item, elements.thinkingIndicator);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

// Clear every chat bubble and cancel any in-flight bubble playback. Used by
// /reset so the UI visibly starts the conversation over, matching what the
// backend did to its history.
function resetChatLog() {
  state.displayToken += 1;
  state.chunkStreams = 0;
  state.chunking = false;
  showThinking(false);
  elements.chatLog
    .querySelectorAll(".message, .session-note")
    .forEach((node) => node.remove());
}

function setConnected(connected) {
  state.connected = connected;
  elements.connectionState.textContent = connected ? "Connected" : "Disconnected";
  elements.liveBadge.textContent = connected ? "connected" : "disconnected";
  elements.liveBadge.className = `badge ${connected ? "live" : "muted"}`;
  elements.messageInput.disabled = !connected;
  elements.sendBtn.disabled = !connected;
  elements.profileSelect.disabled = connected;
  elements.connectBtn.textContent = connected ? "Disconnect" : "Connect";
  elements.pauseBtn.classList.toggle("hidden", !connected);
  elements.savePdfBtn.disabled = !connected;
  if (!connected) {
    elements.pauseBtn.textContent = "Stop";
    state.paused = false;
    elements.savePdfBtn.textContent = "Save PDF";
    // Cancel any reply still being played back as bubbles.
    state.displayToken += 1;
    state.chunkStreams = 0;
    state.chunking = false;
    showThinking(false);
  }
}

function showThinking(visible, immediate = false) {
  if (visible) {
    if (immediate) {
      // Show right away (used while a long reply is being typed out).
      if (state.thinkingTimer) {
        clearTimeout(state.thinkingTimer);
        state.thinkingTimer = null;
      }
      elements.thinkingIndicator.classList.remove("hidden");
    } else if (!state.thinkingTimer && elements.thinkingIndicator.classList.contains("hidden")) {
      const delay = 1500 + Math.random() * 3500; // 1.5s – 5.0s
      state.thinkingTimer = setTimeout(() => {
        state.thinkingTimer = null;
        elements.thinkingIndicator.classList.remove("hidden");
      }, delay);
    }
  } else {
    if (state.thinkingTimer) {
      clearTimeout(state.thinkingTimer);
      state.thinkingTimer = null;
    }
    elements.thinkingIndicator.classList.add("hidden");
  }
}

// ── Bubbles — the reply is played back one message at a time ──────
// The model marks message boundaries with its own line breaks, so a
// reply containing \n / \n\n shows up as several bubbles, one per line,
// in order. Each bubble is preceded by an animated "…" typing bubble
// whose duration mimics how long that client would take to type it.

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function hashString(value) {
  // Stable hash (FNV-1a) — gives each profile its own typing speed, so
  // the same client always "types" at the same pace.
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function splitLongLine(line, maxChars) {
  // Fallback for a single line longer than the cap: break at sentence
  // ends, then clauses, then word boundaries — never mid-word.
  const pieces = [];
  for (const sentence of line.split(/(?<=[.!?…])\s+/)) {
    const s = sentence.trim();
    if (!s) continue;
    if (s.length <= maxChars) {
      pieces.push(s);
      continue;
    }
    for (const clause of s.split(/(?<=[,;:])\s+/)) {
      const c = clause.trim();
      if (!c) continue;
      let rest = c;
      while (rest.length > maxChars) {
        let cut = rest.lastIndexOf(" ", maxChars);
        if (cut <= 0) cut = maxChars;
        pieces.push(rest.slice(0, cut).trim());
        rest = rest.slice(cut).trim();
      }
      if (rest) pieces.push(rest);
    }
  }
  return pieces.length ? pieces : [line];
}

function splitIntoBubbles(text, maxChars) {
  // One bubble per non-empty line, in order. Blank lines and surrounding
  // whitespace are dropped. A reply with no line breaks stays a single
  // bubble.
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const bubbles = [];
  for (const line of lines) {
    if (line.length <= maxChars) {
      bubbles.push(line);
    } else {
      bubbles.push(...splitLongLine(line, maxChars));
    }
  }
  return bubbles;
}

// This client's typing speed (chars/sec); stable per profile, jittered
// slightly per bubble so it doesn't read as a metronome.
function clientTypingSpeed() {
  const spread = Math.max(0, chunkConfig.maxCps - chunkConfig.minCps);
  const profile = state.profile || "";
  const fraction = profile ? (hashString(profile) % 1000) / 1000 : Math.random();
  return chunkConfig.minCps + fraction * spread;
}

// How long the client "takes" to type one bubble: its characters at this
// client's speed, plus a small beat after punctuation, capped so a long
// line can't stall the UI.
function typingDurationMs(text, baseCps) {
  const cps = Math.max(1, baseCps * (0.85 + Math.random() * 0.3));
  const typed = (text.length / cps) * 1000;
  const beats = (text.match(/[.,;:!?…—]/g) || []).length * chunkConfig.punctuationPauseMs;
  return Math.min(chunkConfig.maxTypingMs, Math.max(320, typed + beats));
}

// Pause before the next bubble starts being typed. A short interjection
// ("Yeah.") after a long thought lands almost immediately.
function nextBubbleGapMs(current, previous) {
  let gap = chunkConfig.bubbleGapMs * (0.8 + Math.random() * 0.45);
  if (current.length <= 24 && previous && previous.length > 60) gap *= 0.35;
  return gap;
}

function setTypingBubble(visible) {
  showThinking(chunkConfig.showTypingIndicator && visible, true);
}

async function displayAssistantBubbles(content, timestamp, typing) {
  const text = String(content || "").trim();
  const draftLengths = typing && Array.isArray(typing.lengths) ? typing.lengths : [];
  if (!text && draftLengths.length === 0) return;

  if (!chunkConfig.enabled) {
    if (text) addMessage("assistant", "", text, timestamp);
    return;
  }

  // Claim the playback stream: a newer reply cancels this one, and the
  // idle timer is paused for the whole playback (see chunking_start), so
  // the wait timer doesn't pressure the user while the client is typing.
  const token = (state.displayToken += 1);
  state.chunkStreams = (state.chunkStreams || 0) + 1;
  state.chunking = true;
  if (state.chunkStreams === 1) sendWs("chunking_start");

  const baseCps = clientTypingSpeed();

  try {
    // TYPING_REFLECTION: each draft the client typed and then deleted shows as
    // a "…" that appears, vanishes, and lands nothing. The draft TEXT never
    // reaches the browser — only its length, used to time the dots.
    for (let index = 0; index < draftLengths.length; index += 1) {
      const pause = index === 0
        ? chunkConfig.firstBubbleDelayMs + Math.random() * 300
        : chunkConfig.bubbleGapMs * (0.8 + Math.random() * 0.45);
      await sleep(pause);
      if (token !== state.displayToken) return;

      setTypingBubble(true);
      const length = Math.max(1, Number(draftLengths[index]) || 12);
      await sleep(typingDurationMs("x".repeat(length), baseCps));
      if (token !== state.displayToken) return;

      // Deleted: the dots go away and nothing is sent. A beat of silence makes
      // the "…"→nothing→"…" rhythm legible before the next attempt.
      setTypingBubble(false);
      await sleep(chunkConfig.reflectionVanishMs * (0.7 + Math.random() * 0.6));
      if (token !== state.displayToken) return;
    }

    const bubbles = splitIntoBubbles(text, chunkConfig.maxChars);

    for (let index = 0; index < bubbles.length; index += 1) {
      const bubble = bubbles[index];

      // Composing pause before this bubble's typing begins (the first draft
      // cycle already provided the opening beat when there were drafts).
      const pause = index === 0
        ? (draftLengths.length
          ? 200 + Math.random() * 200
          : chunkConfig.firstBubbleDelayMs + Math.random() * 300)
        : nextBubbleGapMs(bubble, bubbles[index - 1]);
      await sleep(pause);
      if (token !== state.displayToken) return;

      // "…" bubble while the client types this line…
      setTypingBubble(true);
      await sleep(typingDurationMs(bubble, baseCps));
      if (token !== state.displayToken) return;

      // …then the line lands and the dots disappear.
      setTypingBubble(false);
      addMessage(
        "assistant",
        "",
        bubble,
        index === bubbles.length - 1 ? timestamp : "",
        index > 0 ? "chunk" : ""
      );
    }
  } finally {
    setTypingBubble(false);
    state.chunkStreams -= 1;
    if (state.chunkStreams <= 0) {
      state.chunkStreams = 0;
      state.chunking = false;
      sendWs("chunking_end");
    }
  }
}

// A BEHAVIOUR cue is an internal action aside (e.g. *goes offline*,
// *starts typing then stops*), never a chat message — so it is not rendered
// in the trainee's log at all. Only chat bubbles and the start/end session
// dividers belong there.

// ── Graph session id — sidebar row, click to copy ───────────────────
// The id names this chat's Session node in the neo4j-db graph (the key the
// context manager retrieves the session's turns by), so the supervisor can
// copy it straight from the sidebar and look the session up.
let sessionIdFlashTimer = null;

function updateSessionId(rawId) {
  const id = String(rawId || "");
  if (id === state.sessionId) return;
  state.sessionId = id;
  if (sessionIdFlashTimer) {
    clearTimeout(sessionIdFlashTimer);
    sessionIdFlashTimer = null;
  }
  elements.sessionId.classList.remove("copied");
  elements.sessionId.textContent = id || "—";
  elements.sessionId.disabled = !id;
}

async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall through to the legacy path (non-secure contexts, older browsers).
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  area.remove();
  return copied;
}

function updateStatus(payload) {
  state.lastStatus = payload;
  if (!payload) return;
  elements.profileState.textContent = payload.profile_title || payload.profile || "-";
  updateSessionId(payload.memory_session_id);
  if (payload.debounce_method_known === false) {
    // config.yaml names a debounce method that isn't registered, so the
    // backend degraded to its default. Say it out loud — otherwise the
    // pacing silently disagrees with what the config appears to ask for.
    elements.timerState.textContent =
      `⚠ unknown method "${payload.debounce_method_requested}" → using ${payload.debounce_method}`;
    if (!state.methodWarned) {
      state.methodWarned = true;
      console.warn(
        `[debounce] debounce.method="${payload.debounce_method_requested}" is not a registered ` +
        `method; the backend is using "${payload.debounce_method}". Check backend/config.yaml.`
      );
    }
  } else if (payload.debounce_timeout) {
    // The total max_wait_seconds cap released this batch (config.yaml →
    // debounce.max_wait_seconds): however often the window was refreshed, the
    // reply was forced out now. Keep the live phase but annotate it so it is
    // unmistakable that the TIMEOUT — not the normal window — released the turn.
    const phase = payload.api_in_flight ? "thinking…"
      : state.chunking ? "typing…"
      : "";
    elements.timerState.textContent = phase
      ? `${phase} (DEBOUNCE TIMEOUT)`
      : "DEBOUNCE TIMEOUT";
  } else if (state.chunking) {
    // Client is still typing out its reply — don't run the wait timer.
    elements.timerState.textContent = "typing…";
  } else if (payload.debounce_pending) {
    // Waiting for the user to finish typing — show the debounce countdown
    // plus which method produced it and the emotion read it used.
    const n = Number(payload.debounce_pending_count || 0);
    // "cutting in" = this turn was rolled as interrupting, so typing is not
    // holding the turn open any more (see interrupt_typing_probability).
    const label = payload.debounce_interrupting
      ? "cutting in"
      : (payload.debounce_typing ? "typing" : "debounce");
    const meta = [payload.debounce_method, payload.debounce_emotion]
      .filter((part) => part && part !== "neutral")
      .join("/");
    // Show the remaining time until the hard cap forces the turn (if set).
    const cap = Number(payload.debounce_max_wait_seconds || 0);
    const capText = cap > 0
      ? ` • cap ${Number(payload.debounce_cap_remaining || 0).toFixed(1)}s`
      : "";
    elements.timerState.textContent =
      `${label} ${Number(payload.debounce_remaining || 0).toFixed(1)}s • ${n} msg${n === 1 ? "" : "s"}` +
      (meta ? ` • ${meta}` : "") +
      capText;
  } else {
    elements.timerState.textContent = `${Number(payload.last_input_age || 0).toFixed(1)}s / ${Number(payload.time_threshold || 0).toFixed(1)}s`;
  }
  // Pause button state
  if (payload.paused) {
    state.paused = true;
    elements.pauseBtn.textContent = "Continue";
  } else {
    state.paused = false;
    elements.pauseBtn.textContent = "Stop";
  }

  if (payload.api_in_flight) {
    elements.assistMode.textContent = "thinking";
    showThinking(true);
  } else if (state.chunking) {
    // A reply is being played back — the bubble loop owns the "…"
    // indicator (it hides it while the client is not typing), so leave
    // it alone here.
    elements.assistMode.textContent = "typing";
  } else if (payload.debounce_timeout) {
    // The turn was released by the debounce cap — say so here too, so the
    // note survives even after the timer row returns to its idle form.
    elements.assistMode.textContent = "debounce timeout";
    showThinking(false);
  } else if (payload.debounce_pending) {
    // Still collecting the user's burst of messages.
    elements.assistMode.textContent = "collecting";
    showThinking(false);
  } else if (payload.debounce_typing) {
    // The active method is holding the turn open because the user is
    // composing (typing_aware) — let them see that it registered.
    elements.assistMode.textContent = "composing";
    showThinking(false);
  } else if (payload.ended_by_leave) {
    elements.assistMode.textContent = "ended";
    showThinking(false);
  } else if (state.connected) {
    elements.assistMode.textContent = state.paused ? "paused" : "waiting";
    showThinking(false);
  } else {
    elements.assistMode.textContent = "idle";
    showThinking(false);
  }
  if (payload.description && !state.connected) {
    elements.description.textContent = payload.description;
  }
}

function updateProfiles(profiles) {
  const current = elements.profileSelect.value;
  elements.profileSelect.innerHTML = "";
  for (const profile of profiles) {
    const option = document.createElement("option");
    option.value = profile;
    option.textContent = profile;
    elements.profileSelect.appendChild(option);
  }
  if (current && profiles.includes(current)) {
    elements.profileSelect.value = current;
  } else if (profiles.length > 0) {
    elements.profileSelect.value = profiles[0];
  }
}

// ── Phone drawer — the sidebar collapses behind the hamburger ───────
// All of this is a no-op on desktop: the CSS only turns the sidebar into an
// off-canvas drawer under the phone breakpoint, and the controls that drive it
// (hamburger, close button, backdrop) are hidden outside that breakpoint.
const DRAWER_BREAKPOINT = 980;

function drawerAvailable() {
  return window.innerWidth <= DRAWER_BREAKPOINT;
}

function setNavOpen(open) {
  const wantOpen = Boolean(open) && drawerAvailable();
  elements.sidebar.classList.toggle("open", wantOpen);
  elements.scrim.classList.toggle("visible", wantOpen);
  elements.navToggle.setAttribute("aria-expanded", wantOpen ? "true" : "false");
}

function closeNav() {
  setNavOpen(false);
}

function toggleNav() {
  setNavOpen(!elements.sidebar.classList.contains("open"));
}

function socketUrl() {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws`;
}

function connect() {
  // Connecting or disconnecting is the moment the trainee wants the whole
  // screen for the conversation, so collapse the settings drawer. No-op on
  // desktop.
  closeNav();

  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.close();
    setConnected(false);
    elements.description.textContent = "Disconnected. Select a profile and press Connect. (Refresh the page to clear the chat log)";
    state.socket = null;
    return;
  }

  const socket = new WebSocket(socketUrl());
  state.socket = socket;
  state.ended = false;
  setConnected(false);

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "start", profile: elements.profileSelect.value }));
  });

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);

    if (payload.type === "ready") {
      updateProfiles(payload.profiles || []);
      return;
    }

    if (payload.type === "session_started") {
      setConnected(true);
      state.ended = false;
      state.profile = payload.profile;
      const title = payload.profile_title || payload.profile;
      elements.chatTitle.textContent = title;
      elements.description.textContent = payload.description || ("Connected to " + title);
      // Divider at the top of the conversation marking when the client
      // entered the chat (mirrors the one shown when they disconnect).
      addSessionNotice("Client entered the chat");
      updateStatus(payload.status);
      return;
    }

    if (payload.type === "assistant_message") {
      // `content` is only ever what the trainee is allowed to see. A
      // BEHAVIOUR turn has no message: its `behaviour` cue and its private
      // `action.hidden` are backend-side state and are NOT rendered — an
      // empty content means render nothing, exactly like SILENCE.
      const content = payload.content;
      const actionType = payload.action_type || "silence";
      const timestamp = new Date().toLocaleTimeString();
      let playback = null;
      if (actionType === "typing_reflection") {
        // Replay the client's hesitation (dots appearing/vanishing) and then
        // whatever they finally sent — which may be nothing at all.
        playback = displayAssistantBubbles(content, timestamp, payload.typing);
      } else if (content && actionType !== "silence" && actionType !== "behaviour") {
        // BEHAVIOUR is excluded on purpose: its cue ("*starts typing then
        // stops*", "*goes offline*") is internal, and printing it as a
        // centred aside made it read like a stray system message.
        playback = displayAssistantBubbles(content, timestamp);
      }
      if (payload.status) {
        updateStatus(payload.status);
      }
      if (payload.session_end) {
        // A Leave ends the session: the composer locks immediately, while the
        // client's closing bubbles finish playing before the "disconnected"
        // divider lands.
        handleSessionEnd(playback, payload.end_reason || "");
      }
      return;
    }

    if (payload.type === "session_end") {
      handleSessionEnd();
      return;
    }

    if (payload.type === "system_message") {
      // Results of client-side commands (/reset, /help, /context, …) arrive
      // here. The content used to be dropped, which made those commands
      // look like they did nothing — surface it in the chat log instead.
      if (payload.reset) resetChatLog();
      if (payload.content) {
        addMessage("system", "", payload.content, new Date().toLocaleTimeString());
      }
      if (payload.status) {
        updateStatus(payload.status);
      }
      return;
    }

    if (payload.type === "status") {
      updateStatus(payload);
      return;
    }

    if (payload.type === "pdf_ready") {
      downloadFile(payload.url, payload.file);
      resetSavePdfButton();
      const turns = Number(payload.turns || 0);
      elements.description.textContent =
        `Report saved — ${turns} turn${turns === 1 ? "" : "s"} · PDF downloaded ` +
        `(${payload.file}) · JSON at ${payload.json_file}`;
      return;
    }

    if (payload.type === "error") {
      // Surface report failures (and any other error) instead of swallowing
      // them — otherwise a Save PDF click would look like nothing happened.
      if (elements.savePdfBtn.disabled) resetSavePdfButton();
      if (payload.content) elements.description.textContent = payload.content;
      return;
    }
  });

  socket.addEventListener("close", () => {
    // A Leave closes the socket too, but the UI teardown belongs to
    // handleSessionEnd — it still has to let the closing bubbles play and
    // add the "Client disconnected" divider.
    if (state.ended) return;
    setConnected(false);
    elements.description.textContent = "Disconnected. Select a profile and press Connect. (Refresh the page to clear the chat log)";
  });

  socket.addEventListener("error", () => {
    elements.description.textContent = "Connection error.";
  });
}

function sendWs(type, extra = {}) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({ type, ...extra }));
}

// ── Typing signal ───────────────────────────────────────────────────
// The backend's `typing_aware` debounce method refreshes its wait window
// every time the user types NEW content, which is how it tells "still
// composing" from "walked away". Only activity is reported: a stop-typing
// event fires once when the box empties or the message is sent, and the
// backend never refreshes its window on it.
const TYPING_THROTTLE_MS = 200;
const typingSignal = { sentAt: 0, active: false, trailing: null };

function signalTyping(active) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

  if (!active) {
    if (typingSignal.trailing !== null) {
      clearTimeout(typingSignal.trailing);
      typingSignal.trailing = null;
    }
    if (typingSignal.active) {
      typingSignal.active = false;
      typingSignal.sentAt = Date.now();
      sendWs("typing", { active: false });
    }
    return;
  }

  const flush = () => {
    typingSignal.trailing = null;
    typingSignal.active = true;
    typingSignal.sentAt = Date.now();
    sendWs("typing", { active: true });
  };

  const elapsed = Date.now() - typingSignal.sentAt;
  if (elapsed >= TYPING_THROTTLE_MS) {
    flush();
  } else if (typingSignal.trailing === null) {
    // Trailing edge: the last keystroke of a fast burst still refreshes
    // the backend window instead of being throttled away.
    typingSignal.trailing = setTimeout(flush, TYPING_THROTTLE_MS - elapsed);
  }
}

function handleSessionEnd(playback = null, endReason = "") {
  // Session is over — stop accepting input at once (the backend already
  // refuses it), but keep the chat log intact so a closing line that is still
  // being typed out can finish before the UI resets.
  state.ended = true;
  elements.messageInput.disabled = true;
  elements.sendBtn.disabled = true;

  const finish = () => {
    if (endReason === "leave") {
      // Never name the model's verdict: from the trainee's side this is just
      // the client dropping off the chat.
      addSessionNotice("Client disconnected");
    }
    setConnected(false);
    elements.description.textContent = "Session ended.";
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.close();
    }
  };

  if (playback && typeof playback.then === "function") {
    playback.then(finish, finish);
  } else {
    finish();
  }
}

function sendMessage(text) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({ type: "user_message", content: text }));
  addMessage("user", "", text, new Date().toLocaleTimeString());
}

// ── Save PDF — session report download ──────────────────────────────
// The backend builds the report (full cognitive state, should_leave and
// action per turn), writes the JSON under logs/json/ and the PDF under
// logs/pdf/, then replies with the URL to fetch. The download attribute
// makes the browser save it rather than navigate to it.

function downloadFile(url, filename) {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename || "";
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function resetSavePdfButton() {
  elements.savePdfBtn.textContent = "Save PDF";
  elements.savePdfBtn.disabled = !state.connected;
}

function togglePause() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  if (state.paused) {
    state.socket.send(JSON.stringify({ type: "resume" }));
  } else {
    state.socket.send(JSON.stringify({ type: "pause" }));
  }
}

elements.connectBtn.addEventListener("click", connect);
elements.pauseBtn.addEventListener("click", togglePause);

elements.sessionId.addEventListener("click", async () => {
  if (!state.sessionId) return;
  const copied = await copyText(state.sessionId);
  if (!copied) return;
  // Confirm in place — the full id stays in state.sessionId, so the row
  // restores itself (even if the sidebar is crowded) once the flash ends.
  elements.sessionId.textContent = "Copied!";
  elements.sessionId.classList.add("copied");
  if (sessionIdFlashTimer) clearTimeout(sessionIdFlashTimer);
  sessionIdFlashTimer = setTimeout(() => {
    sessionIdFlashTimer = null;
    elements.sessionId.textContent = state.sessionId || "—";
    elements.sessionId.classList.remove("copied");
  }, 1200);
});

// Phone drawer controls.
elements.navToggle.addEventListener("click", toggleNav);
elements.navClose.addEventListener("click", closeNav);
elements.scrim.addEventListener("click", closeNav);
window.addEventListener("resize", () => {
  // Rotating or widening past the phone breakpoint with the drawer open must
  // not leave the backdrop and aria state behind.
  if (!drawerAvailable()) setNavOpen(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeNav();
});

elements.savePdfBtn.addEventListener("click", () => {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  elements.savePdfBtn.disabled = true;
  elements.savePdfBtn.textContent = "Saving…";
  sendWs("save_pdf");
});

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.messageInput.value.trim();
  if (!text) return;
  signalTyping(false);
  sendMessage(text);
  elements.messageInput.value = "";
});

elements.messageInput.addEventListener("input", () => {
  // Report activity only while there is content to send; clearing the box
  // sends the stop-typing event, which the backend never treats as a
  // reason to extend the debounce window.
  signalTyping(elements.messageInput.value.trim().length > 0);
});

elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

// ── Phone keyboard: keep the app bar pinned and the composer visible ──
// iOS overlays the keyboard without resizing the layout viewport, so the
// composer would sit behind it — and Safari then scrolls the page to reveal
// the focused input, which drags the whole app bar off the top of the screen.
// Publishing the covered height as `--kb-inset` lets the CSS lift just the
// composer, so there is nothing left for the browser to scroll into view.
// Android (`interactive-widget=resizes-content`) resizes the viewport
// instead, where the difference is already ~0 — so the two mechanisms cannot
// stack up and double the offset.
function syncKeyboardInset() {
  const vv = window.visualViewport;
  if (!vv) return;
  const covered = window.innerHeight - vv.height - vv.offsetTop;
  // 1px of tolerance: browsers report fractional sizes while the keyboard
  // animates, which would otherwise flicker the inset on and off.
  document.documentElement.style.setProperty(
    "--kb-inset",
    covered > 1 ? `${Math.round(covered)}px` : "0px",
  );
}

if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", syncKeyboardInset);
  window.visualViewport.addEventListener("scroll", syncKeyboardInset);
  syncKeyboardInset();
}
window.addEventListener("orientationchange", syncKeyboardInset);

// Like any chat app: opening the keyboard shows the newest message rather than
// leaving the view wherever it happened to be scrolled to.
elements.messageInput.addEventListener("focus", () => {
  // Delayed because the keyboard is still animating in when focus fires.
  setTimeout(() => {
    elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  }, 300);
});

fetch("/api/profiles")
  .then((response) => response.json())
  .then((payload) => { updateProfiles(payload.profiles || []); })
  .catch(() => {});

// Load the chunking settings from backend/config.yaml via /api/config.
fetch("/api/config")
  .then((response) => response.json())
  .then((cfg) => {
    if (cfg && cfg.chunking) {
      const c = cfg.chunking;
      chunkConfig = {
        enabled: c.enabled !== false,
        showTypingIndicator: c.show_typing_indicator !== false,
        maxChars: Math.max(40, Number(c.max_chars) || 220),
        minCps: Number(c.min_typing_speed_cps) || 4,
        maxCps: Number(c.max_typing_speed_cps) || 12,
        punctuationPauseMs: Number(c.punctuation_pause_ms) || 160,
        maxTypingMs: Math.max(1000, Number(c.max_typing_ms) || 6000),
        bubbleGapMs: Number(c.bubble_gap_ms) || 700,
        firstBubbleDelayMs: Number(c.first_bubble_delay_ms) || 700,
        reflectionVanishMs: Number(c.reflection_vanish_ms) || 900,
      };
    }
  })
  .catch(() => {});