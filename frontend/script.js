const API_BASE_URL = "http://localhost:8000";
const STORAGE_KEYS = {
    userId: "semanticRagUserId",
    theme: "semanticRagTheme",
};
const STORAGE_PREFIX = "semanticRag";
const MAX_MESSAGES = 50;
const DEFAULT_RATE_LIMIT_SECONDS = 8;

const state = {
    userId: "",
    messages: [],
    lastMetrics: null,
    sending: false,
    cooldownEndsAt: 0,
    cooldownTimer: null,
    sidebarOpen: true,
    mobileView: false,
};

const el = {};

window.addEventListener("DOMContentLoaded", init);

function init() {
    cacheElements();
    configureMarkdown();
    initializeTheme();
    initializeUserId();
    initializeResponsiveLayout();
    bindEvents();

    loadStoredChat();
    loadStoredMetrics();

    renderMessages();
    if (state.messages.length === 0) {
        showEmptyState();
    }

    fetchAndRenderFiles();
    updateSendButtonState();
}

function cacheElements() {
    el.app = document.getElementById("app");
    el.toast = document.getElementById("toast");

    el.userIdText = document.getElementById("userIdText");
    el.copyUserIdBtn = document.getElementById("copyUserIdBtn");
    el.resetUserIdBtn = document.getElementById("resetUserIdBtn");
    el.themeToggleBtn = document.getElementById("themeToggleBtn");

    el.openUploadBtn = document.getElementById("openUploadBtn");
    el.resetSessionBtn = document.getElementById("resetSessionBtn");
    el.deleteFilesBtn = document.getElementById("deleteFilesBtn");

    el.sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
    el.closeSidebarBtn = document.getElementById("closeSidebarBtn");
    el.mobileSidebarBtn = document.getElementById("mobileSidebarBtn");

    el.chatMessages = document.getElementById("chatMessages");
    el.typingIndicator = document.getElementById("typingIndicator");
    el.chatForm = document.getElementById("chatForm");
    el.chatInput = document.getElementById("chatInput");
    el.sendBtn = document.getElementById("sendBtn");

    el.uploadModal = document.getElementById("uploadModal");
    el.closeUploadModalBtn = document.getElementById("closeUploadModalBtn");
    el.uploadForm = document.getElementById("uploadForm");
    el.fileInput = document.getElementById("fileInput");
    el.selectedFilesText = document.getElementById("selectedFilesText");
    el.uploadSpinnerWrap = document.getElementById("uploadSpinnerWrap");
    el.uploadSpinnerText = document.getElementById("uploadSpinnerText");
    el.uploadSubmitBtn = document.getElementById("uploadSubmitBtn");
    el.uploadStatus = document.getElementById("uploadStatus");
    el.refreshFilesBtn = document.getElementById("refreshFilesBtn");
    el.modalDeleteFilesBtn = document.getElementById("modalDeleteFilesBtn");
    el.filesList = document.getElementById("filesList");

    el.scoreText = document.getElementById("scoreText");
    el.scoreBar = document.getElementById("scoreBar");
    el.retriesValue = document.getElementById("retriesValue");
    el.confidenceValue = document.getElementById("confidenceValue");
    el.refinedQueryText = document.getElementById("refinedQueryText");
    el.suggestionText = document.getElementById("suggestionText");
    el.driftStatusBadge = document.getElementById("driftStatusBadge");
    el.decisionBadge = document.getElementById("decisionBadge");
    el.driftReasonText = document.getElementById("driftReasonText");
    el.queryDriftValue = document.getElementById("queryDriftValue");
    el.trajectoryDriftValue = document.getElementById("trajectoryDriftValue");
    el.finalDriftValue = document.getElementById("finalDriftValue");
    el.historyLengthValue = document.getElementById("historyLengthValue");
    el.rawDriftJson = document.getElementById("rawDriftJson");
}

function bindEvents() {
    el.copyUserIdBtn.addEventListener("click", copyUserId);
    el.resetUserIdBtn.addEventListener("click", resetUserId);
    el.themeToggleBtn.addEventListener("click", toggleTheme);

    el.openUploadBtn.addEventListener("click", openUploadModal);
    el.closeUploadModalBtn.addEventListener("click", closeUploadModal);
    el.uploadModal.addEventListener("click", (event) => {
        if (event.target && event.target.getAttribute("data-close-modal") === "true") {
            closeUploadModal();
        }
    });

    el.uploadForm.addEventListener("submit", handleUploadSubmit);
    el.fileInput.addEventListener("change", handleFileInputChange);
    el.refreshFilesBtn.addEventListener("click", fetchAndRenderFiles);
    el.modalDeleteFilesBtn.addEventListener("click", deleteAllFiles);

    el.resetSessionBtn.addEventListener("click", resetSession);
    el.deleteFilesBtn.addEventListener("click", deleteAllFiles);

    el.chatForm.addEventListener("submit", handleChatSubmit);
    el.chatInput.addEventListener("input", onChatInputChange);
    el.chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            el.chatForm.requestSubmit();
        }
    });

    el.sidebarToggleBtn.addEventListener("click", () => toggleSidebar());
    el.closeSidebarBtn.addEventListener("click", () => setSidebarOpen(false));
    el.mobileSidebarBtn.addEventListener("click", () => setSidebarOpen(true));

    window.addEventListener("resize", handleResize);
    window.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeUploadModal();
            if (state.mobileView) {
                setSidebarOpen(false);
            }
        }
    });
}

function configureMarkdown() {
    if (!window.marked) {
        return;
    }

    marked.setOptions({
        gfm: true,
        breaks: true,
        mangle: false,
        headerIds: false,
    });

    const renderer = new marked.Renderer();
    renderer.link = (href, title, text) => {
        const safeHref = typeof href === "string" ? href : "#";
        const safeTitle = title ? ` title="${escapeHtml(title)}"` : "";
        return `<a href="${escapeHtml(safeHref)}" target="_blank" rel="noopener noreferrer"${safeTitle}>${text}</a>`;
    };

    marked.use({ renderer });
}

function initializeTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEYS.theme);
    const theme = savedTheme === "light" ? "light" : "dark";
    applyTheme(theme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(STORAGE_KEYS.theme, theme);

    const icon = el.themeToggleBtn.querySelector("i");
    if (icon) {
        icon.className = theme === "dark" ? "fa-regular fa-sun" : "fa-regular fa-moon";
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    applyTheme(currentTheme === "dark" ? "light" : "dark");
}

function initializeUserId() {
    const existingId = localStorage.getItem(STORAGE_KEYS.userId);
    state.userId = existingId || generateUserId();

    if (!existingId) {
        localStorage.setItem(STORAGE_KEYS.userId, state.userId);
    }

    updateUserIdDisplay();
}

function generateUserId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
    }
    const randomPart = Math.random().toString(36).substring(2, 8);
    return `user_${Date.now()}_${randomPart}`;
}

function updateUserIdDisplay() {
    el.userIdText.textContent = state.userId;
}

async function copyUserId() {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(state.userId);
        } else {
            const helper = document.createElement("textarea");
            helper.value = state.userId;
            helper.style.position = "fixed";
            helper.style.opacity = "0";
            document.body.appendChild(helper);
            helper.focus();
            helper.select();
            document.execCommand("copy");
            document.body.removeChild(helper);
        }

        showToast("User ID copied.", "success");
    } catch (_error) {
        showToast("Unable to copy User ID.", "error");
    }
}

async function resetUserId() {
    const confirmed = window.confirm(
        "Reset User ID? This will clear local chat history, reset session memory, and delete uploaded files for the current user."
    );

    if (!confirmed) {
        return;
    }

    const oldUserId = state.userId;

    try {
        await Promise.allSettled([
            apiRequest(`/session/${encodeURIComponent(oldUserId)}`, { method: "DELETE" }),
            apiRequest(`/files/${encodeURIComponent(oldUserId)}`, { method: "DELETE" }),
        ]);
    } catch (_error) {
        // Cleanup requests are intentionally best-effort.
    }

    clearLocalDataForCurrentApp(oldUserId);
    applyTheme("dark");

    const newUserId = generateUserId();
    state.userId = newUserId;
    localStorage.setItem(STORAGE_KEYS.userId, newUserId);
    updateUserIdDisplay();

    state.messages = [];
    state.lastMetrics = null;
    persistChat();
    clearMetricsPanel();
    renderMessages();
    showEmptyState();

    await fetchAndRenderFiles();
    closeUploadModal();

    showToast("New user ID created. Fresh session is ready.", "success");
}

function clearLocalDataForCurrentApp(oldUserId) {
    localStorage.removeItem(chatStorageKey(oldUserId));
    localStorage.removeItem(metricsStorageKey(oldUserId));

    const keysToRemove = [];
    for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (key && key.startsWith(`${STORAGE_PREFIX}Chat:`)) {
            keysToRemove.push(key);
        }
        if (key && key.startsWith(`${STORAGE_PREFIX}Metrics:`)) {
            keysToRemove.push(key);
        }
    }

    keysToRemove.forEach((key) => localStorage.removeItem(key));
    localStorage.removeItem(STORAGE_KEYS.theme);
}

function initializeResponsiveLayout() {
    state.mobileView = window.innerWidth <= 980;
    setSidebarOpen(!state.mobileView);
}

function handleResize() {
    autoResizeTextarea();

    const nowMobile = window.innerWidth <= 980;
    if (nowMobile !== state.mobileView) {
        state.mobileView = nowMobile;
        setSidebarOpen(!nowMobile);
    }
}

function setSidebarOpen(open) {
    state.sidebarOpen = open;
    el.app.classList.toggle("sidebar-open", open);
    el.app.classList.toggle("sidebar-closed", !open);
}

function toggleSidebar() {
    setSidebarOpen(!state.sidebarOpen);
}

function openUploadModal() {
    el.uploadModal.classList.remove("hidden");
    setUploadLoading(false);
    setUploadStatus("", "info");
    fetchAndRenderFiles();
}

function closeUploadModal() {
    el.uploadModal.classList.add("hidden");
}

function handleFileInputChange() {
    const files = Array.from(el.fileInput.files || []);
    if (!files.length) {
        el.selectedFilesText.textContent = "No files selected";
        return;
    }

    if (files.length === 1) {
        el.selectedFilesText.textContent = files[0].name;
    } else {
        el.selectedFilesText.textContent = `${files.length} files selected`;
    }
}

async function handleUploadSubmit(event) {
    event.preventDefault();

    const files = Array.from(el.fileInput.files || []);
    if (!files.length) {
        setUploadStatus("Select at least one file to upload.", "error");
        return;
    }

    setUploadStatus("Uploading files...", "info");
    setUploadLoading(true);

    try {
        const response = await uploadFiles(files);
        const uploadedCount = Array.isArray(response.files) ? response.files.length : files.length;
        setUploadStatus(`Files uploaded (${uploadedCount}). You can start chatting.`, "success");
        showToast("Documents uploaded and indexed.", "success");

        el.fileInput.value = "";
        el.selectedFilesText.textContent = "No files selected";

        await fetchAndRenderFiles();
    } catch (error) {
        handleError(error, "upload");
    } finally {
        setUploadLoading(false);
    }
}

async function uploadFiles(files) {
    const formData = new FormData();
    formData.append("user_id", state.userId);
    files.forEach((file) => formData.append("files", file));

    return apiRequest("/files/upload", {
        method: "POST",
        body: formData,
    });
}

async function fetchAndRenderFiles() {
    try {
        const data = await apiRequest(`/files/${encodeURIComponent(state.userId)}`);
        const files = Array.isArray(data.files) ? data.files : [];
        renderFilesList(files);
    } catch (error) {
        handleError(error, "files");
    }
}

function renderFilesList(files) {
    el.filesList.innerHTML = "";

    if (!files.length) {
        const empty = document.createElement("li");
        empty.textContent = "No uploaded files";
        empty.className = "muted";
        el.filesList.appendChild(empty);
        return;
    }

    files.forEach((fileName) => {
        const li = document.createElement("li");
        li.textContent = fileName;
        el.filesList.appendChild(li);
    });
}

async function deleteAllFiles() {
    const confirmed = window.confirm("Delete all uploaded files and FAISS index for this user?");
    if (!confirmed) {
        return;
    }

    try {
        await apiRequest(`/files/${encodeURIComponent(state.userId)}`, { method: "DELETE" });
        await fetchAndRenderFiles();
        setUploadStatus("All files deleted.", "success");
        showToast("All files deleted for current user.", "success");
    } catch (error) {
        handleError(error, "delete-files");
    }
}

async function resetSession() {
    try {
        await apiRequest(`/session/${encodeURIComponent(state.userId)}`, { method: "DELETE" });
        addMessage("system", "Session drift memory has been reset for this user.");
        showToast("Session reset complete.", "success");
    } catch (error) {
        handleError(error, "reset-session");
    }
}

function onChatInputChange() {
    autoResizeTextarea();
    updateSendButtonState();
}

function autoResizeTextarea() {
    el.chatInput.style.height = "auto";
    el.chatInput.style.height = `${Math.min(el.chatInput.scrollHeight, 180)}px`;
}

async function handleChatSubmit(event) {
    event.preventDefault();

    if (state.sending) {
        return;
    }

    const query = el.chatInput.value.trim();
    if (!query) {
        return;
    }

    if (Date.now() < state.cooldownEndsAt) {
        updateSendButtonState();
        showToast("Please wait for the cooldown to finish.", "error");
        return;
    }

    addMessage("user", query);

    el.chatInput.value = "";
    autoResizeTextarea();

    setSendingState(true);

    try {
        const formData = new FormData();
        formData.append("user_id", state.userId);
        formData.append("query", query);

        const payload = await apiRequest("/generate-answer", {
            method: "POST",
            body: formData,
        });

        processAnswerPayload(payload);
    } catch (error) {
        handleError(error, "chat");
        addMessage("system", error.message || "An error occurred while generating the answer.");
    } finally {
        setSendingState(false);
    }
}

function setSendingState(sending) {
    state.sending = sending;
    el.typingIndicator.classList.toggle("hidden", !sending);
    updateSendButtonState();
}

function processAnswerPayload(payload) {
    const answerData = payload ? payload.answer : null;
    const isAnswerObject = answerData && typeof answerData === "object" && !Array.isArray(answerData);

    const assistantText = isAnswerObject
        ? answerData.answer || "No answer returned by backend."
        : typeof answerData === "string"
            ? answerData
            : "No answer returned by backend.";

    const decision = getFirstDefined(payload?.drift?.decision, payload?.decision, "answer");
    const confidence = getFirstDefined(payload?.drift?.confidence, payload?.confidence, null);
    const sourceData = extractAnswerContentAndSources(assistantText, isAnswerObject ? answerData : null);

    addMessage("assistant", sourceData.content, {
        decision,
        confidence,
        sources: sourceData.sources,
    });

    updateMetricsPanel(payload, isAnswerObject ? answerData : null);

    if (decision === "ask_clarification") {
        showToast("Warning: query is near document boundary.", "error");
    }

    if (decision === "refuse") {
        showToast("Out-of-scope detected. Response was refused.", "error");
    }
}

function updateMetricsPanel(payload, answerObject) {
    const drift = payload && typeof payload.drift === "object" ? payload.drift : null;

    const score = answerObject ? normalizeNumber(answerObject.score) : null;
    const retries = answerObject ? answerObject.retries : null;
    const suggestion = answerObject ? answerObject.suggestion : null;
    const refinedQuery = answerObject ? answerObject.refined_query : null;

    const driftStatus = drift ? drift.status : null;
    const driftReason = drift ? drift.reason : null;
    const queryDrift = drift ? drift.query_drift_score : null;
    const trajectoryDrift = drift ? drift.trajectory_drift_score : null;
    const finalDrift = drift ? drift.final_score : null;
    const decision = getFirstDefined(drift ? drift.decision : null, payload ? payload.decision : null, null);
    const confidence = getFirstDefined(drift ? drift.confidence : null, payload ? payload.confidence : null, null);
    const historyLength = drift ? drift.query_history_length : null;

    setScore(score);
    setText(el.retriesValue, formatNumberOrNA(retries));
    setText(el.confidenceValue, formatPercentOrNA(confidence));
    setText(el.refinedQueryText, refinedQuery || "N/A");
    setText(el.suggestionText, suggestion || "N/A");

    setStatusBadge(el.driftStatusBadge, driftStatus, ["ok", "warning", "out_of_scope", "neutral"]);
    setStatusBadge(el.decisionBadge, decision, ["answer", "ask_clarification", "refuse", "neutral"]);

    setText(el.driftReasonText, driftReason || "N/A");
    setText(el.queryDriftValue, formatFloatOrNA(queryDrift));
    setText(el.trajectoryDriftValue, formatFloatOrNA(trajectoryDrift));
    setText(el.finalDriftValue, formatFloatOrNA(finalDrift));
    setText(el.historyLengthValue, formatNumberOrNA(historyLength));
    setText(el.rawDriftJson, drift ? JSON.stringify(drift, null, 2) : "{}");

    state.lastMetrics = {
        score,
        retries,
        confidence,
        refinedQuery,
        suggestion,
        drift,
        decision,
    };
    persistMetrics();
}

function clearMetricsPanel() {
    setScore(null);
    setText(el.retriesValue, "N/A");
    setText(el.confidenceValue, "N/A");
    setText(el.refinedQueryText, "N/A");
    setText(el.suggestionText, "N/A");
    setStatusBadge(el.driftStatusBadge, "N/A", ["ok", "warning", "out_of_scope", "neutral"]);
    setStatusBadge(el.decisionBadge, "N/A", ["answer", "ask_clarification", "refuse", "neutral"]);
    setText(el.driftReasonText, "No drift analysis yet.");
    setText(el.queryDriftValue, "N/A");
    setText(el.trajectoryDriftValue, "N/A");
    setText(el.finalDriftValue, "N/A");
    setText(el.historyLengthValue, "N/A");
    setText(el.rawDriftJson, "{}");
}

function setScore(score) {
    if (typeof score !== "number" || Number.isNaN(score)) {
        el.scoreText.textContent = "N/A";
        el.scoreBar.style.width = "0%";
        return;
    }

    const scoreClamped = Math.max(0, Math.min(1, score));
    const percent = Math.round(scoreClamped * 100);
    el.scoreText.textContent = `${scoreClamped.toFixed(2)} (${percent}%)`;
    el.scoreBar.style.width = `${percent}%`;
}

function setStatusBadge(element, value, knownClasses) {
    element.classList.remove(...knownClasses);

    const safeValue = typeof value === "string" && value.trim() ? value : "neutral";
    const normalized = safeValue === "N/A" ? "neutral" : safeValue;

    element.classList.add(normalized);
    element.textContent = safeValue.toString().replace(/_/g, " ");
}

function setText(target, value) {
    target.textContent = value;
}

function addMessage(role, text, meta = null) {
    state.messages.push({
        role,
        text,
        meta,
        timestamp: Date.now(),
    });

    if (state.messages.length > MAX_MESSAGES) {
        state.messages = state.messages.slice(-MAX_MESSAGES);
    }

    persistChat();
    renderMessages();
}

function renderMessages() {
    el.chatMessages.innerHTML = "";

    if (!state.messages.length) {
        return;
    }

    state.messages.forEach((message) => {
        const row = document.createElement("div");
        row.className = `msg-row ${message.role}`;

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";

        const content = document.createElement("div");
        if (message.role === "assistant") {
            content.innerHTML = toSafeMarkdown(message.text);
        } else {
            content.textContent = message.text;
            content.style.whiteSpace = "pre-wrap";
        }

        bubble.appendChild(content);

        if (message.role === "assistant" && message.meta) {
            const meta = buildMessageMeta(message.meta);
            if (meta) {
                bubble.appendChild(meta);
            }

            const sourcesDetails = buildSourcesDetails(message.meta.sources);
            if (sourcesDetails) {
                bubble.appendChild(sourcesDetails);
            }
        }

        row.appendChild(bubble);
        el.chatMessages.appendChild(row);
    });

    scrollMessagesToBottom();
}

function buildMessageMeta(meta) {
    const hasDecision = typeof meta.decision === "string" && meta.decision.length > 0;
    const hasConfidence = typeof meta.confidence === "number" && !Number.isNaN(meta.confidence);

    if (!hasDecision && !hasConfidence) {
        return null;
    }

    const container = document.createElement("div");
    container.className = "msg-meta";

    if (hasDecision) {
        const decisionBadge = document.createElement("span");
        decisionBadge.className = "badge";

        if (meta.decision === "refuse") {
            decisionBadge.classList.add("danger");
        } else if (meta.decision === "ask_clarification") {
            decisionBadge.classList.add("warn");
        } else {
            decisionBadge.classList.add("ok");
        }

        decisionBadge.textContent = `Decision: ${meta.decision.replace(/_/g, " ")}`;
        container.appendChild(decisionBadge);
    }

    if (hasConfidence) {
        const confidenceBadge = document.createElement("span");
        confidenceBadge.className = "badge ok";
        confidenceBadge.textContent = `Confidence: ${Math.round(meta.confidence * 100)}%`;
        container.appendChild(confidenceBadge);
    }

    return container;
}

function buildSourcesDetails(sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return null;
    }

    const details = document.createElement("details");
    details.className = "sources-details";

    const summary = document.createElement("summary");
    summary.textContent = `Show sources (${sources.length})`;
    details.appendChild(summary);

    const list = document.createElement("ul");
    list.className = "sources-list";

    sources.forEach((source) => {
        const item = document.createElement("li");
        item.textContent = source;
        list.appendChild(item);
    });

    details.appendChild(list);
    return details;
}

function extractAnswerContentAndSources(answerText, answerObject) {
    const rawText = typeof answerText === "string" ? answerText : "";
    const citationPattern = /\((?:Source|Sources):[^)]+\)/gi;
    const inlineCitations = rawText.match(citationPattern) || [];

    const cleanedText = rawText
        .replace(citationPattern, "")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

    const sources = [];
    const addSource = (sourceText) => {
        const normalized = typeof sourceText === "string" ? sourceText.trim() : "";
        if (normalized && !sources.includes(normalized)) {
            sources.push(normalized);
        }
    };

    inlineCitations.forEach((citation) => addSource(citation.replace(/^\(|\)$/g, "")));

    if (answerObject && Array.isArray(answerObject.docs)) {
        answerObject.docs.forEach((doc) => {
            const metadata = doc && typeof doc === "object" ? doc.metadata : null;
            if (!metadata || typeof metadata !== "object") {
                return;
            }

            const sourceName = typeof metadata.source === "string" ? metadata.source : "";
            const hasPage = metadata.page !== undefined && metadata.page !== null && metadata.page !== "";

            if (!sourceName) {
                return;
            }

            addSource(hasPage ? `${sourceName} (page ${metadata.page})` : sourceName);
        });
    }

    return {
        content: cleanedText || rawText,
        sources,
    };
}

function showEmptyState() {
    const row = document.createElement("div");
    row.className = "msg-row assistant";

    const bubble = document.createElement("div");
    bubble.className = "msg-bubble";
    bubble.innerHTML =
        "<p><strong>Ready.</strong> Upload your files, then ask a question about the document scope.</p><p>You will see drift decisions and evaluation metrics in the sidebar after each answer.</p>";

    row.appendChild(bubble);
    el.chatMessages.appendChild(row);
}

function scrollMessagesToBottom() {
    requestAnimationFrame(() => {
        el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
    });
}

function persistChat() {
    localStorage.setItem(chatStorageKey(state.userId), JSON.stringify(state.messages));
}

function loadStoredChat() {
    const raw = localStorage.getItem(chatStorageKey(state.userId));
    if (!raw) {
        state.messages = [];
        return;
    }

    try {
        const parsed = JSON.parse(raw);
        state.messages = Array.isArray(parsed) ? parsed.slice(-MAX_MESSAGES) : [];
    } catch (_error) {
        state.messages = [];
    }
}

function persistMetrics() {
    localStorage.setItem(metricsStorageKey(state.userId), JSON.stringify(state.lastMetrics));
}

function loadStoredMetrics() {
    const raw = localStorage.getItem(metricsStorageKey(state.userId));
    if (!raw) {
        clearMetricsPanel();
        return;
    }

    try {
        const parsed = JSON.parse(raw);
        state.lastMetrics = parsed;

        if (!parsed || typeof parsed !== "object") {
            clearMetricsPanel();
            return;
        }

        const mockPayload = {
            decision: parsed.decision || null,
            confidence: parsed.confidence || null,
            answer: {
                score: parsed.score,
                retries: parsed.retries,
                suggestion: parsed.suggestion,
                refined_query: parsed.refinedQuery,
            },
            drift: parsed.drift || null,
        };

        const hasAnswerObject = parsed.score != null || parsed.retries != null;
        updateMetricsPanel(mockPayload, hasAnswerObject ? mockPayload.answer : null);
    } catch (_error) {
        clearMetricsPanel();
    }
}

function chatStorageKey(userId) {
    return `${STORAGE_PREFIX}Chat:${userId}`;
}

function metricsStorageKey(userId) {
    return `${STORAGE_PREFIX}Metrics:${userId}`;
}

function setUploadLoading(isLoading) {
    el.uploadSpinnerWrap.classList.toggle("hidden", !isLoading);
    el.uploadSubmitBtn.disabled = isLoading;
    el.fileInput.disabled = isLoading;
    el.refreshFilesBtn.disabled = isLoading;
    el.modalDeleteFilesBtn.disabled = isLoading;
}

function setUploadStatus(message, type) {
    el.uploadStatus.textContent = message;
    el.uploadStatus.classList.remove("error", "success");

    if (type === "error") {
        el.uploadStatus.classList.add("error");
    }
    if (type === "success") {
        el.uploadStatus.classList.add("success");
    }
}

function updateSendButtonState() {
    const remainingCooldown = Math.max(0, Math.ceil((state.cooldownEndsAt - Date.now()) / 1000));
    const blocked = remainingCooldown > 0;
    const hasMessage = el.chatInput.value.trim().length > 0;

    el.sendBtn.disabled = state.sending || blocked || !hasMessage;

    const label = el.sendBtn.querySelector("span");
    if (!label) {
        return;
    }

    if (state.sending) {
        label.textContent = "Sending...";
    } else if (blocked) {
        label.textContent = `Wait ${remainingCooldown}s`;
    } else {
        label.textContent = "Send";
    }
}

function startCooldown(seconds) {
    const safeSeconds = Number.isFinite(Number(seconds)) && Number(seconds) > 0 ? Number(seconds) : DEFAULT_RATE_LIMIT_SECONDS;
    state.cooldownEndsAt = Date.now() + safeSeconds * 1000;

    if (state.cooldownTimer) {
        clearInterval(state.cooldownTimer);
    }

    state.cooldownTimer = window.setInterval(() => {
        updateSendButtonState();
        if (Date.now() >= state.cooldownEndsAt) {
            clearInterval(state.cooldownTimer);
            state.cooldownTimer = null;
        }
    }, 500);

    updateSendButtonState();
}

async function apiRequest(path, options = {}) {
    let response;

    try {
        response = await fetch(`${API_BASE_URL}${path}`, options);
    } catch (_error) {
        throw buildError(0, "Unable to connect to backend. Is FastAPI running at http://localhost:8000?");
    }

    const retryAfter = response.headers.get("Retry-After");
    const payload = await parseResponseBody(response);

    if (!response.ok) {
        throw normalizeRequestError(response.status, payload, retryAfter);
    }

    return payload || {};
}

async function parseResponseBody(response) {
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        try {
            return await response.json();
        } catch (_error) {
            return {};
        }
    }

    try {
        const text = await response.text();
        return text ? { detail: text } : {};
    } catch (_error) {
        return {};
    }
}

function normalizeRequestError(status, payload, retryAfterHeader) {
    const detailMessage = extractErrorMessage(payload);
    const fallback = `Request failed with status ${status}.`;
    const message = detailMessage || fallback;

    const error = buildError(status, message);
    const retryAfter = Number(retryAfterHeader);
    if (Number.isFinite(retryAfter) && retryAfter > 0) {
        error.retryAfter = retryAfter;
    }

    error.payload = payload;
    return error;
}

function extractErrorMessage(payload) {
    if (!payload) {
        return "";
    }

    if (typeof payload === "string") {
        return payload;
    }

    if (typeof payload.detail === "string") {
        return payload.detail;
    }

    if (Array.isArray(payload.detail)) {
        const messages = payload.detail
            .map((item) => {
                if (typeof item === "string") {
                    return item;
                }
                if (item && typeof item === "object") {
                    const loc = Array.isArray(item.loc) ? item.loc.join(".") : "field";
                    const msg = item.msg || "Invalid value";
                    return `${loc}: ${msg}`;
                }
                return "";
            })
            .filter(Boolean);

        return messages.join(" | ");
    }

    return "";
}

function handleError(error, context) {
    const status = error && typeof error.status === "number" ? error.status : 0;
    const message = error && error.message ? error.message : "Unexpected error.";

    if (status === 429) {
        const retryAfter = error.retryAfter || DEFAULT_RATE_LIMIT_SECONDS;
        startCooldown(retryAfter);
        showToast(`Rate limit reached. Try again in ${retryAfter}s.`, "error");

        if (context === "upload") {
            setUploadStatus(`Rate limited. Wait ${retryAfter}s and retry upload.`, "error");
        }
        return;
    }

    if (context === "upload") {
        setUploadStatus(message, "error");
    }

    showToast(message, "error");
}

function buildError(status, message) {
    const error = new Error(message);
    error.status = status;
    return error;
}

function showToast(message, type) {
    el.toast.textContent = message;
    el.toast.classList.remove("hidden", "success", "error");
    if (type === "success") {
        el.toast.classList.add("success");
    }
    if (type === "error") {
        el.toast.classList.add("error");
    }

    window.clearTimeout(showToast._timerId);
    showToast._timerId = window.setTimeout(() => {
        el.toast.classList.add("hidden");
    }, 2800);
}

function toSafeMarkdown(content) {
    if (!content) {
        return "";
    }

    if (!window.marked) {
        return escapeHtml(content).replace(/\n/g, "<br>");
    }

    const rawHtml = marked.parse(content);
    if (window.DOMPurify) {
        return window.DOMPurify.sanitize(rawHtml);
    }

    return rawHtml;
}

function escapeHtml(input) {
    return String(input)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function normalizeNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function formatFloatOrNA(value) {
    const number = normalizeNumber(value);
    return number === null ? "N/A" : number.toFixed(3);
}

function formatNumberOrNA(value) {
    const number = normalizeNumber(value);
    return number === null ? "N/A" : String(Math.round(number));
}

function formatPercentOrNA(value) {
    const number = normalizeNumber(value);
    if (number === null) {
        return "N/A";
    }

    const clamped = Math.max(0, Math.min(1, number));
    return `${Math.round(clamped * 100)}%`;
}

function getFirstDefined(...values) {
    for (const value of values) {
        if (value !== undefined && value !== null) {
            return value;
        }
    }
    return null;
}
