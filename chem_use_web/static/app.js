const devMode = new URLSearchParams(window.location.search).get("dev") === "1";

const els = {
  modelLine: document.getElementById("modelLine"),
  modelStatus: document.getElementById("modelStatus"),
  hardwareStatus: document.getElementById("hardwareStatus"),
  hardwareWarning: document.getElementById("hardwareWarning"),
  oecdStatus: document.getElementById("oecdStatus"),
  textTab: document.getElementById("textTab"),
  fileTab: document.getElementById("fileTab"),
  ragTab: document.getElementById("ragTab"),
  trainTab: document.getElementById("trainTab"),
  textForm: document.getElementById("textForm"),
  fileForm: document.getElementById("fileForm"),
  ragForm: document.getElementById("ragForm"),
  trainingForm: document.getElementById("trainingForm"),
  inputText: document.getElementById("inputText"),
  wordCount: document.getElementById("wordCount"),
  fileInput: document.getElementById("fileInput"),
  columnSelect: document.getElementById("columnSelect"),
  ragQuery: document.getElementById("ragQuery"),
  ragTopK: document.getElementById("ragTopK"),
  ragSourceCsv: document.getElementById("ragSourceCsv"),
  ragOutputColumn: document.getElementById("ragOutputColumn"),
  ragTextColumn: document.getElementById("ragTextColumn"),
  ragPathColumn: document.getElementById("ragPathColumn"),
  ragStatusLine: document.getElementById("ragStatusLine"),
  buildRagIndex: document.getElementById("buildRagIndex"),
  askRagAnswer: document.getElementById("askRagAnswer"),
  enrichRagResults: document.getElementById("enrichRagResults"),
  enrichChemicals: document.getElementById("enrichChemicals"),
  enableOecd: document.getElementById("enableOecd"),
  downloadModel: document.getElementById("downloadModel"),
  modelDownloadProgress: document.getElementById("modelDownloadProgress"),
  modelDownloadText: document.getElementById("modelDownloadText"),
  modelDownloadBar: document.getElementById("modelDownloadBar"),
  downloadOecd: document.getElementById("downloadOecd"),
  oecdFeedback: document.getElementById("oecdFeedback"),
  includeDebug: document.getElementById("includeDebug"),
  developerPanel: document.getElementById("developerPanel"),
  debugOutput: document.getElementById("debugOutput"),
  summaryLine: document.getElementById("summaryLine"),
  downloadLinks: document.getElementById("downloadLinks"),
  message: document.getElementById("message"),
  ragAnswerBox: document.getElementById("ragAnswerBox"),
  trainingPlanBox: document.getElementById("trainingPlanBox"),
  resultsBody: document.getElementById("resultsBody"),
};

let latestResult = null;
let modelDownloadPoller = null;

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
}

function showMessage(text, isError = false) {
  if (!text) {
    els.message.classList.add("hidden");
    els.message.textContent = "";
    return;
  }
  els.message.textContent = text;
  els.message.classList.toggle("error", isError);
  els.message.classList.remove("hidden");
}

function hideRagAnswer() {
  els.ragAnswerBox.innerHTML = "";
  els.ragAnswerBox.classList.add("hidden");
}

function hideTrainingPlan() {
  if (!els.trainingPlanBox) return;
  els.trainingPlanBox.innerHTML = "";
  els.trainingPlanBox.classList.add("hidden");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function switchTab(mode) {
  const textMode = mode === "text";
  const fileMode = mode === "file";
  const ragMode = mode === "rag";
  const trainMode = mode === "train";
  els.textTab.classList.toggle("active", textMode);
  els.fileTab.classList.toggle("active", fileMode);
  els.ragTab.classList.toggle("active", ragMode);
  els.trainTab.classList.toggle("active", trainMode);
  els.textForm.classList.toggle("active-panel", textMode);
  els.fileForm.classList.toggle("active-panel", fileMode);
  els.ragForm.classList.toggle("active-panel", ragMode);
  els.trainingForm.classList.toggle("active-panel", trainMode);
}

function updateWordCount() {
  const words = (els.inputText.value.trim().match(/\b\S+\b/g) || []).length;
  els.wordCount.textContent = `${words} / 500 words`;
  els.wordCount.style.color = words > 500 ? "var(--danger)" : "var(--muted)";
}

function setPill(el, text, state) {
  el.textContent = text;
  el.className = `pill ${state || ""}`;
}

function setInputFromStatus(input, value) {
  if (!input || document.activeElement === input) return;
  input.value = value || "";
}

function formatGb(value) {
  return typeof value === "number" ? `${value.toFixed(1)} GB` : "memory unknown";
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const scaled = bytes / 1024 ** exponent;
  const digits = exponent >= 3 ? 2 : exponent === 0 ? 0 : 1;
  return `${scaled.toFixed(digits)} ${units[exponent]}`;
}

function renderModelDownloadProgress(status) {
  if (!status || status.status === "idle" || status.status === "unavailable") {
    els.modelDownloadProgress.classList.add("hidden");
    els.modelDownloadBar.classList.remove("indeterminate");
    els.modelDownloadBar.style.width = "0%";
    els.modelDownloadText.textContent = "";
    return;
  }

  const downloaded = formatBytes(status.bytes_downloaded);
  const total = status.total_bytes ? formatBytes(status.total_bytes) : "";
  const percent = typeof status.percent === "number" ? Math.max(0, Math.min(100, status.percent)) : null;
  const elapsed = status.elapsed_seconds ? `, ${Math.round(status.elapsed_seconds)}s` : "";
  let detail = status.message || "Downloading model files.";

  if (status.status === "error") {
    detail = status.error || detail;
  } else if (percent !== null && total) {
    detail = `${detail} ${downloaded} / ${total} (${percent.toFixed(1)}%${elapsed}).`;
  } else {
    detail = `${detail} ${downloaded} downloaded${elapsed}.`;
  }

  els.modelDownloadText.textContent = detail;
  els.modelDownloadProgress.classList.remove("hidden");
  els.modelDownloadBar.classList.toggle("indeterminate", status.active && percent === null);
  els.modelDownloadBar.style.width = percent === null ? "" : `${percent}%`;
}

async function refreshModelDownloadProgress() {
  const status = await requestJson("/api/model/download/status");
  renderModelDownloadProgress(status);
  return status;
}

function startModelDownloadPolling() {
  if (modelDownloadPoller) {
    clearInterval(modelDownloadPoller);
  }
  refreshModelDownloadProgress().catch(() => {});
  modelDownloadPoller = setInterval(() => {
    refreshModelDownloadProgress()
      .then((status) => {
        if (!status.active && status.status !== "downloading") {
          clearInterval(modelDownloadPoller);
          modelDownloadPoller = null;
        }
      })
      .catch(() => {});
  }, 1000);
}

function formatQuantization(value) {
  if (!value || value === "none") {
    return "";
  }
  if (value === "fp8") {
    return " FP8";
  }
  if (value === "mlx-8bit") {
    return " MLX 8-bit";
  }
  if (value === "mlx-4bit") {
    return " MLX 4-bit";
  }
  if (value === "mlx") {
    return " MLX";
  }
  return ` ${String(value).toUpperCase()}`;
}

function renderHardware(health) {
  const hardware = health.hardware || health.model?.hardware || {};
  const accelerator = hardware.accelerator || "unknown";
  const memory = formatGb(hardware.total_memory_gb);
  const quantization = formatQuantization(hardware.selected_quantization);
  const label = `${accelerator.toUpperCase()} ${memory}${quantization}`;
  setPill(els.hardwareStatus, label, hardware.warning ? "warn" : "ok");

  if (hardware.warning) {
    els.hardwareWarning.textContent = hardware.warning;
    els.hardwareWarning.classList.remove("hidden");
  } else {
    els.hardwareWarning.textContent = "";
    els.hardwareWarning.classList.add("hidden");
  }
}

function renderRagStatus(rag) {
  if (!rag) return;
  setInputFromStatus(els.ragSourceCsv, rag.source_csv);
  setInputFromStatus(els.ragOutputColumn, rag.output_column);
  setInputFromStatus(els.ragTextColumn, rag.text_column);
  setInputFromStatus(els.ragPathColumn, rag.path_column);
  const text = rag.available
    ? `Search index ready: ${(rag.pair_count || rag.row_count).toLocaleString()} pairs`
    : rag.needs_reindex
      ? "Search index needs rebuild"
    : rag.source_exists
      ? "Search index not built"
      : "Choose a RAG knowledge file";
  els.ragStatusLine.textContent = text;
  els.ragStatusLine.className = `inline-status ${rag.available ? "ok" : "warn"}`;
  els.ragStatusLine.classList.remove("hidden");
}

function renderOecdStatus(oecd, probe = null) {
  if (probe) {
    const providerLabel =
      probe.provider === "ollama" ? "Ollama" : probe.provider === "vllm" ? "vLLM" : "local";
    if (probe.ready) {
      setPill(els.oecdStatus, `OECD ${providerLabel} ready`, "ok");
    } else {
      setPill(els.oecdStatus, "OECD optional", "");
    }
    els.oecdFeedback.textContent = probe.message || "";
    els.oecdFeedback.className = `inline-status ${probe.ready ? "ok" : "warn"}`;
    els.oecdFeedback.classList.remove("hidden");
    return;
  }
  els.oecdFeedback.textContent = "";
  els.oecdFeedback.className = "inline-status hidden";
  if (oecd.provider === "ollama") {
    setPill(els.oecdStatus, "OECD Ollama", "ok");
    els.downloadOecd.textContent = "Check Ollama OECD port";
    return;
  }
  if (oecd.provider === "vllm") {
    setPill(els.oecdStatus, "OECD vLLM", "ok");
    els.downloadOecd.textContent = "Check vLLM OECD port";
    return;
  }
  setPill(
    els.oecdStatus,
    oecd.model_dir_exists ? "OECD local" : "OECD optional",
    oecd.model_dir_exists ? "ok" : "warn",
  );
  els.downloadOecd.textContent = "Download OECD model";
}

async function loadHealth() {
  const health = await requestJson("/api/health");
  const model = health.model || {};
  const oecd = health.oecd || {};
  const settings = health.settings || {};
  const backend = settings.extraction_backend || model.backend || "transformers";
  els.modelLine.textContent = `${backend.toUpperCase()}: ${settings.model_path || ""}`;
  setPill(
    els.modelStatus,
    model.model_path_exists ? "Llama-CUE ready" : "Llama-CUE not downloaded",
    model.model_path_exists ? "ok" : "warn",
  );
  renderOecdStatus(oecd);
  renderRagStatus(health.rag);
  renderHardware(health);
  if (model.model_repo_id) {
    els.downloadModel.classList.remove("hidden");
    els.downloadModel.textContent = model.model_path_exists
      ? "Download/update Llama-CUE model"
      : "Download Llama-CUE model";
  } else {
    els.downloadModel.classList.add("hidden");
  }
  if (devMode) {
    els.developerPanel.classList.remove("hidden");
    els.debugOutput.textContent = JSON.stringify(health, null, 2);
  }
  if (oecd.provider === "ollama" || oecd.provider === "vllm") {
    checkOecdProvider({ quiet: true }).catch((error) => {
      els.oecdFeedback.textContent = error.message;
      els.oecdFeedback.className = "inline-status warn";
      els.oecdFeedback.classList.remove("hidden");
    });
  }
}

async function downloadModel() {
  showMessage("Starting Llama-CUE model download. Large sharded models can take a long time.");
  renderModelDownloadProgress({
    status: "downloading",
    active: true,
    message: "Preparing Hugging Face download.",
    bytes_downloaded: 0,
    total_bytes: null,
    percent: null,
    elapsed_seconds: 0,
  });
  startModelDownloadPolling();
  setBusy(true);
  try {
    const result = await requestJson("/api/model/download", { method: "POST" });
    renderModelDownloadProgress(result.download);
    showMessage(result.message || "Llama-CUE model download complete.");
    if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
    await loadHealth();
    await refreshModelDownloadProgress();
  } catch (error) {
    showMessage(error.message, true);
    await refreshModelDownloadProgress().catch(() => {});
  } finally {
    setBusy(false);
  }
}

async function previewFile() {
  const file = els.fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const preview = await requestJson("/api/preview-file", { method: "POST", body: form });
  els.columnSelect.innerHTML = '<option value="">Auto detect</option>';
  preview.columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    if (column === preview.inferred_text_column) option.selected = true;
    els.columnSelect.appendChild(option);
  });
  showMessage(`${preview.row_count} rows loaded. Text column: ${preview.inferred_text_column}`);
}

async function runText(event) {
  event.preventDefault();
  hideRagAnswer();
  showMessage("");
  setBusy(true);
  try {
    const payload = {
      text: els.inputText.value,
      enable_oecd: els.enableOecd.checked,
      enrich_chemicals: els.enrichChemicals.checked,
      include_debug: devMode && els.includeDebug.checked,
    };
    const result = await requestJson("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderResult(result);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function runFile(event) {
  event.preventDefault();
  hideRagAnswer();
  const file = els.fileInput.files[0];
  if (!file) {
    showMessage("Upload a file first.", true);
    return;
  }
  showMessage("");
  setBusy(true);
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("column", els.columnSelect.value);
    form.append("enable_oecd", String(els.enableOecd.checked));
    form.append("enrich_chemicals", String(els.enrichChemicals.checked));
    form.append("include_debug", String(devMode && els.includeDebug.checked));
    const result = await requestJson("/api/extract-file", { method: "POST", body: form });
    renderResult(result);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function downloadOecd() {
  showMessage("Checking OECD provider.");
  setBusy(true);
  try {
    const health = await requestJson("/api/health");
    const provider = health.oecd?.provider;
    const result =
      provider === "ollama" || provider === "vllm"
        ? await checkOecdProvider()
        : await requestJson("/api/oecd/download", { method: "POST" });
    showMessage(result.message || "OECD model is available locally.");
    if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
    await loadHealth();
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function checkOecdProvider({ quiet = false } = {}) {
  const result = await requestJson("/api/oecd/check", { method: "POST" });
  renderOecdStatus(result, result);
  if (!quiet) {
    showMessage(result.message || "");
  }
  if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
  return result;
}

async function buildRagIndex() {
  hideRagAnswer();
  const sourceCsv = els.ragSourceCsv.value.trim();
  const outputColumn = els.ragOutputColumn.value.trim();
  const textColumn = els.ragTextColumn.value.trim();
  const pathColumn = els.ragPathColumn.value.trim();
  if (!sourceCsv || !outputColumn || !textColumn) {
    showMessage("Enter a knowledge CSV/XLSX path, prediction column, and evidence column.", true);
    return;
  }
  showMessage("Building the local use-search index. Large knowledge CSV files can take a minute.");
  setBusy(true);
  try {
    const result = await requestJson("/api/rag/reindex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_csv: sourceCsv,
        output_column: outputColumn,
        text_column: textColumn,
        path_column: pathColumn,
      }),
    });
    showMessage(result.message || "Search index built.");
    await loadHealth();
    if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function runRagSearch(event) {
  event.preventDefault();
  hideRagAnswer();
  const query = els.ragQuery.value.trim();
  if (!query) {
    showMessage("Enter a search query.", true);
    return;
  }
  showMessage("");
  setBusy(true);
  try {
    const result = await requestJson("/api/rag/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(els.ragTopK.value || 10),
      }),
    });
    renderRagResults(result);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function askRagAnswer() {
  const query = els.ragQuery.value.trim();
  if (!query) {
    showMessage("Enter a RAG question.", true);
    return;
  }
  showMessage("Retrieving evidence and asking the local RAG model.");
  setBusy(true);
  try {
    const result = await requestJson("/api/rag/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: Number(els.ragTopK.value || 10),
      }),
    });
    renderRagAnswer(result);
    renderRagResults(result.retrieval || result, { keepAnswer: true });
    showMessage(result.answer ? "" : result.message || "No grounded RAG answer was generated.");
    if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function enrichVisibleRagChemicals() {
  const records = latestResult?.results || [];
  if (!records.length) {
    showMessage("Search first, then enrich the visible chemicals.", true);
    return;
  }

  const limit = Math.min(records.length, Number(els.ragTopK.value || 10), 50);
  const visible = records.slice(0, limit);
  showMessage(`Enriching ${visible.length} visible chemicals. First-time API lookups may take a few seconds.`);
  setBusy(true);
  try {
    const payloadRows = visible.map((row) => ({
      rowid: row.rowid,
      chemical: row.chemical || row.chemicals || "",
    }));
    const result = await requestJson("/api/rag/enrich", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        results: payloadRows,
        limit,
      }),
    });
    const byRowId = new Map((result.results || []).map((row) => [String(row.rowid), row]));
    latestResult.results = records.map((row) => {
      const enriched = byRowId.get(String(row.rowid));
      return enriched ? { ...row, chemical_info: enriched.chemical_info } : row;
    });
    renderRagResults(latestResult, { keepAnswer: true });
    showMessage(`Enriched ${result.count || 0} rows for ${result.unique_chemicals || 0} unique chemicals.`);
    if (devMode) els.debugOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

function renderRagResults(result, { keepAnswer = false } = {}) {
  hideTrainingPlan();
  if (!keepAnswer) {
    hideRagAnswer();
  }
  latestResult = result;
  const records = result.results || [];
  els.summaryLine.textContent = `${records.length} matching pairs for "${result.query || ""}"`;
  els.downloadLinks.innerHTML = "";
  showMessage(records.length ? "" : "No matching extracted uses found.");
  els.enrichRagResults.classList.toggle("hidden", !records.length);
  els.resultsBody.innerHTML = "";

  const table = els.resultsBody.closest("table");
  table.classList.add("rag-table");
  table.querySelector("thead").innerHTML = `
    <tr>
      <th>Rank</th>
      <th>Relevance</th>
      <th>Chemical</th>
      <th>Use</th>
      <th>Identifiers</th>
      <th>Links</th>
      <th>Evidence</th>
      <th>Source</th>
    </tr>
  `;

  if (!records.length) {
    els.resultsBody.innerHTML = '<tr><td colspan="8" class="empty">No matching extracted uses found.</td></tr>';
    return;
  }

  records.forEach((record, index) => {
    const tr = document.createElement("tr");
    const info = record.chemical_info || {};
    const identifiers = info.identifiers || {};
    const links = info.links || {};
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${Number(record.relevance || 0).toFixed(3)}</td>
      <td><pre class="cell-pre compact">${escapeHtml(record.chemical || record.chemicals || "")}</pre></td>
      <td><pre class="cell-pre">${escapeHtml(record.use || record.uses || "")}</pre></td>
      <td>${info.query ? renderIdentifiers(identifiers) : '<span class="muted">Not enriched</span>'}</td>
      <td>${info.query ? renderLinks(links) : '<span class="muted">Not enriched</span>'}</td>
      <td>${escapeHtml(record.paragraph_preview || "")}</td>
      <td class="source-cell">${escapeHtml(record.path || "")}</td>
    `;
    els.resultsBody.appendChild(tr);
  });

  if (devMode) {
    els.debugOutput.textContent = JSON.stringify(result, null, 2);
  }
}

function renderRagAnswer(result) {
  hideTrainingPlan();
  const answer = result.answer || result.message || "";
  if (!answer) {
    hideRagAnswer();
    return;
  }

  const provider = [result.provider, result.model].filter(Boolean).join(" ");
  const citations = result.citations || [];
  const citationHtml = citations.length
    ? `<div class="rag-citations">${citations
        .map((citation) => {
          const sourceParts = [];
          if (citation.source) sourceParts.push(citation.source);
          if (citation.source_row_number) sourceParts.push(`row ${citation.source_row_number}`);
          if (citation.pair_index) sourceParts.push(`pair ${citation.pair_index}`);
          return `<div class="rag-citation"><span>[${escapeHtml(citation.id)}]</span><strong>${escapeHtml(
            citation.chemical || "",
          )}</strong><em>${escapeHtml(citation.use || "")}</em><small>${escapeHtml(sourceParts.join(" | "))}</small></div>`;
        })
        .join("")}</div>`
    : "";

  els.ragAnswerBox.innerHTML = `
    <div class="rag-answer-head">
      <h3>RAG answer</h3>
      <span>${escapeHtml(provider || "local LLM")}</span>
    </div>
    <pre class="rag-answer-text">${escapeHtml(answer)}</pre>
    ${citationHtml}
  `;
  els.ragAnswerBox.classList.remove("hidden");
}

function renderResult(result) {
  hideRagAnswer();
  hideTrainingPlan();
  latestResult = result;
  const summary = result.summary || {};
  const statuses = summary.statuses || {};
  const statusText = Object.entries(statuses)
    .map(([key, value]) => `${key}: ${value}`)
    .join(", ");
  els.summaryLine.textContent = `${summary.records || 0} records, ${summary.unique_chemicals || 0} chemicals${
    statusText ? ` (${statusText})` : ""
  }`;

  if (result.warnings?.length) {
    showMessage(result.warnings.slice(0, 3).join(" "));
  } else {
    showMessage("");
  }

  renderDownloads(result.download_urls || {});
  renderRows(result.records || []);
  if (devMode) {
    els.debugOutput.textContent = JSON.stringify(result, null, 2);
  }
}

function renderDownloads(urls) {
  els.downloadLinks.innerHTML = "";
  Object.entries(urls).forEach(([label, url]) => {
    const link = document.createElement("a");
    link.href = url;
    link.textContent = label.toUpperCase();
    els.downloadLinks.appendChild(link);
  });
}

function renderRows(records) {
  els.resultsBody.innerHTML = "";
  const table = els.resultsBody.closest("table");
  table.classList.remove("rag-table");
  table.querySelector("thead").innerHTML = `
    <tr>
      <th>Row</th>
      <th>Status</th>
      <th>Chemical</th>
      <th>Use</th>
      <th>OECD</th>
      <th>Identifiers</th>
      <th>Links</th>
    </tr>
  `;
  if (!records.length) {
    els.resultsBody.innerHTML = '<tr><td colspan="7" class="empty">No chemical-use pairs found.</td></tr>';
    return;
  }

  records.forEach((record) => {
    const tr = document.createElement("tr");
    const info = record.chemical_info || {};
    const identifiers = info.identifiers || {};
    const links = info.links || {};
    tr.innerHTML = `
      <td>${escapeHtml(String((record.source_row ?? 0) + 1))}</td>
      <td><span class="status ${escapeHtml(record.status || "")}">${escapeHtml(record.status || "")}</span></td>
      <td>${escapeHtml(record.chemical || "")}</td>
      <td>${escapeHtml(record.use || record.error || "")}</td>
      <td>${escapeHtml((record.oecd_labels || []).join(" | ") || record.oecd_error || "")}</td>
      <td>${renderIdentifiers(identifiers)}</td>
      <td>${renderLinks(links)}</td>
    `;
    els.resultsBody.appendChild(tr);
  });
}

function renderIdentifiers(identifiers) {
  const parts = [];
  if (identifiers.pubchem_cid) parts.push(`CID ${escapeHtml(identifiers.pubchem_cid)}`);
  if (identifiers.cas?.length) parts.push(`CAS ${escapeHtml(identifiers.cas.slice(0, 3).join(", "))}`);
  if (identifiers.inchi_key) parts.push(`InChIKey ${escapeHtml(identifiers.inchi_key)}`);
  if (identifiers.chembl_id) parts.push(escapeHtml(identifiers.chembl_id));
  if (identifiers.smiles) parts.push(`SMILES ${escapeHtml(identifiers.smiles)}`);
  return parts.length ? parts.join("<br>") : '<span class="muted">No identifiers</span>';
}

function renderLinks(links) {
  const entries = Object.entries(links || {});
  if (!entries.length) return '<span class="muted">No links</span>';
  const html = entries
    .slice(0, 8)
    .map(([name, url]) => `<a href="${escapeAttr(url)}" target="_blank" rel="noreferrer">${escapeHtml(name)}</a>`)
    .join("");
  return `<div class="links">${html}</div>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

els.textTab.addEventListener("click", () => switchTab("text"));
els.fileTab.addEventListener("click", () => switchTab("file"));
els.ragTab.addEventListener("click", () => switchTab("rag"));
els.trainTab.addEventListener("click", () => switchTab("train"));
els.inputText.addEventListener("input", updateWordCount);
els.fileInput.addEventListener("change", () => previewFile().catch((error) => showMessage(error.message, true)));
els.textForm.addEventListener("submit", runText);
els.fileForm.addEventListener("submit", runFile);
els.ragForm.addEventListener("submit", runRagSearch);
els.downloadModel.addEventListener("click", downloadModel);
els.downloadOecd.addEventListener("click", downloadOecd);
els.buildRagIndex.addEventListener("click", buildRagIndex);
els.askRagAnswer.addEventListener("click", askRagAnswer);
els.enrichRagResults.addEventListener("click", enrichVisibleRagChemicals);

updateWordCount();
loadHealth().catch((error) => showMessage(error.message, true));
