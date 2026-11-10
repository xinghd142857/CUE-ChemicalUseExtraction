(() => {
  const trainingEls = {
    form: document.getElementById("trainingForm"),
    machine: document.getElementById("trainingMachine"),
    statusLine: document.getElementById("trainingStatusLine"),
    planBox: document.getElementById("trainingPlanBox"),
    modelLink: document.getElementById("trainingModelLink"),
    modelSize: document.getElementById("trainingModelSize"),
    learningRate: document.getElementById("trainingLearningRate"),
    epochs: document.getElementById("trainingEpochs"),
    seqLength: document.getElementById("trainingSeqLength"),
    batchSize: document.getElementById("trainingBatchSize"),
    gradAccum: document.getElementById("trainingGradAccum"),
    loraRank: document.getElementById("trainingLoraRank"),
    loraAlpha: document.getElementById("trainingLoraAlpha"),
    loraDropout: document.getElementById("trainingLoraDropout"),
    dpoBeta: document.getElementById("trainingDpoBeta"),
    load4Bit: document.getElementById("trainingLoad4Bit"),
    checkpointing: document.getElementById("trainingCheckpointing"),
    dataPath: document.getElementById("trainingDataPath"),
    inputColumn: document.getElementById("trainingInputColumn"),
    outputColumn: document.getElementById("trainingOutputColumn"),
    promptColumn: document.getElementById("trainingPromptColumn"),
    chosenColumn: document.getElementById("trainingChosenColumn"),
    rejectedColumn: document.getElementById("trainingRejectedColumn"),
    promptTemplate: document.getElementById("trainingPromptTemplate"),
    loraColumns: document.querySelector(".lora-columns"),
    dpoColumns: document.querySelector(".dpo-columns"),
    summaryLine: document.getElementById("summaryLine"),
    downloadLinks: document.getElementById("downloadLinks"),
    resultsBody: document.getElementById("resultsBody"),
    enrichRagResults: document.getElementById("enrichRagResults"),
  };

  if (!trainingEls.form) return;

  let modelSizeEdited = false;

  function getTrainingStrategy() {
    return document.querySelector('input[name="trainingStrategy"]:checked')?.value || "lora";
  }

  function updateTrainingStrategyFields() {
    const isDpo = getTrainingStrategy() === "dpo";
    trainingEls.loraColumns.classList.toggle("hidden", isDpo);
    trainingEls.dpoColumns.classList.toggle("hidden", !isDpo);
  }

  function inferModelSizeFromLink(value) {
    const matches = String(value || "").match(/(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/g);
    if (!matches?.length) return "";
    const clean = matches[matches.length - 1].replace(/[bB]/g, "").trim();
    return clean || "";
  }

  function maybeUpdateModelSize() {
    if (modelSizeEdited) return;
    const inferred = inferModelSizeFromLink(trainingEls.modelLink.value);
    if (inferred) trainingEls.modelSize.value = inferred;
  }

  function renderTrainingMachine(health) {
    const hardware = health.hardware || {};
    const accelerator = hardware.accelerator || "unknown";
    const memory = typeof hardware.total_memory_gb === "number" ? `${hardware.total_memory_gb.toFixed(1)} GB` : "unknown memory";
    const name = hardware.name ? `, ${hardware.name}` : "";
    const quantization = hardware.selected_quantization && hardware.selected_quantization !== "none"
      ? `, ${hardware.selected_quantization}`
      : "";
    trainingEls.machine.textContent = `Machine: ${accelerator.toUpperCase()}${name}, ${memory}${quantization}`;
  }

  async function refreshTrainingMachine() {
    try {
      const health = await requestJson("/api/health");
      renderTrainingMachine(health);
    } catch (error) {
      trainingEls.machine.textContent = `Machine information unavailable: ${error.message}`;
    }
  }

  function buildTrainingPayload() {
    return {
      strategy: getTrainingStrategy(),
      model_link: trainingEls.modelLink.value.trim(),
      model_size_b: trainingEls.modelSize.value,
      training_data_path: trainingEls.dataPath.value.trim(),
      input_column: trainingEls.inputColumn.value.trim(),
      output_column: trainingEls.outputColumn.value.trim(),
      prompt_column: trainingEls.promptColumn.value.trim(),
      chosen_column: trainingEls.chosenColumn.value.trim(),
      rejected_column: trainingEls.rejectedColumn.value.trim(),
      prompt_template: trainingEls.promptTemplate.value,
      learning_rate: trainingEls.learningRate.value,
      epochs: trainingEls.epochs.value,
      per_device_batch_size: trainingEls.batchSize.value,
      gradient_accumulation_steps: trainingEls.gradAccum.value,
      max_seq_length: trainingEls.seqLength.value,
      lora_rank: trainingEls.loraRank.value,
      lora_alpha: trainingEls.loraAlpha.value,
      lora_dropout: trainingEls.loraDropout.value,
      dpo_beta: trainingEls.dpoBeta.value,
      load_in_4bit: trainingEls.load4Bit.checked,
      gradient_checkpointing: trainingEls.checkpointing.checked,
    };
  }

  function setTrainingStatus(plan) {
    const estimate = plan.estimate || {};
    const status = estimate.status || "unknown";
    const available = estimate.available_memory_gb == null ? "unknown" : `${estimate.available_memory_gb.toFixed(2)} GB`;
    const margin = estimate.margin_gb == null ? "" : `, margin ${estimate.margin_gb.toFixed(2)} GB`;
    trainingEls.statusLine.textContent = `Estimated ${estimate.estimated_memory_gb?.toFixed(2) || "?"} GB needed; detected ${available}${margin}.`;
    trainingEls.statusLine.className = `inline-status ${status === "ok" ? "ok" : "warn"}`;
    trainingEls.statusLine.classList.remove("hidden");
  }

  function renderTrainingPlan(plan) {
    hideRagAnswer();
    setTrainingStatus(plan);
    trainingEls.enrichRagResults.classList.add("hidden");
    trainingEls.downloadLinks.innerHTML = "";
    trainingEls.summaryLine.textContent =
      `${plan.strategy_label}: ${plan.estimate.estimated_memory_gb.toFixed(2)} GB estimated for ${plan.model.repo_id}`;

    const warnings = plan.warnings || [];
    showMessage(warnings.length ? warnings.slice(0, 3).join(" ") : "");

    const statusLabel = {
      ok: "Enough memory",
      tight: "Tight",
      too_large: "Too large",
      unknown: "Unknown",
    }[plan.estimate.status] || "Unknown";

    trainingEls.planBox.innerHTML = `
      <div class="training-plan-head">
        <div>
          <h3>${escapeHtml(plan.strategy_label)}</h3>
          <p>${escapeHtml(plan.framework)} plan for ${escapeHtml(plan.model.repo_id)} using ${escapeHtml(plan.dataset.path)}</p>
        </div>
        <span class="training-estimate-pill ${escapeHtml(plan.estimate.status)}">${escapeHtml(statusLabel)}</span>
      </div>
      <div class="training-plan-grid">
        ${renderMetric("Need", `${plan.estimate.estimated_memory_gb.toFixed(2)} GB`)}
        ${renderMetric("Detected", plan.estimate.available_memory_gb == null ? "Unknown" : `${plan.estimate.available_memory_gb.toFixed(2)} GB`)}
        ${renderMetric("Rows", String(plan.dataset.row_count || 0))}
        ${renderMetric("Effective batch", String(plan.hyperparameters.per_device_batch_size * plan.hyperparameters.gradient_accumulation_steps))}
      </div>
      <h4>Memory Breakdown</h4>
      <ul class="training-list">
        ${Object.entries(plan.estimate.breakdown || {})
          .map(([key, value]) => `<li>${escapeHtml(labelize(key))}: ${Number(value).toFixed(2)} GB</li>`)
          .join("")}
      </ul>
      ${warnings.length ? `<h4>Warnings</h4><ul class="training-list">${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
      <h4>Install Optional Training Packages</h4>
      <pre class="training-code">${escapeHtml(plan.install_command)}</pre>
      <h4>CUE Training Config</h4>
      <pre class="training-code">${escapeHtml(JSON.stringify(plan.training_config, null, 2))}</pre>
      ${renderDatasetPreview(plan.dataset)}
    `;
    trainingEls.planBox.classList.remove("hidden");
    renderTrainingRows(plan);
  }

  function renderMetric(label, value) {
    return `<div class="training-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
  }

  function renderDatasetPreview(dataset) {
    const rows = dataset.preview || [];
    const columns = dataset.columns || [];
    if (!rows.length || !columns.length) return "";
    return `
      <h4>Data Preview</h4>
      <div class="training-preview">
        <table>
          <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows
              .map(
                (row) =>
                  `<tr>${columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("")}</tr>`,
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function renderTrainingRows(plan) {
    const table = trainingEls.resultsBody.closest("table");
    table.classList.remove("rag-table");
    table.querySelector("thead").innerHTML = `
      <tr>
        <th>Item</th>
        <th>Value</th>
        <th>Notes</th>
      </tr>
    `;
    const rows = [
      ["Strategy", plan.strategy_label, plan.strategy === "dpo" ? "Prompt, chosen, rejected" : "Input and target output"],
      ["Base model", plan.model.repo_id, `${plan.model.estimated_parameters_b}B parameter estimate`],
      ["Training data", plan.dataset.path, `${plan.dataset.row_count} rows, ${plan.dataset.format}`],
      ["Columns", (plan.required_columns || []).join(", "), "Validated against the file header"],
      ["Learning rate", String(plan.hyperparameters.learning_rate), `${plan.hyperparameters.epochs} epochs`],
      ["Sequence length", String(plan.hyperparameters.max_seq_length), "Lower this first if memory is tight"],
    ];
    trainingEls.resultsBody.innerHTML = rows
      .map(
        ([item, value, note]) => `
          <tr>
            <td>${escapeHtml(item)}</td>
            <td><pre class="cell-pre compact">${escapeHtml(value)}</pre></td>
            <td>${escapeHtml(note)}</td>
          </tr>
        `,
      )
      .join("");
  }

  async function estimateTraining(event) {
    event.preventDefault();
    hideTrainingPlan();
    showMessage("");
    trainingEls.statusLine.textContent = "Estimating training memory and checking dataset columns.";
    trainingEls.statusLine.className = "inline-status";
    trainingEls.statusLine.classList.remove("hidden");
    setBusy(true);
    try {
      const plan = await requestJson("/api/training/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildTrainingPayload()),
      });
      renderTrainingPlan(plan);
      if (devMode) {
        document.getElementById("debugOutput").textContent = JSON.stringify(plan, null, 2);
      }
    } catch (error) {
      trainingEls.statusLine.textContent = error.message;
      trainingEls.statusLine.className = "inline-status warn";
      trainingEls.statusLine.classList.remove("hidden");
      showMessage(error.message, true);
    } finally {
      setBusy(false);
    }
  }

  function labelize(value) {
    return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  document.querySelectorAll('input[name="trainingStrategy"]').forEach((input) => {
    input.addEventListener("change", updateTrainingStrategyFields);
  });
  trainingEls.modelLink.addEventListener("input", maybeUpdateModelSize);
  trainingEls.modelSize.addEventListener("input", () => {
    modelSizeEdited = true;
  });
  trainingEls.form.addEventListener("submit", estimateTraining);

  updateTrainingStrategyFields();
  refreshTrainingMachine();
})();
