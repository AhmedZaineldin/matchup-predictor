// Matchup Predictor frontend. No build step, no external libraries - plain fetch + DOM,
// since this app is meant to run standalone on a training machine that may not have
// internet access for CDN-hosted assets.

const state = {
  datasets: [],
  currentDataset: null,   // full dataset info object from /api/datasets
  entityCache: {},        // datasetName -> [{name, hasHistory}]
  debutantCache: {},      // datasetName -> Set of names with no completed fight
};

const els = {
  datasetTabs: document.getElementById("datasetTabs"),
  datasetTargetDescription: document.getElementById("datasetTargetDescription"),
  noModelWarning: document.getElementById("noModelWarning"),
  entity1Label: document.getElementById("entity1Label"),
  entity2Label: document.getElementById("entity2Label"),
  entity1Input: document.getElementById("entity1Input"),
  entity2Input: document.getElementById("entity2Input"),
  entity1List: document.getElementById("entity1List"),
  entity2List: document.getElementById("entity2List"),
  ufcContext: document.getElementById("ufcContext"),
  basketballContext: document.getElementById("basketballContext"),
  weightClassSelect: document.getElementById("weightClassSelect"),
  scheduledRoundsSelect: document.getElementById("scheduledRoundsSelect"),
  titleFightCheckbox: document.getElementById("titleFightCheckbox"),
  seasonTypeSelect: document.getElementById("seasonTypeSelect"),
  neutralSiteCheckbox: document.getElementById("neutralSiteCheckbox"),
  conferenceCompetitionCheckbox: document.getElementById("conferenceCompetitionCheckbox"),
  predictButton: document.getElementById("predictButton"),
  predictError: document.getElementById("predictError"),
  resultCard: document.getElementById("resultCard"),
  resultBlocks: document.getElementById("resultBlocks"),
  unavailableNote: document.getElementById("unavailableNote"),
  entity1Note: document.getElementById("entity1Note"),
  entity2Note: document.getElementById("entity2Note"),
  debutCaveat: document.getElementById("debutCaveat"),
};

async function FetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Request to ${url} failed (${response.status}).`);
  }
  return body;
}

async function Init() {
  els.predictButton.addEventListener("click", OnPredictClick);
  // Re-check the "no fight history" note whenever either name box changes.
  els.entity1Input.addEventListener("input", UpdateEntityNotes);
  els.entity2Input.addEventListener("input", UpdateEntityNotes);
  try {
    state.datasets = await FetchJson("/api/datasets");
  } catch (err) {
    els.datasetTabs.innerHTML = `<p class="error">Could not load datasets: ${err.message}</p>`;
    return;
  }
  RenderDatasetTabs();
  if (state.datasets.length > 0) {
    await SelectDataset(state.datasets[0].name);
  }
}

function RenderDatasetTabs() {
  els.datasetTabs.innerHTML = "";
  for (const ds of state.datasets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dataset-tab" + (ds.hasModel ? "" : " no-model");
    btn.textContent = ds.label + (ds.hasModel ? "" : " (no model)");
    btn.addEventListener("click", () => SelectDataset(ds.name));
    btn.dataset.name = ds.name;
    els.datasetTabs.appendChild(btn);
  }
}

function MarkActiveTab(name) {
  for (const btn of els.datasetTabs.children) {
    btn.classList.toggle("active", btn.dataset.name === name);
  }
}

async function SelectDataset(name) {
  const info = state.datasets.find((d) => d.name === name);
  if (!info) return;
  state.currentDataset = info;
  MarkActiveTab(name);

  els.datasetTargetDescription.textContent = info.targetDescription;
  els.noModelWarning.classList.toggle("hidden", info.hasModel);
  els.resultCard.classList.add("hidden");
  els.predictError.classList.add("hidden");

  const isUfc = info.sport === "ufc";
  els.entity1Label.textContent = isUfc ? "Fighter 1" : "Home team";
  els.entity2Label.textContent = isUfc ? "Fighter 2" : "Away team";
  els.entity1Input.placeholder = isUfc ? "Start typing a fighter's name…" : "Start typing a team name…";
  els.entity2Input.placeholder = els.entity1Input.placeholder;
  els.entity1Input.value = "";
  els.entity2Input.value = "";
  els.ufcContext.classList.toggle("hidden", !isUfc);
  els.basketballContext.classList.toggle("hidden", isUfc);

  await Promise.all([LoadEntities(name), LoadContextOptions(name)]);
}

async function LoadEntities(name) {
  if (!(name in state.entityCache)) {
    try {
      state.entityCache[name] = await FetchJson(`/api/entities?dataset=${encodeURIComponent(name)}`);
    } catch (err) {
      state.entityCache[name] = [];
      ShowError(err.message);
    }
  }
  const entities = state.entityCache[name];
  // Remember which entities have no completed-fight history, so the picker can label them.
  state.debutantCache[name] = new Set(
    entities.filter((e) => e.hasHistory === false).map((e) => e.name));
  for (const list of [els.entity1List, els.entity2List]) {
    list.innerHTML = "";
    for (const entity of entities) {
      const opt = document.createElement("option");
      opt.value = entity.name;
      // Browsers that render datalist option labels show this beside the name in the dropdown.
      if (entity.hasHistory === false) {
        opt.label = `${entity.name} — debut, no fight history`;
      }
      list.appendChild(opt);
    }
  }
  UpdateEntityNotes();
}

// Show or hide the "no fight history" note under each name box, based on what's typed in it.
function UpdateEntityNotes() {
  const debutants = state.currentDataset
    ? (state.debutantCache[state.currentDataset.name] || new Set())
    : new Set();
  const pairs = [[els.entity1Input, els.entity1Note], [els.entity2Input, els.entity2Note]];
  for (const [input, note] of pairs) {
    const isDebutant = debutants.has(input.value.trim());
    note.textContent = isDebutant
      ? "No completed UFC fight — league-typical stand-in stats will be used."
      : "";
    note.classList.toggle("hidden", !isDebutant);
  }
}

async function LoadContextOptions(name) {
  let options;
  try {
    options = await FetchJson(`/api/context-options?dataset=${encodeURIComponent(name)}`);
  } catch (err) {
    ShowError(err.message);
    return;
  }
  if (options.weightClasses) {
    els.weightClassSelect.innerHTML = options.weightClasses
      .map((w) => `<option value="${w}">${w}</option>`).join("");
    els.scheduledRoundsSelect.innerHTML = options.scheduledRoundsOptions
      .map((r) => `<option value="${r}">${r} rounds</option>`).join("");
  }
  if (options.seasonTypes) {
    els.seasonTypeSelect.innerHTML = options.seasonTypes
      .map((s) => `<option value="${s.value}">${s.label}</option>`).join("");
  }
}

function ShowError(message) {
  els.predictError.textContent = message;
  els.predictError.classList.remove("hidden");
}

async function OnPredictClick() {
  els.predictError.classList.add("hidden");
  const info = state.currentDataset;
  if (!info) return;

  const entity1 = els.entity1Input.value.trim();
  const entity2 = els.entity2Input.value.trim();
  if (!entity1 || !entity2) {
    ShowError(`Pick both ${info.sport === "ufc" ? "fighters" : "teams"} first.`);
    return;
  }

  const body = { dataset: info.name };
  if (info.sport === "ufc") {
    body.fighter1 = entity1;
    body.fighter2 = entity2;
    body.weightClass = els.weightClassSelect.value;
    body.scheduledRounds = Number(els.scheduledRoundsSelect.value);
    body.isTitleFight = els.titleFightCheckbox.checked;
  } else {
    body.homeTeam = entity1;
    body.awayTeam = entity2;
    body.seasonType = Number(els.seasonTypeSelect.value);
    body.neutralSite = els.neutralSiteCheckbox.checked;
    body.conferenceCompetition = els.conferenceCompetitionCheckbox.checked;
  }

  els.predictButton.disabled = true;
  els.predictButton.textContent = "Predicting…";
  try {
    const result = await FetchJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    RenderResult(info, entity1, entity2, result);
  } catch (err) {
    ShowError(err.message);
    els.resultCard.classList.add("hidden");
  } finally {
    els.predictButton.disabled = false;
    els.predictButton.textContent = "Predict";
  }
}

// Turn a raw class label ("1", "3", "KO/TKO", ...) into a human-readable string for display,
// given which dataset produced it and which two entities were in the matchup.
function FormatClassLabel(datasetName, entity1, entity2, label) {
  if (datasetName === "UfcRound") return `Round ${label}`;
  if (datasetName === "UfcMethod") return label;
  // Every other dataset in DATASET_INFO is a basketball home-win dataset with classes "0"/"1".
  return (label === "1") ? `${entity1} win (home)` : `${entity2} win (away)`;
}

// Render every prediction this matchup produced. A UFC matchup answers several questions at
// once (round of finish, method of victory), so each one gets its own block; basketball returns
// a single block for the league that was picked.
function RenderResult(info, entity1, entity2, result) {
  els.resultBlocks.innerHTML = "";

  // Qualify the prediction when either side has never actually fought in the UFC: their record,
  // finishing tendency and in-fight stats are league-typical stand-ins, not their own numbers,
  // so the result rests almost entirely on physicals and the opponent.
  const debutants = result.debutants || [];
  if (debutants.length > 0) {
    const who = debutants.join(" and ");
    els.debutCaveat.textContent =
      `${who} ${debutants.length > 1 ? "have" : "has"} no completed UFC fight. ` +
      `Their record, finishing tendency and in-fight stats are league-typical stand-ins, so this ` +
      `prediction rests mainly on physical attributes and the opponent's record — treat it as ` +
      `far less informed than a matchup between two fighters with history.`;
    els.debutCaveat.classList.remove("hidden");
  } else {
    els.debutCaveat.classList.add("hidden");
  }

  for (const prediction of result.predictions) {
    els.resultBlocks.appendChild(BuildPredictionBlock(prediction, entity1, entity2));
  }

  // Datasets with no model in best_models/ are reported quietly rather than as a failure —
  // the user simply hasn't trained or curated one for that question yet.
  const unavailable = result.unavailable || [];
  if (unavailable.length > 0) {
    els.unavailableNote.textContent =
      "Not shown: " + unavailable.map((u) => `${u.label} (${u.reason})`).join(", ") + ".";
    els.unavailableNote.classList.remove("hidden");
  } else {
    els.unavailableNote.classList.add("hidden");
  }

  els.resultCard.classList.remove("hidden");
  els.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Build one prediction block: a headline, the probability bars, and the drivers behind it.
function BuildPredictionBlock(prediction, entity1, entity2) {
  const block = document.createElement("div");
  block.className = "prediction-block";

  const topLabel = FormatClassLabel(prediction.dataset, entity1, entity2, prediction.predictedLabel);
  const topProb = prediction.probabilities[prediction.predictedLabel];

  const header = document.createElement("div");
  header.className = "prediction-header";
  header.innerHTML = `
    <div class="prediction-question">${EscapeHtml(prediction.targetDescription)}</div>
    <div class="predicted-headline">${EscapeHtml(topLabel)}
      <span class="top-prob">(${(topProb * 100).toFixed(1)}%)</span></div>
  `;
  block.appendChild(header);

  const bars = document.createElement("div");
  bars.className = "probability-bars";
  for (const [label, prob] of Object.entries(prediction.probabilities)) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-label">${EscapeHtml(FormatClassLabel(prediction.dataset, entity1, entity2, label))}</div>
      <div class="prob-track"><div class="prob-fill" style="width:${(prob * 100).toFixed(1)}%"></div></div>
      <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
    `;
    bars.appendChild(row);
  }
  block.appendChild(bars);

  if (prediction.explanation) {
    block.appendChild(BuildDriversPanel(prediction, topLabel));
  }
  if (prediction.orderAveraged) {
    const note = document.createElement("p");
    note.className = "order-note";
    note.textContent =
      "Averaged over both fighter orderings, so the answer doesn't change if you swap the names.";
    block.appendChild(note);
  }
  return block;
}

// Build the "why" panel: which groups of features pushed toward the predicted answer, and which
// pushed against it. Contributions come from the server as percentage-point swings measured by
// resetting that group to league-typical values and re-predicting.
function BuildDriversPanel(prediction, topLabel) {
  const panel = document.createElement("div");
  panel.className = "drivers-panel";

  const supporting = prediction.explanation.supporting || [];
  const opposing = prediction.explanation.opposing || [];
  if (supporting.length === 0 && opposing.length === 0) {
    panel.innerHTML = `<p class="drivers-empty">No single factor moved this prediction
      measurably — the model is reading it as a close, evenly-balanced matchup.</p>`;
    return panel;
  }

  // Scale every bar against the largest swing in this panel so the widths stay comparable.
  const maxSwing = Math.max(
    ...supporting.concat(opposing).map((d) => Math.abs(d.contributionPp)), 0.01);

  panel.innerHTML = `<h3 class="drivers-title">Why ${EscapeHtml(topLabel)}?</h3>`;
  if (supporting.length > 0) {
    panel.appendChild(BuildDriverList("Pushed toward this answer", supporting, maxSwing, "for"));
  }
  if (opposing.length > 0) {
    panel.appendChild(BuildDriverList("Pushed against it", opposing, maxSwing, "against"));
  }
  return panel;
}

// Build one side of the drivers panel.
function BuildDriverList(heading, drivers, maxSwing, direction) {
  const wrapper = document.createElement("div");
  wrapper.className = `driver-group driver-group-${direction}`;
  wrapper.innerHTML = `<div class="driver-heading">${EscapeHtml(heading)}</div>`;
  for (const driver of drivers) {
    const magnitude = Math.abs(driver.contributionPp);
    const row = document.createElement("div");
    row.className = "driver-row";
    row.innerHTML = `
      <div class="driver-label">
        <span class="driver-name">${EscapeHtml(driver.label)}</span>
        ${driver.detail ? `<span class="driver-detail">${EscapeHtml(driver.detail)}</span>` : ""}
      </div>
      <div class="driver-track">
        <div class="driver-fill driver-fill-${direction}"
             style="width:${((magnitude / maxSwing) * 100).toFixed(1)}%"></div>
      </div>
      <div class="driver-value">${driver.contributionPp > 0 ? "+" : "−"}${magnitude.toFixed(1)} pts</div>
    `;
    wrapper.appendChild(row);
  }
  return wrapper;
}

// Escape anything that came from data (fighter names, labels) before putting it in innerHTML.
function EscapeHtml(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

Init();
