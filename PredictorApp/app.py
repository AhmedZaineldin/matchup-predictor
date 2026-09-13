import difflib
import os
import sys
import threading
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup. This file is meant to be run directly (`python app.py`) from
# INSIDE the PredictorApp/ folder, but it works no matter what directory the
# user launches it from, because every path below is built from this file's
# own location rather than the process's current working directory.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent          # .../LocalPipeline/PredictorApp
BASE_DIR = APP_DIR.parent                          # .../LocalPipeline
sys.path.insert(0, str(BASE_DIR))                  # so "import TabularMLPipeline" etc. work

import model_loader                                # noqa: E402  (import after sys.path fix)
import feature_builder                             # noqa: E402
import explain                                     # noqa: E402
import UFCFeatures as uf                           # noqa: E402
import BasketballFeatures as bf                    # noqa: E402

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402


def fprint(msg):
  # Print immediately, matching the rest of the pipeline's logging style.
  print(msg, flush=True)


# ---------------------------------------------------------------------------
# Configuration. Every path can be overridden with an environment variable,
# but sensible defaults are derived from this file's location so the app
# just works when the whole LocalPipeline folder is copied somewhere new.
# ---------------------------------------------------------------------------
UFC_DATA_DIR = os.environ.get("UFC_DATA_DIR", str(BASE_DIR / "Data" / "UFC"))
BASKETBALL_DATA_DIR = os.environ.get("BASKETBALL_DATA_DIR", str(BASE_DIR / "Data" / "Basketball"))
BEST_MODELS_DIR = Path(os.environ.get("BEST_MODELS_DIR", BASE_DIR / "best_models"))

# Every dataset this app knows how to serve, and what kind of matchup UI it needs.
# "League" is BasketballFeatures's lowercase league code (only used for basketball datasets).
DATASET_INFO = {
  "UfcRound": {
    "Sport": "ufc", "Label": "UFC – Round of Finish", "EntityLabel": "Fighter",
    "TargetDescription": "Which round the fight ends in",
  },
  "UfcMethod": {
    "Sport": "ufc", "Label": "UFC – Method of Victory", "EntityLabel": "Fighter",
    "TargetDescription": "How the fight ends (KO/TKO, submission, decision, etc.)",
  },
  "NbaWin": {
    "Sport": "basketball", "League": "nba", "Label": "NBA – Home Win", "EntityLabel": "Team",
    "TargetDescription": "Whether the home team wins",
  },
  "WnbaWin": {
    "Sport": "basketball", "League": "wnba", "Label": "WNBA – Home Win", "EntityLabel": "Team",
    "TargetDescription": "Whether the home team wins",
  },
  "NcaaMbbWin": {
    "Sport": "basketball", "League": "ncaa_mbb", "Label": "NCAA Men's Basketball – Home Win",
    "EntityLabel": "Team", "TargetDescription": "Whether the home team wins",
  },
  "NcaaWbbWin": {
    "Sport": "basketball", "League": "ncaa_wbb", "Label": "NCAA Women's Basketball – Home Win",
    "EntityLabel": "Team", "TargetDescription": "Whether the home team wins",
  },
}

# UFC weight classes, as seen in the training data (UFCFeatures.BuildUFCDataset()'s
# weightClassClean column). Hardcoded here (rather than rebuilding the whole ~20s UFC
# dataset just to read one column) since this list is stable domain knowledge.
UFC_WEIGHT_CLASSES = [
  "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
  "Middleweight", "Light Heavyweight", "Heavyweight", "Catch Weight", "Open Weight", "Other",
  "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
]
UFC_SCHEDULED_ROUNDS = [3, 5]

# season_type in the raw basketball schedule data is a numeric code with no label column
# anywhere in the source CSVs. These labels are inferred from the standard convention used by
# ESPN-derived sports datasets (the same convention hoopR/sportsdataverse-style scrapes use:
# 1=preseason, 2=regular season, 3=postseason, 4=play-in, 5=all-star/other) and from the
# observed value counts (2 is overwhelmingly the most common, consistent with "regular
# season"). Treat these labels as a best-effort guess, not a verified data dictionary.
BASKETBALL_SEASON_TYPES = [
  {"value": 1, "label": "Preseason"},
  {"value": 2, "label": "Regular Season"},
  {"value": 3, "label": "Postseason"},
  {"value": 5, "label": "All-Star / Other"},
]

app = Flask(__name__, static_folder=str(APP_DIR / "static"), template_folder=str(APP_DIR / "templates"))

# ---------------------------------------------------------------------------
# In-memory caches. Models and snapshots are each expensive to build (a pretrained
# model "load" is actually a full refit; a snapshot rereads and re-aggregates every
# raw CSV), so each is built at most once per running process and reused after that.
# A snapshot never goes stale mid-process because the underlying CSVs don't change
# while the app is running; restart the app after updating the raw data or the
# best_models/ folder to pick up changes.
# ---------------------------------------------------------------------------
_modelCache = {}
_modelLock = threading.Lock()
_snapshotCache = {}
_snapshotLock = threading.Lock()


def GetModel(datasetName):
  with _modelLock:
    if (datasetName not in _modelCache):
      fprint(f"Loading model for {datasetName} (first request for this dataset)...")
      _modelCache[datasetName] = model_loader.LoadModelForDataset(datasetName, BEST_MODELS_DIR)
      fprint(f"Model for {datasetName} ready.")
    return _modelCache[datasetName]


def GetFighterSnapshot():
  with _snapshotLock:
    if ("ufc" not in _snapshotCache):
      fprint("Building current fighter snapshot (first UFC request)...")
      _snapshotCache["ufc"] = uf.BuildCurrentFighterSnapshot(dataDir=UFC_DATA_DIR)
      fprint(f"Fighter snapshot ready: {len(_snapshotCache['ufc'])} fighters.")
    return _snapshotCache["ufc"]


def GetTeamSnapshot(league):
  with _snapshotLock:
    key = f"bb_{league}"
    if (key not in _snapshotCache):
      fprint(f"Building current team snapshot for {league} (first request for this league)...")
      snap = bf.BuildCurrentTeamSnapshot(league, dataDir=BASKETBALL_DATA_DIR)
      # Index by team name (unique per league) so lookups in /api/predict are simple.
      _snapshotCache[key] = snap.reset_index(drop=True).set_index("teamName")
      fprint(f"Team snapshot ready for {league}: {len(_snapshotCache[key])} teams.")
    return _snapshotCache[key]


def DatasetHasModel(datasetName):
  folder = BEST_MODELS_DIR / datasetName
  return ((folder / "BestModel.joblib").exists() or (folder / "ModelMetadata.joblib").exists())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def Index():
  return send_from_directory(str(APP_DIR / "templates"), "index.html")


@app.route("/api/datasets")
def ApiDatasets():
  result = []
  for name, info in DATASET_INFO.items():
    result.append({
      "name": name,
      "sport": info["Sport"],
      "label": info["Label"],
      "entityLabel": info["EntityLabel"],
      "targetDescription": info["TargetDescription"],
      "hasModel": DatasetHasModel(name),
    })
  return jsonify(result)


@app.route("/api/entities")
def ApiEntities():
  datasetName = request.args.get("dataset", "")
  if (datasetName not in DATASET_INFO):
    return jsonify({"error": f"Unknown dataset '{datasetName}'."}), 400
  info = DATASET_INFO[datasetName]
  try:
    if (info["Sport"] == "ufc"):
      # The UFC snapshot spans the whole ufcstats roster, including fighters with a booked debut
      # and no completed fight yet. Those are predictable (the training set contains 923 debut
      # rows) but their features are league-typical stand-ins rather than their own record, so
      # flag them here and let the UI say so instead of quietly treating them like a veteran.
      snap = GetFighterSnapshot()
      names = sorted(snap.index.tolist())
      hasHistory = snap["HasFightHistory"].astype(bool).to_dict()
      entities = [{"name": n, "hasHistory": bool(hasHistory.get(n, True))} for n in names]
    else:
      entities = [{"name": n, "hasHistory": True}
                  for n in sorted(GetTeamSnapshot(info["League"]).index.tolist())]
  except FileNotFoundError as e:
    return jsonify({"error": str(e)}), 500
  return jsonify(entities)


@app.route("/api/context-options")
def ApiContextOptions():
  datasetName = request.args.get("dataset", "")
  if (datasetName not in DATASET_INFO):
    return jsonify({"error": f"Unknown dataset '{datasetName}'."}), 400
  info = DATASET_INFO[datasetName]
  if (info["Sport"] == "ufc"):
    return jsonify({"weightClasses": UFC_WEIGHT_CLASSES, "scheduledRoundsOptions": UFC_SCHEDULED_ROUNDS})
  return jsonify({"seasonTypes": BASKETBALL_SEASON_TYPES})


# Define the function that lists every dataset one matchup can be run through. A UFC matchup is
# the same two fighters and the same bout context whichever question you ask about it, so one
# click can answer all of them; each basketball league is its own set of teams, so a basketball
# matchup only ever runs the league the user picked.
def BuildPredictFn(model, sport: str, symmetric: bool):
  # Basketball keeps the raw model: home vs away is a real asymmetry, not an artifact.
  if (sport != "ufc" or not symmetric):
    return model.Predict

  # UFC averages the model's answer over both fighter orderings, so "who did you type first"
  # stops changing the result. See feature_builder.SwapUfcMatchupRow for why this is needed.
  def PredictOrderAveraged(row):
    _, forwardProbs = model.Predict(row)
    _, reverseProbs = model.Predict(feature_builder.SwapUfcMatchupRow(row))
    averaged = {cls: (forwardProbs[cls] + reverseProbs.get(cls, 0.0)) / 2.0 for cls in forwardProbs}
    return max(averaged, key=averaged.get), averaged

  return PredictOrderAveraged


def SiblingDatasets(datasetName: str):
  info = DATASET_INFO[datasetName]
  if (info["Sport"] != "ufc"):
    return [datasetName]
  # Keep the dataset the user picked first, then every other UFC dataset.
  others = [n for n, i in DATASET_INFO.items() if (i["Sport"] == "ufc" and n != datasetName)]
  return [datasetName] + others


@app.route("/api/predict", methods=["POST"])
def ApiPredict():
  body = request.get_json(force=True, silent=True) or {}
  datasetName = body.get("dataset", "")
  if (datasetName not in DATASET_INFO):
    return jsonify({"error": f"Unknown dataset '{datasetName}'."}), 400
  info = DATASET_INFO[datasetName]
  wantExplanations = bool(body.get("explain", True))
  symmetric = bool(body.get("orderAveraged", True))

  try:
    if (info["Sport"] == "ufc"):
      snap = GetFighterSnapshot()
      fighter1Name, fighter2Name = body.get("fighter1", ""), body.get("fighter2", "")
      missing = [n for n in (fighter1Name, fighter2Name) if (n not in snap.index)]
      if (missing):
        # A name that isn't on the roster at all is nearly always a spelling difference, so
        # offer the closest roster entries rather than a bare rejection.
        suggestions = {}
        for name in missing:
          close = difflib.get_close_matches(name, snap.index.tolist(), n=3, cutoff=0.6)
          if (close):
            suggestions[name] = close
        message = f"Unknown fighter(s): {missing}"
        if (suggestions):
          message += " - did you mean: " + "; ".join(
            f"{name} -> {', '.join(options)}" for name, options in suggestions.items())
        return jsonify({"error": message, "suggestions": suggestions}), 400
      if (fighter1Name == fighter2Name):
        return jsonify({"error": "Pick two different fighters."}), 400
      # Note which side (if either) has no completed UFC fight, so the result can be qualified.
      debutants = [n for n in (fighter1Name, fighter2Name)
                   if (not bool(snap.loc[n, "HasFightHistory"]))]
      row = feature_builder.BuildUfcMatchupRow(
        snap.loc[fighter1Name], snap.loc[fighter2Name],
        weightClass=body.get("weightClass", "Lightweight"),
        isTitleFight=bool(body.get("isTitleFight", False)),
        scheduledRounds=float(body.get("scheduledRounds", 3)),
      )
      entity1Name, entity2Name = fighter1Name, fighter2Name
    else:
      debutants = []
      snap = GetTeamSnapshot(info["League"])
      homeTeam, awayTeam = body.get("homeTeam", ""), body.get("awayTeam", "")
      missing = [n for n in (homeTeam, awayTeam) if (n not in snap.index)]
      if (missing):
        return jsonify({"error": f"Unknown team(s): {missing}"}), 400
      if (homeTeam == awayTeam):
        return jsonify({"error": "Pick two different teams."}), 400
      row = feature_builder.BuildBasketballMatchupRow(
        snap.loc[homeTeam], snap.loc[awayTeam],
        neutralSite=bool(body.get("neutralSite", False)),
        conferenceCompetition=bool(body.get("conferenceCompetition", False)),
        seasonType=body.get("seasonType", 2),
      )
      entity1Name, entity2Name = homeTeam, awayTeam
  except Exception as e:
    fprint(f"Building the matchup failed for {datasetName}: {e}\n{traceback.format_exc()}")
    return jsonify({"error": f"Building the matchup failed: {e}"}), 500

  # Work out the driver groups once - they depend only on the sport, not on the model.
  if (info["Sport"] == "ufc"):
    driverGroups = explain.UFC_DRIVER_GROUPS
  else:
    driverGroups = explain.BuildBasketballDriverGroups(bf.BOX_STAT_COLS)

  # Run the matchup through every model that has a best_models/ folder. A UFC matchup answers
  # several questions at once (round of finish, method of victory), so all of them are returned
  # together; a dataset with no model is reported as unavailable rather than failing the request.
  predictions, unavailable = [], []
  for siblingName in SiblingDatasets(datasetName):
    siblingInfo = DATASET_INFO[siblingName]
    try:
      model = GetModel(siblingName)
    except FileNotFoundError:
      unavailable.append({"dataset": siblingName, "label": siblingInfo["Label"],
                          "reason": "no model in best_models/"})
      continue
    except Exception as e:
      fprint(f"Model load failed for {siblingName}: {e}\n{traceback.format_exc()}")
      unavailable.append({"dataset": siblingName, "label": siblingInfo["Label"],
                          "reason": f"model failed to load ({e})"})
      continue

    try:
      predictFn = BuildPredictFn(model, siblingInfo["Sport"], symmetric)
      predictedLabel, probabilities = predictFn(row)
      # Sort descending so the frontend can render bars in order without re-sorting.
      sortedProbs = dict(sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True))
      entry = {
        "dataset": siblingName,
        "label": siblingInfo["Label"],
        "targetDescription": siblingInfo["TargetDescription"],
        "predictedLabel": predictedLabel,
        "probabilities": sortedProbs,
        "orderAveraged": bool(symmetric and siblingInfo["Sport"] == "ufc"),
      }
      if (wantExplanations):
        entry["explanation"] = explain.ExplainPrediction(
          predictFn, row, snap, siblingInfo["Sport"], driverGroups, entity1Name, entity2Name)
      predictions.append(entry)
    except Exception as e:
      fprint(f"Prediction failed for {siblingName}: {e}\n{traceback.format_exc()}")
      unavailable.append({"dataset": siblingName, "label": siblingInfo["Label"],
                          "reason": f"prediction failed ({e})"})

  if (not predictions):
    detail = "; ".join(f"{u['label']}: {u['reason']}" for u in unavailable)
    return jsonify({"error": f"No usable model for this matchup. {detail}"}), 404

  return jsonify({
    "entity1": entity1Name, "entity2": entity2Name,
    "predictions": predictions, "unavailable": unavailable, "debutants": debutants,
  })


if (__name__ == "__main__"):
  port = int(os.environ.get("PORT", 5000))
  fprint(f"Best models directory: {BEST_MODELS_DIR}")
  fprint(f"UFC data directory: {UFC_DATA_DIR}")
  fprint(f"Basketball data directory: {BASKETBALL_DATA_DIR}")
  for name in DATASET_INFO:
    fprint(f"  {name}: {'model found' if DatasetHasModel(name) else 'NO MODEL FOUND'}")
  app.run(host="0.0.0.0", port=port, debug=False)
