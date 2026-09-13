# matchup-predictor

**Pick two fighters. Pick two teams. Find out what happens — and why.**

A config-driven machine learning pipeline for two-sided contests, plus a local web app that turns
whatever model you trained into a live prediction you can actually interrogate.

Two fighters walk into the octagon. Two teams tip off. The interesting question isn't just *who
wins* — it's **what round it ends in, how it ends, and which piece of the tale of the tape the
model actually cared about.** This repo trains models to answer that, and then shows its work.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw CSVs       │ --> │  9-step pipeline │ --> │  best_models/   │
│  UFC + hoops    │     │  x 4,032 combos  │     │  your keepers   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                    ┌─────────────────────▼─────────┐
                                    │  PredictorApp — pick two,     │
                                    │  hit Predict, read the why    │
                                    └───────────────────────────────┘
```

---

## What's in the box

**Three pipelines that share one brain.** The classical sweep, the from-scratch deep learning
pipeline, and the pretrained foundation models all run through the *same* Steps 1–7 — cleaning,
outlier removal, splitting, redundant-column pruning, imbalance handling, normalization,
three-method feature selection — imported from one place, so a comparison between them is a fair
fight rather than an accident of preprocessing.

**A grid you can make as big as your patience.** 14 classical models × 6 scalers × 8 imbalance
techniques × 6 outlier techniques is **4,032 combinations per dataset**, each one Optuna-tuned and
dropped into its own folder with confusion matrices, per-split metrics, feature importances and
row-level predictions. Comment lines out of `config.yaml` to shrink it; uncomment to sweep wider.
A combination that blows up gets logged and skipped — the grid keeps going.

**Models that don't train.** Alongside a standard `MLP` and a from-scratch `TabTransformer`
(every column becomes a token, attention learns the interactions), the DL pipeline can run
**TabPFN**, **TabICL** and **SAP-RPT-1-OSS** — transformers pretrained on millions of synthetic
tables that do in-context learning instead of gradient descent. They read your training rows the
way an LLM reads examples in a prompt.

**A predictor that explains itself.** Not just a number — a ranked breakdown of what pushed the
prediction there and what pushed against it, in percentage points, with the actual stats beside
each one.

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python TabularMLPipeline.py --config config.yaml      # classical sweep
python DeepLearningPipeline.py --config config_dl.yaml  # neural + pretrained
```

Then curate your keepers into `best_models/<Dataset>/` and launch the app:

```bash
cd PredictorApp && python app.py     # http://localhost:5000
```

The data is already in `Data/` — nothing to download.

---

## The six questions it answers

| Dataset | Predicts | Classes |
|---|---|---|
| `UfcRound` | Which round the fight ends in | 1–5 |
| `UfcMethod` | How it ends | KO/TKO · Submission · Decision |
| `NbaWin` | Home team wins | yes / no |
| `WnbaWin` | Home team wins | yes / no |
| `NcaaMbbWin` | Home team wins | yes / no |
| `NcaaWbbWin` | Home team wins | yes / no |

The UFC feature builder turns six raw ufcstats CSVs into **8,694 fights × 98 leakage-safe
features** — career and last-5 form for striking, takedowns, control time and knockdowns;
finishing tendency; layoff and activity rate; stance matchups; and a fighter-1-minus-fighter-2
difference for every one of them. Every stat is computed *entering* the fight, never including
it.

---

## The app

Type two names, set the context, hit Predict. You get:

- **Every question at once.** One UFC matchup runs round *and* method together — same fighters,
  same bout, two answers with full probability bars.
- **The whole roster.** All ~4,610 fighters on ufcstats, not just the 2,747 with a fight history.
  Debutants get the exact zero-history profile the training set contains 923 examples of, and the
  UI flags them so you know the prediction is thin.
- **A "why" panel.** Reach, weight, finishing tendency, control time, streak — ranked by how many
  percentage points each one moved *this* prediction, measured by resetting it to league-typical
  and re-predicting. Works identically for a random forest, a TabTransformer or a refit TabPFN,
  because it goes through the same `.Predict()` call the real prediction did.
- **Answers that don't depend on typing order.** ufcstats lists the winner first in every bout
  string, so "fighter 1" carries real positional signal — swapping two names changed the predicted
  round in 36% of test matchups. The app predicts both orderings and averages them. Basketball is
  left alone on purpose: home vs away is a real edge, not an artifact.

No build step, no CDN, no internet required — vanilla HTML/CSS/JS against a Flask backend, so it
runs on an air-gapped training box.

---

## Under the hood, briefly

A few things that were more interesting to build than they look:

**No saved encoder, no drift.** The app reconstructs training-time one-hot encoding and scaling at
inference from `scaler.feature_names_in_` alone — the trained column set and order come straight
off the scaler, so a live prediction can't silently disagree with how the model was trained.

**Pretrained models get refit, not reloaded.** TabPFN and friends have no run-specific weights to
restore, so "loading" one replays Steps 1–7 from its `ConfigUsed.yaml` and fits fresh. Cheap,
because there's no gradient descent to redo.

**Nine optimizers, and honest fallbacks.** Adam, AdamW, SGD, RMSprop, RAdam, Lion, Prodigy,
ScheduleFree, Sophia — with AMP, EMA and early stopping. The three that need optional packages say
so out loud and fall back to AdamW rather than failing a six-hour sweep at hour five.

---

## Known limits

- **There is no winner model for UFC.** The registry defines round and method only. Adding one
  means a new target *and* mirrored training rows — ufcstats orders bouts winner-first (5,579
  `W/L` vs 3,138 `L/W`), so an unmirrored winner model would mostly learn "fighter 1 usually wins".
- **Basketball `season_type` labels are an educated guess.** The raw data has integer codes and no
  dictionary.
- **`restDays` goes quiet on stale data.** It's real days against today's date, clipped at 30.
- **Driver rankings are descriptive, not causal.** They say what this model keyed on for this row,
  not what would actually decide the fight.

---

## Layout

```
matchup-predictor/
├── TabularMLPipeline.py       Classical sweep + the shared Steps 1-7
├── DeepLearningPipeline.py    MLP / TabTransformer / TabPFN / TabICL / SAP-RPT-1-OSS
├── UFCFeatures.py             UFC feature engineering + live fighter snapshot
├── BasketballFeatures.py      Basketball feature engineering + live team snapshot
├── PredictorApp/              Flask app, feature builder, model loader, explainer
├── Data/                      Raw UFC + basketball CSVs
├── best_models/               Your curated picks, one folder per dataset
├── config.yaml                Classical grid
└── config_dl.yaml             Deep learning grid
```

Full documentation — every config key, every output file, the setup dance for the gated
checkpoints — lives in the pipeline README and `PredictorApp/README.md`.
