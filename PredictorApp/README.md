# Matchup Predictor

A small local website that runs live predictions with your best trained models: pick two
fighters (UFC) or two teams (NBA/WNBA/NCAA-MBB/NCAA-WBB), fill in a bit of context, and get
a prediction with a probability breakdown.

It does **not** run the training pipelines itself. It loads whichever models you've already
trained and copied into a `best_models/` folder, reconstructs each fighter's/team's
current stats straight from the raw CSVs in `Data/`, and feeds that through the exact same
feature engineering + encoding + scaling steps training used.

## 1. Folder layout it expects

This folder (`PredictorApp/`) must live directly inside `LocalPipeline/`, next to
`TabularMLPipeline.py`, `DeepLearningPipeline.py`, `UFCFeatures.py`, `BasketballFeatures.py`,
and `Data/` — exactly where it already sits in this delivered zip. Don't move it out on its own.

```
LocalPipeline/
├── TabularMLPipeline.py
├── DeepLearningPipeline.py
├── UFCFeatures.py
├── BasketballFeatures.py
├── Data/
│   ├── UFC/...
│   └── Basketball/...
├── best_models/                  <- you create this
│   ├── UfcRound/
│   ├── UfcMethod/
│   ├── NbaWin/
│   ├── WnbaWin/
│   ├── NcaaMbbWin/
│   └── NcaaWbbWin/
└── PredictorApp/
    ├── app.py
    ├── model_loader.py
    ├── feature_builder.py
    ├── templates/index.html
    └── static/{style.css,app.js}
```

`best_models/` doesn't need every dataset filled in — any dataset without a folder just shows
up in the UI tagged "no model" and can't be predicted yet. Add folders for whichever datasets
you actually have a best result for.

### What goes inside each `best_models/<DatasetName>/` folder

Copy the **whole output folder** a pipeline run produced (in `Results/` for
`TabularMLPipeline.py`, `ResultsDL/` for `DeepLearningPipeline.py`) for your best experiment
on that dataset, and rename it to the dataset name. Which files actually matter depends on
what kind of model it was:

- **Classical model** (anything from `TabularMLPipeline.py` — RandomForest, XGBoost, SVC,
  etc.): only `BestModel.joblib` is needed. It's fully self-contained (model + selected
  features + label encoder + scaler all bundled together).

- **From-scratch deep learning model** (`DeepLearningPipeline.py`, architecture `MLP` or
  `TabTransformer`): needs both `ModelMetadata.joblib` **and** `BestModel.pt` side by side.

- **Pretrained foundation model** (`DeepLearningPipeline.py`, architecture `TabPFN`,
  `TabICL`, or `SapRpt1Oss`): these have no saved weights of their own to reload — "loading"
  one means refitting it fresh at startup using the exact settings that experiment used. That
  needs `ModelMetadata.joblib` **plus `ConfigUsed.yaml` and `FullExperimentResult.json`**, all
  three from the same run. If you only copy `ModelMetadata.joblib` for one of these, the app
  will fail to load that dataset with a clear error naming the missing files. Refitting a
  pretrained model happens once, the first time that dataset is requested (not at startup),
  and the app keeps it in memory after that.

  If you're using `SapRpt1Oss`, remember it needs the separate Python 3.11 environment
  documented in the main `README.md` — run the predictor app from that same environment if any
  of your best models use it.

You don't need to touch any of the plot images (`OptunaHistory.png`, `FeatureImportance.png`,
etc.) — they're harmless to leave in the folder, the app just ignores them.

## 2. Running it

```bash
cd LocalPipeline/PredictorApp
pip install flask
python app.py
```

Then open `http://localhost:5000` (or `http://<machine's LAN IP>:5000` from another device on
the same network). Leave the terminal running — closing it stops the site.

By default it looks for models in `LocalPipeline/best_models/` and raw data in
`LocalPipeline/Data/UFC` and `LocalPipeline/Data/Basketball`. Override any of these with
environment variables if you keep them somewhere else:

```bash
BEST_MODELS_DIR=/path/to/best_models UFC_DATA_DIR=/path/to/ufc/csvs python app.py
```

`PORT` also works if 5000 is taken by something else.

## 3. How a prediction actually happens

1. The app rebuilds each fighter's or team's **current** career snapshot straight from the raw
   CSVs the first time that sport is used (`UFCFeatures.BuildCurrentFighterSnapshot()` /
   `BasketballFeatures.BuildCurrentTeamSnapshot()`), then keeps it cached in memory. This is
   the unshifted, up-to-the-present version of the same stats the training features use — not
   a lookup into the training dataset itself.
2. `feature_builder.py` combines the two chosen fighters'/teams' snapshots plus whatever
   context you picked in the UI (weight class, title fight, rounds / neutral site, conference
   game, season type) into one raw feature row, using the exact column names training used.
3. `model_loader.py` one-hot encodes and scales that row using the dataset's saved scaler
   (`scaler.feature_names_in_` recovers the exact column set and order training fit on, with no
   separate encoder file needed), selects down to that model's chosen features, and calls
   `predict_proba`.

## 4. Fighters with no completed UFC fight

The UFC name list covers the **whole ufcstats roster (~4,610 fighters)**, not just the 2,747
who have actually fought. A fighter whose only appearance in the data is an UPCOMING bout —
Joe Kropschot, Roberto Soldic and 8 others currently booked — has no fight history to average,
so `BuildCurrentFighterSnapshot()` gives them the same profile `BuildUFCDataset()` produces for
any fighter's **first** fight:

| Feature group | Value for a debutant |
|---|---|
| `WinsEntering`, `LossesEntering`, `FightsEntering`, `StreakEntering` | `0` |
| `WinPctEntering` | `0.5` (even prior) |
| `DaysSinceLastFightEntering` | `180.0` |
| Finishing rates, in-fight stats, Roll5 recent form | league-typical stand-in (see below) |
| Height / reach / weight / DOB / stance | their real values from `ufc_fighter_tott.csv` |

This is a legitimate case, not a fudge: **923 of the training set's 8,694 rows are debut rows**
(510 have both fighters debuting), so the model has learned what a zero-history fighter looks
like. Training filled those rows' NaNs with the training split's per-column median, so the
snapshot fills them with the equivalent median over fighters who have fought — **weighted by
each fighter's number of fights**, because training's median was over one row per
fighter-*appearance*, not per fighter. An unweighted median over the roster badly undershoots
it (e.g. `KoWinRateEntering` 0.000 vs training's 0.154, `SigStrLandedEntering` 28.2 vs 33.7);
fight-weighting lands within ~6% on average instead of ~24%.

The UI flags these fighters: a note appears under the name box as soon as one is picked, and
the result card carries a caveat. `/api/entities` returns `{"name": ..., "hasHistory": bool}`
per fighter and `/api/predict` echoes a `debutants` list, so the flag is available to anything
else calling the API. **Treat a debutant prediction as much weaker** — it rests almost entirely
on physical attributes and the opponent's record.

One related fix ships with this: the bout strings and the attributes table spell 35 fighters'
names differently (`Joe Kropschot` vs `Joseph Kropschot`, `Kai Kamaka III` vs `Kai Kamaka`),
which left **16 fighters who have fought** silently receiving median height/reach/weight and a
missing date of birth. Those now resolve through the shared fighter-details URL. The alias is
only ever used to fill a gap, never to override a name that already matched — the two raw files
genuinely disagree about identity in a couple of places, so overriding would break as much as
it fixes.

## 5. What the result card shows, and why

**One UFC matchup answers every UFC question at once.** The same two fighters and the same bout
context feed `UfcRound` and `UfcMethod` together, so a single Predict shows both the round of
finish and the method of victory, each with its own probability bars. A dataset with no folder
in `best_models/` is listed quietly under the result instead of failing the request.

> **There is no "who wins" model.** `TASK_REGISTRY` defines only `roundClass` and `methodClass`
> for UFC — no winner target exists in the dataset, so the app cannot show one. Adding it means
> a new task with a `targetF1Win` column, and it should be trained with mirrored rows: ufcstats
> lists the winner first in a BOUT string (5,579 `W/L` vs 3,138 `L/W`), so an unmirrored winner
> model would mostly learn "fighter 1 usually wins".

**Every prediction is order-averaged (UFC only).** That same positional bias leaks into the
round and method models too. Measured on a test model, swapping the two fighter names changed
the predicted round in **36% of 120 random matchups**, shifting probabilities by a median of
12.5 points — for a question about the *bout*, which should not depend on typing order. So the
app predicts both orderings and averages them (`feature_builder.SwapUfcMatchupRow`), which makes
the answer exactly order-independent (verified: 0 class flips and a 0.00000000pp gap across 150
random matchups). Basketball is deliberately **not** averaged — home vs away is a real
asymmetry the model should be using. Send `"orderAveraged": false` to `/api/predict` to see the
raw model output instead.

**The "why" panel** ranks groups of features by how much each one moved this specific
prediction, as a percentage-point swing, split into what pushed toward the answer and what
pushed against it.

The method is **occlusion**: reset one group of related features to league-typical values,
re-predict, and measure how far the predicted class's probability moved. This runs through the
same `.Predict(row)` interface the real prediction used, so it works identically for a classical
estimator, a from-scratch MLP/TabTransformer and a refit pretrained model — no SHAP dependency,
and no risk of the explainer disagreeing with the model about what the model does.

Features are grouped rather than occluded one at a time because the raw row is deliberately
redundant: reach reaches the model as `f1reachIn`, `f2reachIn` **and** `reachDiff`, so blanking
any one alone barely moves anything and would make reach look irrelevant. The ~96 raw UFC
columns collapse into 18 readable drivers (~15 for basketball), and the grouping is checked for
completeness — every raw feature belongs to exactly one group, no feature appears twice, and
neutralizing all groups at once reproduces the neutral-matchup prediction to within 1e-9.

The league-typical baseline is **weighted by fights/games played**, for the same reason the
debut fighters above are: training rows are per-appearance, not per-entity.

**Read the drivers as "what this model keyed on", not as causal claims.** They are a local,
single-row measurement; a driver's swing depends on the other features around it, and two
correlated groups can each look smaller than their combined effect.

## 6. Known limitations

- **`restDays` (basketball) can be a constant.** It's computed as real elapsed days between a
  team's last game in the data and today's actual date, clipped to 0–30. If your basketball
  data is old relative to when you run this, every team's `restDays` hits the 30-day clip
  ceiling and stops being an informative feature for that prediction. This doesn't break
  anything — it's just a feature that goes quiet when the data is stale.
- **Basketball `season_type` codes are a best guess.** The raw schedule data has no label
  column for it, just an integer code (1/2/3/5 observed). The dropdown labels
  ("Preseason"/"Regular Season"/"Postseason"/"All-Star or Other") are inferred from the
  standard convention used by this style of ESPN-derived sports data, not confirmed against an
  official data dictionary — if predictions seem off for postseason games specifically, this
  mapping is the first thing worth double-checking.
- **A handful of NBA/WNBA "teams" in the dropdown aren't actually NBA/WNBA teams** (e.g. an
  Adelaide 36ers-style entry showing up under the NBA list). This comes from the raw schedule
  data tagging some preseason exhibition games against overseas clubs under the same league
  code as regular games — it's a pre-existing data quirk from the source CSVs, not something
  this app introduces, and the underlying models were trained on those rows too. It's safe to
  ignore in the UI; just don't be surprised to see an unfamiliar name in the list occasionally.
- **A UFC fighter with no recorded stance** predicts fine (the model just treats stance as
  "missing," matching how training handles it) but the age fallback for a fighter with no
  recorded date of birth defaults to 30 years old — a reasonable league-average guess, not a
  real value.
- **Changes to `Data/` or `best_models/` need a restart.** Snapshots and models are both built
  once and cached in memory for the life of the running process, to keep predictions fast.
