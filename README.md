# Tabular ML Pipeline — your UFC + basketball data, run it locally

A config-driven ML pipeline: `config.yaml` lists which datasets to run and
which model / scaler / imbalance-technique / outlier-technique combinations
to sweep. Each combination runs the full 9-step pipeline (cleaning, outlier
removal, split, redundant-column removal, class-imbalance handling,
normalization, 3-method feature selection, Optuna hyperparameter tuning,
final evaluation) and gets its own output folder — same idea as your own
`Models x Scalers` grid script.

## 1. Set up locally

You need Python 3.10+ installed. Then, from this folder:

```bash
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Your data

Already included in `Data/UFC/` and `Data/Basketball/` — nothing to copy.
`config.yaml` is set to run all 6 of your datasets by default:

| Dataset name | What it predicts | Source files |
|---|---|---|
| `UfcRound` | Which round a UFC fight ends in (1–5) | `Data/UFC/ufc_fight_results.csv`, `ufc_event_details.csv`, `ufc_fighter_tott.csv` |
| `UfcMethod` | How a UFC fight ends: KO/TKO, Submission, or Decision | same UFC files as above |
| `NbaWin` | Whether the home team wins an NBA game | `Data/Basketball/schedule.csv`, `team_box.csv` |
| `WnbaWin` | Whether the home team wins a WNBA game | same basketball files, filtered to WNBA |
| `NcaaMbbWin` | Whether the home team wins an NCAA men's basketball game | same basketball files, filtered to NCAA men's |
| `NcaaWbbWin` | Whether the home team wins an NCAA women's basketball game | same basketball files, filtered to NCAA women's |

What "built-in" means: the CSV loading and feature engineering for these 6
already live in `UFCFeatures.py` and `BasketballFeatures.py`, so the config
just says `Builtin: UfcRound` instead of a `CsvPath` — you don't set a
target column or drop columns yourself for these, because that's already
been worked out and hard-coded into those two files. Specifically:

- **UFC (`UfcRound` / `UfcMethod`)**: every feature is something known
  *before* the fight — each fighter's height, reach, weight, age, stance,
  and win/loss record and streak entering the fight (computed
  chronologically so a fighter's record never includes the fight being
  predicted), plus the weight class, whether it's a title fight, and how
  many rounds the bout is scheduled for (3 vs. 5 - set when the fight is
  booked, so this is known in advance, not leakage).

  In-fight statistics (significant strikes landed/absorbed, striking
  accuracy, takedowns landed/accuracy, control time, knockdowns, submission
  attempts) are used too, but never *that fight's own* numbers - that would
  be leakage, since they don't exist until after the fight happens. Instead,
  each fighter's **career-to-date average** of these stats, computed only
  from their earlier fights (the same chronological "entering" approach used
  for win/loss record), is carried in as a feature - "how this fighter
  typically performs," which is legitimate pre-fight signal, exactly
  analogous to the basketball pipeline's rolling pre-game team averages. A
  fighter's UFC debut fight has no earlier fights to average, so these start
  out missing and get filled in by the pipeline's usual imputation, same as
  any other feature with no history yet.

  **A note on `UfcRound` specifically**: it mixes two different phenomena
  into one 5-way target. If a fight goes the distance (Decision), `ROUND`
  is just the bout's scheduled length (round 3 of a 3-round fight, round 5
  of a 5-round fight) - not a reflection of anything the fighters did, so
  the `scheduledRounds` feature makes that ~46% of the dataset almost
  trivially learnable. If a fight ends in a finish (KO/TKO or Submission),
  the round is genuinely hard to predict from pre-fight stats - two skilled
  fighters can finish each other in round 1 or grind to round 5, and
  pre-fight form only weakly signals which. Round 4 in particular is
  extremely rare (about 0.6% of fights - it can only happen in a 5-round
  fight, and specifically a finish landing in that one round), so expect
  any model to struggle on that class regardless of how clean the data is;
  a wide train/test gap on `UfcRound` is partly this genuine class rarity
  and unpredictability, not purely overfitting from a pipeline bug.
- **Basketball (`NbaWin` / `WnbaWin` / `NcaaMbbWin` / `NcaaWbbWin`)**: every
  feature is a team's rolling form *entering* the game — last-10-game and
  season-to-date averages of scoring, shooting %, rebounds, assists,
  steals, blocks, turnovers, fouls, points in the paint, fast-break points,
  shot volume (field goals/threes/free throws *attempted*, not just made -
  how many shots a team typically takes), turnover points allowed, largest
  lead, and technical fouls, plus win/loss record, current streak, and rest
  days, for both the home and away team (and the home-minus-away
  difference). A completed game's own box score is never used to predict
  that same game — that would be leakage, since you don't know the box
  score before the game happens. Three columns present in the raw data are
  deliberately left out of every rolling feature: `team_curated_rank` is
  100% empty for every league (nothing to average), and `lead_changes` /
  `lead_percentage` are missing on ~84% of rows - too sparse to trust.

If you ever want to add a CSV of your own alongside these, `config.yaml` has
a commented-out example showing the format (`CsvPath` + `TargetColumn`
instead of `Builtin`) — just uncomment and fill it in.

## 3. Choose what to sweep

`config.yaml` has four grid lists. Every combination of everything listed
runs and gets its own `Exp-{Dataset}-{Model}-{Scaler}-{Imbalance}-{Outlier}/`
folder — comment lines out (with `#`) to control how large the grid gets:

```yaml
Models: [RandomForest, XGBoost]              # + ExtraTrees, GradientBoosting, AdaBoost,
                                               #   DecisionTree, KNN, LogisticRegression, SVC,
                                               #   GaussianNB, SGD, MLP, Bagging, LightGBM
Scalers: [Standard]                           # + MinMax, Robust, MaxAbs, Normalizer, QuantileTransformer
ImbalanceMethod: [SMOTE]                      # + ADASYN, BorderlineSMOTE, RandomOverSampler,
                                               #   RandomUnderSampler, SMOTETomek, ClassWeight, None
OutlierMethod: [IQR]                          # + ZScore, IsolationForest, LocalOutlierFactor,
                                               #   EllipticEnvelope, None
```

Every option in every list is implemented and tested — the full menu is
written out (commented) in `config.yaml` so you can just uncomment what you
want to try. As shipped: 6 datasets x 2 models x 1 scaler x 1 imbalance
method x 1 outlier method = 12 experiments per run. Widen any list to sweep
more; each added option multiplies the total run count.

## 4. Run it

```bash
python TabularMLPipeline.py --config config.yaml
```

A failed combination (e.g. a model that doesn't converge on a particular
dataset) is logged and skipped — the rest of the grid keeps going.

## 5. What you get, per experiment

```
Results/Exp-<Dataset>-<Model>-<Scaler>-<Imbalance>-<Outlier>/
  FullExperimentResult.json   Every step's report end-to-end
  ClassDistribution.png       Class counts across Train/Val/Test
  OptunaHistory.png           Validation macro-F1 per trial + running best
  FeatureImportance.png       Top-20 final-model feature importances
  BestModel.joblib            {Model, Features, LabelEncoder, Scaler} - reload with joblib.load(...)
  ConfigUsed.yaml              Config snapshot for this specific run
  Test/  Val/  Train/          Per split: {Split}CM.png, {Split}CM.csv,
                               {Split}ClassificationReport.txt,
                               {Split}EvaluationMetrics.json (accuracy,
                               precision/recall/F1 macro & weighted, ROC-AUC,
                               log loss, full TP/FP/FN/TN metric suite per
                               class and macro/weighted),
                               {Split}DetailedPredictions.csv (RowId, Actual,
                               Pred, PredProb per row)
```

## Notes on specific options

- **Models without a native `feature_importances_` or `coef_`** (KNN,
  GaussianNB, MLP, non-linear-kernel SVC) get an all-zero
  `FeatureImportance.png` — that's expected, not a bug; those algorithms
  don't expose per-feature importance in scikit-learn.
- **`ClassWeight` has no effect for KNN and MLP** — neither supports
  `class_weight` or `sample_weight` in scikit-learn. Every other model either
  takes `class_weight` directly or gets weighted via `sample_weight` at fit
  time automatically; you don't need to configure which.
- **`EllipticEnvelope`** assumes roughly Gaussian features and can fail on
  highly collinear data; if it does, that run logs a warning and skips
  outlier removal for that experiment rather than crashing.
- **Outlier removal always has a safety cap** (`OutlierMaxRowDropFrac`,
  default 0.15) — no method is allowed to remove more than that fraction of
  rows, regardless of how aggressive its own math says to be.
- **Missing-value imputation uses only the training fold's median** — Train,
  Val, and Test all get their remaining missing values filled with the
  median computed from Train, the same way `NormalizeFeatures` fits its
  scaler on Train only. (An earlier version of this pipeline computed each
  split's median from its own rows, which quietly let Val/Test's own
  distribution leak into their own imputed values - fixed.)
- **Train/Val accuracy in `TabularMLPipeline.py`'s output can look almost
  perfect (90-99%), especially on the basketball datasets, while Test stays
  much lower** - this looks alarming but is a known characteristic of this
  pipeline's design, not a leakage bug: once Optuna hyperparameter tuning
  is done, `TrainFinalModel()` deliberately re-fits the chosen model on
  **Train+Val merged**, since a classical model like RandomForest/XGBoost
  doesn't need a held-out set once its hyperparameters are already chosen.
  That means the Train/Val numbers you see are essentially in-sample
  (measuring how well the model memorized the data it was just fit on), and
  **only the `Test/` folder is a genuinely held-out measurement** - that's
  the number to trust. The more features a dataset has (basketball's rolling
  stats give tree models a lot to work with), the easier it is for a model
  to fit that merged Train+Val fold almost perfectly, so this gap tends to
  widen as you add features - watch Test, not Train/Val, to judge whether a
  change actually helped. The deep-learning pipeline below does not have
  this characteristic: it keeps Val genuinely held out through the final fit
  (needed for early stopping), so Train/Val/Test all stay meaningfully
  distinct there.

## Extending it further

- **Add a model**: add its name to `SUPPORTED_MODELS`, a case in
  `SuggestHyperparameters()` (the Optuna search space) and in
  `BuildModelFromParams()` (the constructor) in `TabularMLPipeline.py`.
  Everything else (tuning, evaluation, plotting) picks it up automatically.
- **Add an imbalance or outlier technique**: same pattern, one `elif` branch
  in `HandleClassImbalance()` / `RemoveOutliers()`.
- **Add your own CSV**: see the commented example in `config.yaml`'s
  `Datasets` list.
- **Voting/Stacking ensembles**: not included (they need sub-model config of
  their own), but could be added the same way if you want them.

## 6. Deep learning version (PyTorch)

`DeepLearningPipeline.py` is a second, parallel pipeline: same 6 datasets,
same Steps 1-7 (cleaning, outliers, split, redundant columns, imbalance,
normalization, 3-method feature selection) imported directly from
`TabularMLPipeline.py` so both pipelines clean data identically - but Steps
8-9 (tuning + final model) are a PyTorch neural network instead of a
scikit-learn/XGBoost model. It's driven by its own config file,
`config_dl.yaml`, and run the same way:

```bash
python DeepLearningPipeline.py --config config_dl.yaml
```

### Architectures

- **`MLP`** - a standard feedforward network: configurable number of
  hidden layers and width, with BatchNorm + ReLU + Dropout between each.
- **`TabTransformer`** - a from-scratch FT-Transformer/TabTransformer-style
  network: every feature (after scaling and one-hot encoding) becomes its
  own "token" via a shared linear projection, a learned per-feature
  embedding tells the model which column is which, a CLS token is
  prepended, and stacked self-attention blocks mix information across
  features before the CLS token is classified. This lets the network learn
  feature interactions on its own, the way attention does for text.

`MLP` and `TabTransformer` are both trained from scratch on your data,
the normal way (random init -> gradient descent -> Optuna-tuned
hyperparameters -> early stopping). The next three architectures are
different in kind:

- **`TabPFN`** - [Prior Labs' TabPFN](https://huggingface.co/Prior-Labs/TabPFN-v2-clf),
  a transformer *pretrained once on millions of synthetic datasets* and
  shipped as a fixed checkpoint. It never trains on your data with gradient
  descent at all - at "fit" time it just stores your training rows, and at
  predict time it does in-context learning (like asking an LLM to keep
  examples in its prompt), reading your stored rows as context to make each
  prediction. Recommended for up to ~10,000 rows / ~500 features.
- **`TabICL`** - a similar in-context tabular foundation model, open-weight
  (BSD-3-Clause) and not gated on Hugging Face, scaling comfortably to
  ~100,000 rows.
- **`SapRpt1Oss`** - [SAP's SAP-RPT-1-OSS](https://huggingface.co/SAP/sap-rpt-1-oss),
  another pretrained in-context tabular model, the one you originally linked.
  Its checkpoints are released for research purposes and it needs **Python
  3.11 specifically** plus realistically a CUDA GPU (ideally ~80GB VRAM) -
  see "Setting up the pretrained models" below before trying it.

Because these three don't train with gradient descent, `OptimizerName`,
`LossFunction`, `SchedulerName`, `UseAmp`, `UseEma`, `NumEpochs`, and
`Patience` in `config_dl.yaml` are all silently ignored when the
architecture is one of them - there's no training loop for those settings
to apply to. "Tuning" for these three (Step 8) instead picks the best of a
few cheap inference-time settings (ensemble size / softmax temperature /
bagging count) by validation macro-F1 - see the
`PretrainedN*`/`Pretrained*Temp*`/`SapRpt*` keys in `config_dl.yaml`.

### Setting up the pretrained models

`pip install -r requirements.txt` already installs `tabpfn` and `tabicl` -
both run fine on CPU at these dataset sizes, no extra setup needed for
**TabICL**. **TabPFN** is different: its checkpoint is a *gated* Hugging
Face repo, so the first time you run it you need to:

1. Accept the license at https://huggingface.co/Prior-Labs/TabPFN-v2-clf
   (requires a free Hugging Face account).
2. Authenticate locally: `pip install huggingface_hub` then
   `hf auth login` (or set the `HF_TOKEN` environment variable to a token
   generated at https://huggingface.co/settings/tokens).

After that one-time setup, TabPFN downloads its checkpoint automatically on
first use and caches it locally - later runs don't need internet access
for it again. **`SapRpt1Oss` is intentionally NOT in `requirements.txt`** -
it requires Python 3.11 specifically (a different interpreter than the rest
of this pipeline needs) and a CUDA GPU with a lot of VRAM, so it needs its
own environment:

```bash
# in a separate Python 3.11 venv:
pip install git+https://github.com/SAP-samples/sap-rpt-1-oss
```

`BuildPretrainedModel()` in `DeepLearningPipeline.py` already has the full
integration code for it (constructs `SAP_RPT_OSS_Classifier`, etc.) - if
the package isn't installed, adding `SapRpt1Oss` to `config_dl.yaml`'s
`Architecture` list just raises a clear `ImportError` telling you what to
install, rather than crashing confusingly.

**`ClassWeight` has no effect on any of the three pretrained architectures**
- none of them take a `sample_weight`/`class_weight` argument at fit time
  (there's no "fit" in the gradient-descent sense to weight). If you want
  to address class imbalance for these three, use one of the actual
  resampling methods (`SMOTE`, `ADASYN`, etc.) instead - the pipeline
  prints a warning if you combine `ClassWeight` with a pretrained
  architecture, since it's silently a no-op there.

Every pretrained architecture caps how many training rows it actually sees
(subsampled once per run, not per Optuna trial) - see the row-cap note in
`config_dl.yaml` next to `PretrainedNOptunaTrials` for the defaults and how
to override them per architecture.

This sandbox this pipeline was developed in has no internet access to
Hugging Face, so the TabPFN/TabICL/SapRpt1Oss integration code was verified
against each library's real Python API (constructor arguments,
`.fit()`/`.predict()`/`.predict_proba()`) rather than an actual end-to-end
checkpoint download - expect the very first run on your machine to need
internet access (and, for TabPFN, the one-time license+token step above).

### Training features

Mirrors your own image-classification script's training loop:

- **Optimizers** (`OptimizerName`): Adam, AdamW, SGD, RMSprop, RAdam, Lion,
  Prodigy, ScheduleFreeAdamW, Sophia. RAdam and Sophia need no extra
  packages (Sophia is implemented directly in `DeepLearningPipeline.py`);
  Lion/Prodigy/ScheduleFreeAdamW need their optional pip packages (see
  `requirements.txt`) and silently fall back to AdamW with a printed
  warning if not installed.
- **Loss functions** (`LossFunction`): CrossEntropy, LabelSmoothing, Focal
  (for hard/imbalanced examples).
- **LR schedulers** (`SchedulerName`): None, CosineWarmup, StepLR,
  ReduceLROnPlateau.
- **AMP** (`UseAmp`): automatic mixed precision - real speedup on a CUDA
  GPU, harmless (no speedup) on CPU.
- **EMA** (`UseEma`): keeps an exponential-moving-average shadow copy of the
  weights and evaluates/checkpoints from that instead of the raw weights -
  usually a small, free accuracy bump.
- **Early stopping**: the final fit trains up to `NumEpochs` but stops once
  validation loss hasn't improved for `Patience` epochs in a row; Optuna
  trials use their own shorter `OptunaMaxEpochsPerTrial`/`OptunaPatience` so
  tuning stays fast.

### Why train+val isn't merged before the final fit

The classical pipeline (`TabularMLPipeline.py`) merges train+val for the
final model, since scikit-learn models don't need a held-out set once
hyperparameters are chosen. Neural nets are different: early stopping
*during* the final fit is what decides how many epochs to train for, and
that needs its own held-out validation fold to watch. So here, Train stays
Train and Val stays Val all the way through - only Test is held out purely
for the final numbers, exactly as before.

### Output, per experiment

Same idea as the classical pipeline's `Results/`, under `ResultsDL/` by
default:

```
ResultsDL/Exp-<Dataset>-<Architecture>-<Optimizer>-<Loss>-<Scaler>-<Imbalance>-<Outlier>/
  FullExperimentResult.json   Every step's report end-to-end, plus TrainingHistory
  TrainingHistory.png         Train/Val loss and accuracy curves over epochs
  ClassDistribution.png       Class counts across Train/Val/Test
  OptunaHistory.png           Validation macro-F1 per trial + running best
  BestModel.pt                Best model weights (reload with torch.load)
  ModelMetadata.joblib        {Features, LabelEncoder, Scaler, Architecture, BestParams}
  ConfigUsed.yaml              Config snapshot for this specific run
  Test/  Val/  Train/          Same per-split files as the classical pipeline:
                               {Split}CM.png/.csv, {Split}ClassificationReport.txt,
                               {Split}EvaluationMetrics.json, {Split}DetailedPredictions.csv
```

### Notes specific to the deep-learning pipeline

- **No `FeatureImportance.png`** - neural nets don't expose per-feature
  importance the way tree models do, so that plot is only in the classical
  pipeline.
- **Slower** - even the smallest grid here trains real neural networks, so
  it's much slower per experiment than `TabularMLPipeline.py`. `config_dl.yaml`
  starts with a small grid (1 architecture x 1 optimizer x 1 loss x 1 scaler
  x 1 imbalance x 1 outlier x 6 datasets = 6 experiments) for that reason -
  widen it once you've confirmed it runs comfortably on your machine.
- **CPU vs GPU**: `Device: auto` in `config_dl.yaml` uses a CUDA GPU if
  `torch.cuda.is_available()`, otherwise CPU. `pip install torch` from
  `requirements.txt` installs the CPU build unless you already have CUDA
  set up; either works, GPU is just faster.
- **Extending it**: add an architecture the same way as the classical
  pipeline's models - a branch in `BuildModel()` and `SuggestDlHyperparameters()`.
  Add an optimizer/loss/scheduler the same way, in `GetOptimizer()` /
  `GetLossFunction()` / `GetScheduler()`.

## 7. Running live predictions with your best models

`PredictorApp/` is a small local website for actually using your trained models: pick two
fighters or two teams, fill in a bit of context, get a prediction with a probability
breakdown - no notebook or script needed. It reads whichever models you copy into a
`best_models/` folder here (one subfolder per dataset) and reconstructs each fighter's/team's
current stats straight from `Data/`, so predictions always reflect their latest form.

See `PredictorApp/README.md` for the exact folder layout it expects and how to run it
(`cd PredictorApp && python app.py`, then open `http://localhost:5000`).
