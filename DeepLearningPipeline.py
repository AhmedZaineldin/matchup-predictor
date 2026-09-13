import csv  # Import the csv module for writing CSV files.
import json  # Import the json module for data serialization.
import yaml  # Import the yaml module for configuration parsing.
import time  # Import the time module for timing operations.
import numpy  # Import the numpy module for numerical operations.
import pandas  # Import the pandas module for tabular data handling.
import optuna  # Import optuna for hyperparameter tuning.
import warnings  # Import the warnings module to silence noisy library warnings.
import joblib  # Import joblib for scaler/encoder persistence.
import torch  # Import torch for deep learning.
import torch.nn as nn  # Import the neural network module from torch.
from tqdm import tqdm  # Import tqdm for progress bars.
from pathlib import Path  # Import the Path class from pathlib.
from typing import Dict, List, Tuple, Optional, Union, Any  # Import typing utilities.
from torch.utils.data import Dataset, DataLoader  # Import the Dataset and DataLoader utilities from torch.
from torch.optim.swa_utils import AveragedModel  # Import AveragedModel for EMA.
from sklearn.metrics import confusion_matrix, classification_report  # Import metric functions from sklearn.

# Import every generic step this pipeline shares with TabularMLPipeline.py, so the
# cleaning / outlier / split / redundant-column / imbalance / normalization / feature
# selection logic is defined in exactly one place and never drifts between the two.
from TabularMLPipeline import (
  fprint, NumpyEncoder, LoadConfig, ResolveDataset,
  DropUnnecessaryColumns, RemoveOutliers, SplitData, DropRedundantColumns,
  EncodeCategoricalColumns, HandleClassImbalance, NormalizeFeatures, SelectFeatures,
  CalculatePerformanceMetrics, PlotConfusionMatrix, PlotClassDistribution,
  SUPPORTED_SCALERS, SUPPORTED_IMBALANCE_METHODS, SUPPORTED_OUTLIER_METHODS,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib.

# Silence noisy library warnings so the console log stays readable.
warnings.filterwarnings("ignore")
# Silence optuna's per-trial console spam (we print our own summary lines instead).
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import the optional third-party optimizers exactly like the image-classification
# script does, falling back gracefully (with a printed warning) if one isn't installed.
try:
  from lion_pytorch import Lion
  LION_AVAILABLE = True
except ImportError:
  LION_AVAILABLE = False
try:
  from prodigyopt import Prodigy
  PRODIGY_AVAILABLE = True
except ImportError:
  PRODIGY_AVAILABLE = False
try:
  from schedulefree import AdamWScheduleFree
  SCHEDULEFREE_AVAILABLE = True
except ImportError:
  SCHEDULEFREE_AVAILABLE = False

# Import the optional pretrained tabular foundation models - each is a self-contained,
# already-pretrained in-context learner with a plain sklearn-style .fit()/.predict() API: .fit()
# just stores your training rows as "context" (no gradient descent, no epochs, seconds not
# minutes), and .predict()/.predict_proba() condition on that context to classify new rows. This
# is what "pretrained instead of trained from scratch" means for tabular data - there's no
# Hugging Face `transformers`-style text/vision model that applies here, since these are a
# different, purpose-built model family for rows-of-numbers data.
try:
  from tabpfn import TabPFNClassifier
  TABPFN_AVAILABLE = True
except ImportError:
  TABPFN_AVAILABLE = False
try:
  from tabicl import TabICLClassifier
  TABICL_AVAILABLE = True
except ImportError:
  TABICL_AVAILABLE = False
try:
  from sap_rpt_oss import SAP_RPT_OSS_Classifier
  SAP_RPT_AVAILABLE = True
except ImportError:
  SAP_RPT_AVAILABLE = False


# Define the sets of architectures / optimizers / loss functions / schedulers this
# pipeline supports. Comment options out of your config.yaml's lists to skip them.
# MLP and TabTransformer are trained from scratch on your data (Steps 8-9 below); TabPFN,
# TabICL, and SapRpt1Oss are pretrained foundation models used purely for inference (see the
# "PRETRAINED_ARCHITECTURES" block and README for setup requirements, GPU/CPU notes, and the
# per-model row-count caps applied automatically before .fit()).
PRETRAINED_ARCHITECTURES = ["TabPFN", "TabICL", "SapRpt1Oss"]
SUPPORTED_ARCHITECTURES = ["MLP", "TabTransformer"] + PRETRAINED_ARCHITECTURES
SUPPORTED_OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSprop", "RAdam", "Lion", "Prodigy", "ScheduleFreeAdamW", "Sophia"]
SUPPORTED_LOSS_FUNCTIONS = ["CrossEntropy", "Focal", "LabelSmoothing"]
SUPPORTED_SCHEDULERS = ["None", "CosineWarmup", "StepLR", "ReduceLROnPlateau"]

# Practical row-count ceiling for each pretrained model's training context, applied by randomly
# subsampling down to this many rows before .fit() if the actual training fold is larger. These
# aren't training-from-scratch capacity limits (nothing is being "trained" - .fit() just stores
# context) - they're the regime each model's own documentation recommends for reliable inference
# and reasonable memory/runtime, especially on CPU. Override per-architecture via
# "<Architecture>MaxTrainRows" in your config (e.g. "TabPFNMaxTrainRows: 5000").
PRETRAINED_MAX_TRAIN_ROWS = {"TabPFN": 10000, "TabICL": 100000, "SapRpt1Oss": 8192}


# Define the FocalLoss class, for training on hard-to-classify / imbalanced examples.
class FocalLoss(nn.Module):
  """
  Standard multiclass Focal Loss: down-weights easy (already well-classified)
  examples so training focuses on hard ones. A generalization of cross-entropy
  (gamma=0 recovers plain weighted cross-entropy).
  """

  # Initialize the loss function.
  def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
    # Call the parent constructor.
    super().__init__()
    # Store the focusing parameter.
    self.gamma = gamma
    # Store the optional per-class weight tensor.
    self.weight = weight

  # Define the forward pass.
  def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Compute the standard per-sample cross-entropy loss (unreduced).
    ceLoss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
    # Compute the model's predicted probability of the true class.
    pt = torch.exp(-ceLoss)
    # Apply the focal weighting term and average over the batch.
    return ((1 - pt) ** self.gamma * ceLoss).mean()


# Define the SophiaG optimizer, implemented locally to avoid an extra dependency.
class SophiaG(torch.optim.Optimizer):
  """
  A simplified SophiaG optimizer (diagonal-Hessian-proxy second-order method).
  """

  # Initialize the optimizer.
  def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=1e-1, weight_decay=1e-1):
    # Build the defaults dictionary.
    defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
    # Call the parent constructor.
    super().__init__(params, defaults)

  # Define the optimization step.
  @torch.no_grad()
  def step(self, closure=None):
    # Optionally evaluate the closure to get the loss.
    loss = closure() if (closure is not None) else None
    # Iterate through every parameter group.
    for group in self.param_groups:
      # Unpack the beta coefficients.
      beta1, beta2 = group["betas"]
      # Iterate through every parameter in this group.
      for p in group["params"]:
        # Skip parameters with no gradient.
        if (p.grad is None):
          continue
        # Fetch this parameter's optimizer state.
        state = self.state[p]
        # Initialize the state on first use.
        if (len(state) == 0):
          state["step"] = 0
          state["expAvg"] = torch.zeros_like(p)
          state["expAvgSq"] = torch.zeros_like(p)
        # Apply weight decay directly to the parameter.
        if (group["weight_decay"] != 0):
          p.data.mul_(1 - group["lr"] * group["weight_decay"])
        # Update the first and second moment estimates.
        state["step"] += 1
        state["expAvg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
        state["expAvgSq"].mul_(beta2).add_(p.grad.abs(), alpha=1 - beta2)
        # Compute the clamped update and apply it.
        update = torch.clamp(state["expAvg"] / (state["expAvgSq"] + 1e-8), -group["rho"], group["rho"])
        p.data.add_(update, alpha=-group["lr"])
    # Return the loss (if a closure was provided).
    return loss


# Define the TabularDataset class, wrapping a feature matrix + target vector + row ids.
class TabularDataset(Dataset):
  # Initialize the dataset.
  def __init__(self, features: pandas.DataFrame, targets: pandas.Series, rowIds: pandas.Series):
    # Store the features as a float32 tensor.
    self.features = torch.tensor(features.values, dtype=torch.float32)
    # Store the targets as a long tensor (required for CrossEntropyLoss).
    self.targets = torch.tensor(targets.values, dtype=torch.long)
    # Store the row ids as a plain list (kept out of the tensors so DataLoader collation stays simple).
    self.rowIds = list(rowIds)

  # Return the number of samples.
  def __len__(self) -> int:
    return len(self.targets)

  # Return one (features, target, rowId) triple.
  def __getitem__(self, idx: int):
    return self.features[idx], self.targets[idx], self.rowIds[idx]


# Define the MlpClassifier: a standard feedforward network for tabular data.
class MlpClassifier(nn.Module):
  # Initialize the network.
  def __init__(self, numFeatures: int, numClasses: int, hiddenDims: List[int], dropout: float):
    # Call the parent constructor.
    super().__init__()
    # Build the hidden layers one at a time.
    layers = []
    inputDim = numFeatures
    for hiddenDim in hiddenDims:
      # Add a linear layer, batch norm, activation, and dropout for this block.
      layers += [nn.Linear(inputDim, hiddenDim), nn.BatchNorm1d(hiddenDim), nn.ReLU(), nn.Dropout(dropout)]
      # Advance the running input dimension.
      inputDim = hiddenDim
    # Store the hidden layers as a sequential backbone.
    self.backbone = nn.Sequential(*layers)
    # Define the final classification head.
    self.head = nn.Linear(inputDim, numClasses)

  # Define the forward pass.
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Run the backbone, then the classification head.
    return self.head(self.backbone(x))


# Define the TabularTransformerBlock: one pre-norm self-attention + MLP block.
class TabularTransformerBlock(nn.Module):
  # Initialize the block.
  def __init__(self, embedDim: int, numHeads: int, mlpRatio: float = 2.0, dropout: float = 0.1):
    # Call the parent constructor.
    super().__init__()
    # Define the first normalization layer.
    self.norm1 = nn.LayerNorm(embedDim)
    # Define the multi-head self-attention layer.
    self.attn = nn.MultiheadAttention(embedDim, numHeads, batch_first=True, dropout=dropout)
    # Define the second normalization layer.
    self.norm2 = nn.LayerNorm(embedDim)
    # Define the feedforward MLP.
    self.mlp = nn.Sequential(
      nn.Linear(embedDim, int(embedDim * mlpRatio)), nn.GELU(),
      nn.Dropout(dropout), nn.Linear(int(embedDim * mlpRatio), embedDim),
    )
    # Define the residual dropout.
    self.dropout = nn.Dropout(dropout)

  # Define the forward pass.
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Apply pre-norm self-attention with a residual connection.
    attnOut, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
    x = x + self.dropout(attnOut)
    # Apply the pre-norm MLP with a residual connection.
    x = x + self.dropout(self.mlp(self.norm2(x)))
    # Return the updated token sequence.
    return x


# Define the TabTransformerClassifier: every feature becomes its own token.
class TabTransformerClassifier(nn.Module):
  """
  FT-Transformer / TabTransformer-style architecture for already-numeric,
  already-scaled tabular features (categoricals have already been one-hot
  encoded by Step 6, so there is no separate categorical-embedding table here -
  every column, including the one-hot ones, is treated as a continuous token).
  Each scalar feature is linearly projected to an embedding, a learned
  per-feature "positional" embedding tells the model which column is which,
  a CLS token is prepended, and standard Transformer blocks mix information
  across features before the CLS token is classified.
  """

  # Initialize the network.
  def __init__(self, numFeatures: int, numClasses: int, embedDim: int, numHeads: int,
               numLayers: int, dropout: float):
    # Call the parent constructor.
    super().__init__()
    # Store the number of input features.
    self.numFeatures = numFeatures
    # Define the shared per-feature scalar-to-embedding projection.
    self.featureEmbed = nn.Linear(1, embedDim)
    # Define the learned per-feature identity embedding.
    self.featurePosEmbed = nn.Parameter(torch.zeros(1, numFeatures, embedDim))
    # Define the learned CLS token.
    self.clsToken = nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the stack of Transformer blocks.
    self.blocks = nn.ModuleList(
      [TabularTransformerBlock(embedDim, numHeads, dropout=dropout) for _ in range(numLayers)])
    # Define the final normalization layer.
    self.norm = nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = nn.Linear(embedDim, numClasses)
    # Initialize the learned embeddings.
    nn.init.trunc_normal_(self.featurePosEmbed, std=0.02)
    nn.init.trunc_normal_(self.clsToken, std=0.02)

  # Define the forward pass.
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Get the batch size.
    b = x.shape[0]
    # Add a trailing dimension so each scalar feature becomes a length-1 "sequence".
    x = x.unsqueeze(-1)
    # Project every feature scalar to its embedding.
    x = self.featureEmbed(x)
    # Add the per-feature identity embedding.
    x = x + self.featurePosEmbed
    # Prepend the CLS token.
    clsTokens = self.clsToken.expand(b, -1, -1)
    x = torch.cat([clsTokens, x], dim=1)
    # Run every Transformer block.
    for block in self.blocks:
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x)
    # Classify from the CLS token's final representation.
    return self.head(x[:, 0])


# Define the BuildModel function: dispatches an architecture name to its module.
def BuildModel(architecture: str, numFeatures: int, numClasses: int, hyperparams: Dict[str, Any]) -> nn.Module:
  # Check which architecture was requested.
  if (architecture == "MLP"):
    # Build the MLP classifier from the sampled hyperparameters.
    return MlpClassifier(
      numFeatures, numClasses, hiddenDims=hyperparams["hiddenDims"], dropout=hyperparams["dropout"])
  elif (architecture == "TabTransformer"):
    # Build the Transformer classifier from the sampled hyperparameters.
    return TabTransformerClassifier(
      numFeatures, numClasses, embedDim=hyperparams["embedDim"], numHeads=hyperparams["numHeads"],
      numLayers=hyperparams["numLayers"], dropout=hyperparams["dropout"])
  else:
    # Raise an error for an unrecognized architecture name.
    raise ValueError(f"Unsupported architecture: {architecture}. Choose from {SUPPORTED_ARCHITECTURES}.")


# Define the BuildPretrainedModel function: dispatches a pretrained-architecture name to its
# already-pretrained classifier instance. Unlike BuildModel() above, nothing here is trained from
# scratch - these constructors just configure how the frozen pretrained model does in-context
# inference (how many ensemble members, how confident its softmax is, etc.).
def BuildPretrainedModel(architecture: str, hyperparams: Dict[str, Any], randomState: int) -> Any:
  # Check which pretrained architecture was requested.
  if (architecture == "TabPFN"):
    # Raise a clear, actionable error if the package isn't installed.
    if (not TABPFN_AVAILABLE):
      raise ImportError(
        "TabPFN is not installed. Run `pip install tabpfn`. Its checkpoint is also gated on "
        "Hugging Face - before first use, accept the license at "
        "https://huggingface.co/Prior-Labs/tabpfn_3 and either run `hf auth login` or set the "
        "HF_TOKEN environment variable (the checkpoint downloads once and is cached after that).")
    # Build the TabPFN classifier ("auto" device picks a GPU if one is available, else CPU).
    return TabPFNClassifier(
      n_estimators=hyperparams.get("nEstimators", 4), softmax_temperature=hyperparams.get("softmaxTemperature", 0.9),
      device="auto", random_state=randomState, ignore_pretraining_limits=True)
  elif (architecture == "TabICL"):
    # Raise a clear, actionable error if the package isn't installed.
    if (not TABICL_AVAILABLE):
      raise ImportError("TabICL is not installed. Run `pip install tabicl`.")
    # Build the TabICL classifier (device=None lets it auto-detect a GPU, else CPU).
    return TabICLClassifier(
      n_estimators=hyperparams.get("nEstimators", 8), softmax_temperature=hyperparams.get("softmaxTemperature", 0.9),
      random_state=randomState)
  elif (architecture == "SapRpt1Oss"):
    # Raise a clear, actionable error if the package isn't installed (this one is the least
    # portable of the three - see the README for its Python 3.11 and GPU/VRAM requirements).
    if (not SAP_RPT_AVAILABLE):
      raise ImportError(
        "SAP-RPT-1-OSS is not installed (or not importable in this Python environment - it "
        "specifically requires Python 3.11). Install with "
        "`pip install git+https://github.com/SAP-samples/sap-rpt-1-oss` inside a Python 3.11 "
        "environment. It also needs a CUDA GPU - see the README for VRAM guidance.")
    # Build the SAP-RPT-1-OSS classifier.
    return SAP_RPT_OSS_Classifier(
      max_context_size=hyperparams.get("maxContextSize", 8192), bagging=hyperparams.get("bagging", 8))
  else:
    # Raise an error for an unrecognized pretrained-architecture name.
    raise ValueError(f"Unsupported pretrained architecture: {architecture}. Choose from {PRETRAINED_ARCHITECTURES}.")


# Define the SubsampleForPretrainedContext function: caps how many rows a pretrained model's
# .fit() call is given, since these models condition on their training rows as "context" rather
# than learning from them via gradient descent - a huge context doesn't make them more accurate
# the way more epochs would for a from-scratch model, it just costs more memory/time, and each
# model has a documented regime it was actually validated in (see PRETRAINED_MAX_TRAIN_ROWS).
def SubsampleForPretrainedContext(
  xTrain: pandas.DataFrame, yTrain: pandas.Series, architecture: str, config: Dict[str, Any],
) -> Tuple[pandas.DataFrame, pandas.Series]:
  # Look up this architecture's row-count ceiling, allowing a config override.
  maxRows = config.get(f"{architecture}MaxTrainRows", PRETRAINED_MAX_TRAIN_ROWS.get(architecture))
  # Return the data unchanged if it's already within the ceiling.
  if (maxRows is None or len(xTrain) <= maxRows):
    return xTrain, yTrain
  # Print a message explaining the subsampling, since this is a deliberate design choice, not a bug.
  fprint(
    f"  {architecture}: training fold has {len(xTrain)} rows, above its {maxRows}-row practical "
    f"context limit - randomly subsampling down to {maxRows} rows for fit() (a pretrained "
    f"in-context model needs a representative sample to condition on, not every row).")
  # Randomly select maxRows row indices, reproducibly.
  rng = numpy.random.RandomState(config.get("RandomState", 42))
  sampledIdx = rng.choice(len(xTrain), size=maxRows, replace=False)
  # Return the subsampled feature matrix and target vector, with fresh contiguous indices.
  return (
    xTrain.iloc[sampledIdx].reset_index(drop=True),
    pandas.Series(numpy.asarray(yTrain)[sampledIdx]).reset_index(drop=True))


# Define the SuggestPretrainedHyperparameters function: the (small) Optuna search space for a
# pretrained architecture - these models have no learning rate/epochs/architecture-size to tune
# (the network itself is frozen), just a few inference-time knobs.
def SuggestPretrainedHyperparameters(trial: optuna.Trial, architecture: str, config: Dict[str, Any]) -> Dict[str, Any]:
  # Check which pretrained architecture's search space to build.
  if (architecture in ("TabPFN", "TabICL")):
    # Sample the ensemble size (more members = slightly more accurate and slower) and the
    # softmax temperature (lower = more confident/peaked probability outputs).
    return {
      "nEstimators": trial.suggest_int(
        "nEstimators", config.get("PretrainedNEstimatorsMin", 2), config.get("PretrainedNEstimatorsMax", 8)),
      "softmaxTemperature": trial.suggest_float(
        "softmaxTemperature", config.get("PretrainedSoftmaxTempMin", 0.5), config.get("PretrainedSoftmaxTempMax", 1.5)),
    }
  elif (architecture == "SapRpt1Oss"):
    # Sample the bagging count; max_context_size is left fixed (it's a hardware/memory ceiling,
    # not something to tune for accuracy).
    return {
      "bagging": trial.suggest_int("bagging", config.get("SapRptBaggingMin", 1), config.get("SapRptBaggingMax", 8)),
      "maxContextSize": config.get("SapRptMaxContextSize", 8192),
    }
  else:
    # Raise an error for an unrecognized pretrained-architecture name.
    raise ValueError(f"Unsupported pretrained architecture: {architecture}. Choose from {PRETRAINED_ARCHITECTURES}.")


# Define the TunePretrainedHyperparametersWithOptuna function (Step 8, pretrained-model version):
# unlike TuneDlHyperparametersWithOptuna, each trial is a single cheap .fit()+.predict() call (no
# epochs, no early stopping needed) since these models don't learn via gradient descent.
def TunePretrainedHyperparametersWithOptuna(
  xTrain: pandas.DataFrame, yTrain: pandas.Series, xVal: pandas.DataFrame, yVal: pandas.Series,
  architecture: str, config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  # Subsample the training context once up front (shared across every trial).
  xTrainCtx, yTrainCtx = SubsampleForPretrainedContext(xTrain, yTrain, architecture, config)

  # Define the Optuna objective function, maximizing validation macro-F1.
  def Objective(trial: optuna.Trial) -> float:
    # Sample this trial's hyperparameters and build the pretrained model with them.
    hyperparams = SuggestPretrainedHyperparameters(trial, architecture, config)
    model = BuildPretrainedModel(architecture, hyperparams, config.get("RandomState", 42))
    # Fit (store context) and evaluate on the validation fold.
    model.fit(xTrainCtx, yTrainCtx)
    preds = model.predict(xVal)
    return f1_score(yVal, preds, average="macro")

  # Create the Optuna study using the TPE sampler for reproducibility.
  study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.get("RandomState", 42)))
  # Run the optimization for the configured number of trials (pretrained trials are much cheaper
  # than from-scratch training trials, but default to the same trial count unless overridden).
  study.optimize(
    Objective, n_trials=config.get("PretrainedNOptunaTrials", config.get("NOptunaTrials", 20)),
    show_progress_bar=False)

  # The winning trial's raw suggested params are already exactly the hyperparams dict this
  # architecture needs (no derived/expanded values, unlike the from-scratch architectures).
  bestParams = dict(study.best_trial.params)
  if (architecture == "SapRpt1Oss"):
    # Re-attach the fixed (not tuned) max_context_size.
    bestParams["maxContextSize"] = config.get("SapRptMaxContextSize", 8192)
  # Build the report summarizing the tuning run.
  report = {
    "NTrials"       : len(study.trials),
    "BestValMacroF1": study.best_value,
    "BestParams"    : bestParams,
    "TrialValues"   : [t.value for t in study.trials],
  }
  # Return the best hyperparameters and the tuning report.
  return bestParams, report


# Define the IsScheduleFreeOptimizer helper, used to toggle train()/eval() mode on the optimizer itself.
def IsScheduleFreeOptimizer(optimizer: torch.optim.Optimizer) -> bool:
  # Check the optimizer's class name for the schedule-free marker.
  return "ScheduleFree" in type(optimizer).__name__


# Define the GetOptimizer function: dispatches an optimizer name to its instance,
# falling back to AdamW (with a warning) if an optional package isn't installed.
def GetOptimizer(optimizerName: str, modelParams, lr: float, weightDecay: float) -> torch.optim.Optimizer:
  # Check which optimizer was requested.
  if (optimizerName == "Adam"):
    return torch.optim.Adam(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "AdamW"):
    return torch.optim.AdamW(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "SGD"):
    return torch.optim.SGD(modelParams, lr=lr, momentum=0.9, weight_decay=weightDecay)
  elif (optimizerName == "RMSprop"):
    return torch.optim.RMSprop(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "RAdam"):
    return torch.optim.RAdam(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "Lion"):
    # Fall back to AdamW if the optional lion-pytorch package isn't installed.
    if (not LION_AVAILABLE):
      fprint("  Warning: lion-pytorch not installed; falling back to AdamW. `pip install lion-pytorch` to enable Lion.")
      return torch.optim.AdamW(modelParams, lr=lr, weight_decay=weightDecay)
    return Lion(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "Prodigy"):
    # Fall back to AdamW if the optional prodigyopt package isn't installed.
    if (not PRODIGY_AVAILABLE):
      fprint("  Warning: prodigyopt not installed; falling back to AdamW. `pip install prodigyopt` to enable Prodigy.")
      return torch.optim.AdamW(modelParams, lr=lr, weight_decay=weightDecay)
    # Prodigy is self-tuning and conventionally initialized with lr=1.0.
    return Prodigy(modelParams, lr=1.0, weight_decay=weightDecay)
  elif (optimizerName == "ScheduleFreeAdamW"):
    # Fall back to AdamW if the optional schedulefree package isn't installed.
    if (not SCHEDULEFREE_AVAILABLE):
      fprint("  Warning: schedulefree not installed; falling back to AdamW. `pip install schedulefree` to enable it.")
      return torch.optim.AdamW(modelParams, lr=lr, weight_decay=weightDecay)
    return AdamWScheduleFree(modelParams, lr=lr, weight_decay=weightDecay)
  elif (optimizerName == "Sophia"):
    return SophiaG(modelParams, lr=lr, weight_decay=weightDecay)
  else:
    # Raise an error for an unrecognized optimizer name.
    raise ValueError(f"Unsupported optimizer: {optimizerName}. Choose from {SUPPORTED_OPTIMIZERS}.")


# Define the GetLossFunction function: dispatches a loss name to its instance.
def GetLossFunction(lossName: str, labelSmoothing: float, focalGamma: float,
                     classWeightsTensor: Optional[torch.Tensor]) -> nn.Module:
  # Check which loss function was requested.
  if (lossName == "CrossEntropy"):
    return nn.CrossEntropyLoss(weight=classWeightsTensor)
  elif (lossName == "LabelSmoothing"):
    return nn.CrossEntropyLoss(weight=classWeightsTensor, label_smoothing=labelSmoothing)
  elif (lossName == "Focal"):
    return FocalLoss(gamma=focalGamma, weight=classWeightsTensor)
  else:
    # Raise an error for an unrecognized loss function name.
    raise ValueError(f"Unsupported loss function: {lossName}. Choose from {SUPPORTED_LOSS_FUNCTIONS}.")


# Define the GetScheduler function: dispatches a scheduler name to its instance.
def GetScheduler(schedulerName: str, optimizer: torch.optim.Optimizer, numEpochs: int):
  # Check which scheduler was requested.
  if (schedulerName == "None"):
    return None
  elif (schedulerName == "CosineWarmup"):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs, eta_min=1e-6)
  elif (schedulerName == "StepLR"):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, numEpochs // 3), gamma=0.1)
  elif (schedulerName == "ReduceLROnPlateau"):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
  else:
    # Raise an error for an unrecognized scheduler name.
    raise ValueError(f"Unsupported scheduler: {schedulerName}. Choose from {SUPPORTED_SCHEDULERS}.")


# Define the TrainEpoch function.
def TrainEpoch(model, dataLoader, criterion, optimizer, device, useAmp, scaler, emaModel) -> Tuple[float, float]:
  # Set the model to training mode.
  model.train()
  # Switch a schedule-free optimizer into its training-mode weight state.
  if (IsScheduleFreeOptimizer(optimizer)):
    optimizer.train()
  # Initialize the running loss and accuracy accumulators.
  runningLoss, runningCorrect, total = 0.0, 0, 0
  # Iterate through every batch.
  for xBatch, yBatch, _ in dataLoader:
    # A batch of exactly 1 sample crashes BatchNorm1d in training mode (it needs at least 2
    # samples to compute batch statistics). BuildTrainLoader() already avoids this for typical
    # dataset sizes via drop_last, but a dataset barely bigger than one batch can still produce
    # one - so skip it here too as a last-resort safety net rather than crashing the whole run.
    if (xBatch.size(0) < 2):
      continue
    # Move the batch to the training device.
    xBatch, yBatch = xBatch.to(device), yBatch.to(device)
    # Reset the gradients.
    optimizer.zero_grad()
    # Run the forward and backward pass, with or without automatic mixed precision.
    if (useAmp):
      # Use autocast for the forward pass (autocast itself works on CPU or CUDA).
      with torch.amp.autocast(device_type=device.type):
        outputs = model(xBatch)
        loss = criterion(outputs, yBatch)
      # The GradScaler is only meaningful (and only built) on CUDA - on CPU, autocast
      # alone is used and gradients are scaled/stepped the normal way.
      if (scaler is not None):
        # Scale the loss, backpropagate, and step the optimizer through the AMP scaler.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        # Plain backward/step under CPU autocast (no gradient scaling needed).
        loss.backward()
        optimizer.step()
    else:
      # Run the forward pass without AMP.
      outputs = model(xBatch)
      loss = criterion(outputs, yBatch)
      # Backpropagate and step the optimizer.
      loss.backward()
      optimizer.step()
    # Update the EMA shadow weights, if enabled.
    if (emaModel is not None):
      emaModel.update_parameters(model)
    # Accumulate the loss and accuracy statistics.
    runningLoss += loss.item() * xBatch.size(0)
    runningCorrect += (outputs.argmax(dim=1) == yBatch).sum().item()
    total += xBatch.size(0)
  # Return the epoch's average loss and accuracy.
  return runningLoss / total, runningCorrect / total


# Define the ValidateEpoch function.
def ValidateEpoch(model, dataLoader, criterion, device, emaModel, optimizer) -> Tuple[float, float]:
  # Use the EMA shadow model for evaluation if it is enabled, otherwise the live model.
  evalModel = emaModel if (emaModel is not None) else model
  # Set the evaluation model to evaluation mode.
  evalModel.eval()
  # Switch a schedule-free optimizer into its evaluation-mode (averaged) weight state.
  if (IsScheduleFreeOptimizer(optimizer)):
    optimizer.eval()
  # Initialize the running loss and accuracy accumulators.
  runningLoss, runningCorrect, total = 0.0, 0, 0
  # Disable gradient computation for evaluation.
  with torch.no_grad():
    # Iterate through every batch.
    for xBatch, yBatch, _ in dataLoader:
      # Move the batch to the evaluation device.
      xBatch, yBatch = xBatch.to(device), yBatch.to(device)
      # Run the forward pass.
      outputs = evalModel(xBatch)
      # Compute the loss.
      loss = criterion(outputs, yBatch)
      # Accumulate the loss and accuracy statistics.
      runningLoss += loss.item() * xBatch.size(0)
      runningCorrect += (outputs.argmax(dim=1) == yBatch).sum().item()
      total += xBatch.size(0)
  # Return the epoch's average loss and accuracy.
  return runningLoss / total, runningCorrect / total


# Define the RunTrainingLoop function: shared by both the Optuna objective and the final fit.
def RunTrainingLoop(
  model: nn.Module, trainLoader: DataLoader, valLoader: DataLoader, criterion: nn.Module,
  optimizerName: str, learningRate: float, weightDecay: float, schedulerName: str,
  numEpochs: int, patience: int, device: torch.device, useAmp: bool, useEma: bool,
  checkpointPath: Optional[Path] = None, showProgress: bool = False,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
  # Move the model to the training device.
  model = model.to(device)
  # Build the optimizer for this run.
  optimizer = GetOptimizer(optimizerName, model.parameters(), learningRate, weightDecay)
  # Build the learning-rate scheduler for this run.
  scheduler = GetScheduler(schedulerName, optimizer, numEpochs)
  # Build the AMP gradient scaler, if AMP is enabled and a CUDA device is available.
  scaler = torch.amp.GradScaler() if (useAmp and device.type == "cuda") else None
  # Build the EMA shadow model, if enabled.
  emaModel = AveragedModel(model).to(device) if (useEma) else None

  # Initialize the training history dictionary.
  history = {"TrainLoss": [], "ValLoss": [], "TrainAcc": [], "ValAcc": []}
  # Initialize the early-stopping trackers.
  bestValLoss, epochsNoImprove = float("inf"), 0
  # Initialize the in-memory best-weights snapshot (used when no checkpoint path is given).
  bestStateDict = None

  # Wrap the epoch range in a progress bar if requested (used for the final fit, not every Optuna trial).
  epochIterator = tqdm(range(numEpochs), desc="Epochs", leave=False) if (showProgress) else range(numEpochs)
  # Iterate through every training epoch.
  for epoch in epochIterator:
    # Run one training epoch.
    trainLoss, trainAcc = TrainEpoch(model, trainLoader, criterion, optimizer, device, useAmp, scaler, emaModel)
    # Run one validation epoch.
    valLoss, valAcc = ValidateEpoch(model, valLoader, criterion, device, emaModel, optimizer)
    # Step the learning-rate scheduler, if one is configured.
    if (scheduler is not None):
      # ReduceLROnPlateau needs the monitored metric passed in; the others don't.
      if (isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
        scheduler.step(valLoss)
      else:
        scheduler.step()
    # Record this epoch's metrics.
    history["TrainLoss"].append(trainLoss)
    history["ValLoss"].append(valLoss)
    history["TrainAcc"].append(trainAcc)
    history["ValAcc"].append(valAcc)

    # Check whether validation loss improved.
    if (valLoss < bestValLoss):
      # Update the best validation loss and reset the patience counter.
      bestValLoss, epochsNoImprove = valLoss, 0
      # Snapshot the model powering evaluation right now (the EMA shadow model, if enabled).
      snapshotModel = emaModel.module if (emaModel is not None) else model
      # Save the snapshot to disk if a checkpoint path was given, otherwise keep it in memory.
      if (checkpointPath is not None):
        torch.save(snapshotModel.state_dict(), checkpointPath)
      else:
        bestStateDict = {k: v.clone() for k, v in snapshotModel.state_dict().items()}
    else:
      # Increment the epochs-without-improvement counter.
      epochsNoImprove += 1

    # Stop early if validation loss hasn't improved for `patience` epochs in a row.
    if (epochsNoImprove >= patience):
      break

  # Reload the best-checkpoint weights before returning.
  if (checkpointPath is not None and checkpointPath.exists()):
    model.load_state_dict(torch.load(checkpointPath, map_location=device))
  elif (bestStateDict is not None):
    model.load_state_dict(bestStateDict)

  # Return the best model, its training history, and the best validation loss achieved.
  return model, history, bestValLoss


# Define the SuggestDlHyperparameters function: the Optuna search space for a given architecture.
def SuggestDlHyperparameters(trial: optuna.Trial, architecture: str, config: Dict[str, Any]) -> Dict[str, Any]:
  # Sample the learning rate and weight decay, shared by every architecture.
  params = {
    "learningRate": trial.suggest_float("learningRate", 1e-5, 1e-2, log=True),
    "weightDecay" : trial.suggest_float("weightDecay", 1e-6, 1e-2, log=True),
  }
  # Check which architecture-specific search space to add.
  if (architecture == "MLP"):
    # Sample the number of hidden layers and their shared width.
    numLayers = trial.suggest_int("mlpNumLayers", config.get("MlpNumLayersMin", 1), config.get("MlpNumLayersMax", 4))
    hiddenDim = trial.suggest_int(
      "mlpHiddenDim", config.get("MlpHiddenDimMin", 32), config.get("MlpHiddenDimMax", 256), step=32)
    # Store the resulting list of hidden layer sizes.
    params["hiddenDims"] = [hiddenDim] * numLayers
    # Sample the dropout rate.
    params["dropout"] = trial.suggest_float(
      "dropout", config.get("MlpDropoutMin", 0.0), config.get("MlpDropoutMax", 0.5))
  elif (architecture == "TabTransformer"):
    # Sample the embedding dimension and number of attention heads (kept mutually divisible).
    params["embedDim"] = trial.suggest_categorical("embedDim", config.get("TransformerEmbedDimChoices", [16, 32, 64]))
    params["numHeads"] = trial.suggest_categorical("numHeads", config.get("TransformerNumHeadsChoices", [2, 4]))
    # Sample the number of Transformer blocks.
    params["numLayers"] = trial.suggest_int(
      "transformerNumLayers", config.get("TransformerNumLayersMin", 1), config.get("TransformerNumLayersMax", 4))
    # Sample the dropout rate.
    params["dropout"] = trial.suggest_float(
      "dropout", config.get("TransformerDropoutMin", 0.0), config.get("TransformerDropoutMax", 0.5))
  else:
    # Raise an error for an unrecognized architecture name.
    raise ValueError(f"Unsupported architecture: {architecture}. Choose from {SUPPORTED_ARCHITECTURES}.")
  # Return the sampled hyperparameters.
  return params


# Define the BuildTrainLoader function: builds the *training* DataLoader specifically.
# MlpClassifier uses BatchNorm1d, which needs at least 2 samples in a batch to compute batch
# statistics - and PyTorch's default DataLoader behavior hands out a final, smaller "leftover"
# batch every epoch whenever the dataset size isn't a multiple of the batch size. If that
# leftover happens to be exactly 1 sample, training crashes with "Expected more than 1 value
# per channel". This only matters for the *training* loader (the one actually run in .train()
# mode) - Val/Test loaders run in .eval() mode, which uses running statistics instead of
# batch statistics, so a batch of 1 there is harmless.
def BuildTrainLoader(dataset: Dataset, batchSize: int) -> DataLoader:
  # Compute how many samples would land in the final, leftover batch of every epoch.
  leftover = len(dataset) % batchSize
  # Only drop that leftover batch if it would be exactly 1 sample (the crashing case), and only
  # if the dataset is big enough that dropping it still leaves at least one full batch to train
  # on - never drop_last on a dataset so small it would zero out every epoch's training data.
  dropLast = (leftover == 1) and (len(dataset) >= 2 * batchSize)
  # Build and return the training DataLoader with the computed drop_last setting.
  return DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=dropLast)


# Define the TuneHyperparametersWithOptuna function (Step 8 of the pipeline, deep-learning version).
def TuneDlHyperparametersWithOptuna(
  trainDataset: TabularDataset, valDataset: TabularDataset, numFeatures: int, numClasses: int,
  architecture: str, optimizerName: str, lossFunction: str, schedulerName: str,
  classWeightsTensor: Optional[torch.Tensor], device: torch.device, config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  # Build the data loaders used by every trial.
  batchSize = config.get("BatchSize", 64)
  trainLoader = BuildTrainLoader(trainDataset, batchSize)
  valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)

  # Define the Optuna objective function, maximizing validation macro-F1.
  def Objective(trial: optuna.Trial) -> float:
    # Sample this trial's hyperparameters.
    hyperparams = SuggestDlHyperparameters(trial, architecture, config)
    # Build the model, loss, and run a short training loop with its own early stopping.
    model = BuildModel(architecture, numFeatures, numClasses, hyperparams)
    criterion = GetLossFunction(
      lossFunction, config.get("LabelSmoothing", 0.1), config.get("FocalGamma", 2.0), classWeightsTensor)
    model, _, _ = RunTrainingLoop(
      model, trainLoader, valLoader, criterion, optimizerName, hyperparams["learningRate"],
      hyperparams["weightDecay"], schedulerName,
      numEpochs=config.get("OptunaMaxEpochsPerTrial", 20), patience=config.get("OptunaPatience", 5),
      device=device, useAmp=config.get("UseAmp", False), useEma=False,
    )
    # Evaluate the trial's model on the validation set and score it by macro-F1.
    model.eval()
    allPreds, allTargets = [], []
    with torch.no_grad():
      for xBatch, yBatch, _ in valLoader:
        outputs = model(xBatch.to(device))
        allPreds.append(outputs.argmax(dim=1).cpu().numpy())
        allTargets.append(yBatch.numpy())
    return f1_score(numpy.concatenate(allTargets), numpy.concatenate(allPreds), average="macro")

  # Create the Optuna study using the TPE sampler for reproducibility.
  study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.get("RandomState", 42)))
  # Run the optimization for the configured number of trials.
  study.optimize(Objective, n_trials=config.get("NOptunaTrials", 20), show_progress_bar=False)

  # Re-expand the winning trial's raw params into the same structured hyperparams dict SuggestDlHyperparameters builds.
  bestTrial = study.best_trial
  bestParams = {"learningRate": bestTrial.params["learningRate"], "weightDecay": bestTrial.params["weightDecay"]}
  if (architecture == "MLP"):
    bestParams["hiddenDims"] = [bestTrial.params["mlpHiddenDim"]] * bestTrial.params["mlpNumLayers"]
    bestParams["dropout"] = bestTrial.params["dropout"]
  else:
    bestParams["embedDim"] = bestTrial.params["embedDim"]
    bestParams["numHeads"] = bestTrial.params["numHeads"]
    bestParams["numLayers"] = bestTrial.params["transformerNumLayers"]
    bestParams["dropout"] = bestTrial.params["dropout"]

  # Build the report summarizing the tuning run.
  report = {
    "NTrials"       : len(study.trials),
    "BestValMacroF1": study.best_value,
    "BestParams"    : bestParams,
    "TrialValues"   : [t.value for t in study.trials],
  }
  # Return the best hyperparameters and the tuning report.
  return bestParams, report


# Define the PlotTrainingHistory function.
def PlotTrainingHistory(history: Dict[str, List[float]], outputDir: Union[str, Path]) -> None:
  # Create a figure with two subplots for loss and accuracy curves.
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  # Plot the training and validation loss curves.
  axes[0].plot(history["TrainLoss"], label="Train Loss", linewidth=2)
  axes[0].plot(history["ValLoss"], label="Val Loss", linewidth=2)
  axes[0].set_xlabel("Epoch", fontsize=12)
  axes[0].set_ylabel("Loss", fontsize=12)
  axes[0].set_title("Loss Over Training Epochs", fontsize=14, fontweight="bold")
  axes[0].legend(fontsize=10)
  axes[0].grid(True, alpha=0.3)
  # Plot the training and validation accuracy curves.
  axes[1].plot(history["TrainAcc"], label="Train Acc", linewidth=2)
  axes[1].plot(history["ValAcc"], label="Val Acc", linewidth=2)
  axes[1].set_xlabel("Epoch", fontsize=12)
  axes[1].set_ylabel("Accuracy", fontsize=12)
  axes[1].set_title("Accuracy Over Training Epochs", fontsize=14, fontweight="bold")
  axes[1].legend(fontsize=10)
  axes[1].grid(True, alpha=0.3)
  # Adjust the layout and save the figure.
  plt.tight_layout()
  plt.savefig(Path(outputDir) / "TrainingHistory.png", dpi=200, bbox_inches="tight")
  plt.close()
  # Print a confirmation message.
  fprint(f"Training history plot saved to {Path(outputDir) / 'TrainingHistory.png'}")


# Define the EvaluateModel function (deep-learning version): runs inference, then reuses the
# same CalculatePerformanceMetrics / PlotConfusionMatrix machinery as the classical-ML pipeline
# so the two pipelines' output files are directly comparable.
def EvaluateModel(
  model: nn.Module, dataLoader: DataLoader, device: torch.device, classNames: List[str],
  outputDir: Union[str, Path], prefix: str = "Test",
) -> Dict[str, Any]:
  # Print the evaluation start message.
  fprint(f"Starting evaluation: prefix={prefix} | outputDir={outputDir}")

  # Set the model to evaluation mode and run inference over every batch.
  model.eval()
  allPreds, allTargets, allProbs, allRowIds = [], [], [], []
  with torch.no_grad():
    for xBatch, yBatch, rowIdBatch in dataLoader:
      outputs = model(xBatch.to(device))
      probs = torch.softmax(outputs, dim=1).cpu().numpy()
      allPreds.append(probs.argmax(axis=1))
      allProbs.append(probs)
      allTargets.append(yBatch.numpy())
      allRowIds.extend(rowIdBatch.tolist() if torch.is_tensor(rowIdBatch) else list(rowIdBatch))
  # Concatenate every batch's results into single arrays, then hand off to the shared
  # metrics/plots/file-writing logic (identical regardless of what produced the predictions).
  preds = numpy.concatenate(allPreds)
  predProbs = numpy.concatenate(allProbs)
  targets = numpy.concatenate(allTargets)
  return EvaluateFromPredictions(preds, predProbs, targets, allRowIds, classNames, outputDir, prefix)


# Define the EvaluatePretrainedModel function: the same evaluation as EvaluateModel, but for a
# pretrained in-context model (TabPFN/TabICL/SAP-RPT-1-OSS) that exposes a plain sklearn-style
# .predict()/.predict_proba() API on a full feature matrix directly - no DataLoader, no batches,
# no .eval()/.train() mode switching needed.
def EvaluatePretrainedModel(
  model: Any, xData: pandas.DataFrame, yData: pandas.Series, rowIds: pandas.Series,
  classNames: List[str], outputDir: Union[str, Path], prefix: str = "Test",
) -> Dict[str, Any]:
  # Print the evaluation start message.
  fprint(f"Starting evaluation: prefix={prefix} | outputDir={outputDir}")
  # Run inference directly on the full feature matrix (these models take one call, not batches).
  preds = numpy.asarray(model.predict(xData))
  predProbs = numpy.asarray(model.predict_proba(xData))
  targets = numpy.asarray(yData)
  # Hand off to the shared metrics/plots/file-writing logic.
  return EvaluateFromPredictions(preds, predProbs, targets, list(rowIds), classNames, outputDir, prefix)


# Define the EvaluateFromPredictions function: everything after "we have predictions" is
# identical regardless of what produced them (a torch model's batched inference, or a pretrained
# model's single .predict() call) - confusion matrix, extended metrics, plots, and the same
# {Split}CM.png/.csv, {Split}ClassificationReport.txt, {Split}EvaluationMetrics.json, and
# {Split}DetailedPredictions.csv files every architecture in this pipeline produces.
def EvaluateFromPredictions(
  preds: numpy.ndarray, predProbs: numpy.ndarray, targets: numpy.ndarray, rowIds: List[Any],
  classNames: List[str], outputDir: Union[str, Path], prefix: str = "Test",
) -> Dict[str, Any]:
  # Create a folder for this split's evaluation results.
  evalOutputDir = Path(outputDir) / prefix
  evalOutputDir.mkdir(parents=True, exist_ok=True)
  # Keep the parameter name used throughout the rest of this function.
  allRowIds = rowIds

  # Build the sorted list of label ids present in this split, and the confusion matrix.
  labelsSorted = sorted(numpy.unique(targets).tolist())
  confMatrix = confusion_matrix(targets, preds, labels=labelsSorted)
  targetNames = [classNames[i] for i in labelsSorted]
  # Calculate the extended performance metrics from the confusion matrix (shared with the ML pipeline).
  pmMetrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True, addPerClass=True)
  # Generate the detailed sklearn classification report.
  classReport = classification_report(targets, preds, labels=labelsSorted, target_names=targetNames, zero_division=0)

  # Compute the standard summary metrics used for cross-model comparison.
  summaryMetrics = {
    "Accuracy"         : float(accuracy_score(targets, preds)),
    "PrecisionMacro"   : float(precision_score(targets, preds, average="macro", zero_division=0)),
    "RecallMacro"      : float(recall_score(targets, preds, average="macro", zero_division=0)),
    "F1Macro"          : float(f1_score(targets, preds, average="macro", zero_division=0)),
    "PrecisionWeighted": float(precision_score(targets, preds, average="weighted", zero_division=0)),
    "RecallWeighted"   : float(recall_score(targets, preds, average="weighted", zero_division=0)),
    "F1Weighted"       : float(f1_score(targets, preds, average="weighted", zero_division=0)),
    "LogLoss"          : float(log_loss(targets, predProbs, labels=labelsSorted)),
  }
  # Attempt to compute the ROC-AUC score (binary vs. multiclass one-vs-rest).
  try:
    if (len(labelsSorted) == 2):
      summaryMetrics["RocAuc"] = float(roc_auc_score(targets, predProbs[:, 1]))
    else:
      summaryMetrics["RocAucOvrMacro"] = float(roc_auc_score(targets, predProbs, multi_class="ovr", average="macro"))
  except Exception as e:
    summaryMetrics["RocAucError"] = str(e)

  # Save the confusion matrix plot (reusing the classical-ML pipeline's plotting function).
  PlotConfusionMatrix(confMatrix, classNames=targetNames, outputDir=evalOutputDir, fileName=f"{prefix}CM.png")

  # Save the confusion matrix to a CSV file with class names as headers and row labels.
  with open(Path(evalOutputDir) / f"{prefix}CM.csv", "w", newline="") as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow([""] + targetNames)
    for i, className in enumerate(targetNames):
      csvWriter.writerow([className] + confMatrix[i].tolist())

  # Save the classification report to a text file for reference.
  with open(Path(evalOutputDir) / f"{prefix}ClassificationReport.txt", "w") as f:
    f.write(classReport)

  # Compile all metrics into a single structured dictionary for return and serialization.
  metrics = {
    **summaryMetrics, **pmMetrics,
    "ConfusionMatrix"      : confMatrix.tolist(),
    "ConfusionMatrixLabels": targetNames,
    "ClassificationReport" : classReport,
  }
  # Store the metrics in a JSON file for downstream use.
  with open(Path(evalOutputDir) / f"{prefix}EvaluationMetrics.json", "w") as jsonFile:
    json.dump(metrics, jsonFile, indent=2, cls=NumpyEncoder)

  # Store detailed per-row predictions in a CSV file for downstream analysis.
  with open(Path(evalOutputDir) / f"{prefix}DetailedPredictions.csv", "w", newline="") as csvFile:
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["Split", "RowId", "Actual", "Pred", "PredProb"])
    for rowId, actual, pred, predProb in zip(allRowIds, targets, preds, predProbs):
      csvWriter.writerow([prefix, rowId, int(actual), int(pred), predProb.tolist()])

  # Print a one-line summary for quick console feedback.
  fprint(f"  {prefix}: accuracy={summaryMetrics['Accuracy']:.4f} f1Macro={summaryMetrics['F1Macro']:.4f}")
  # Return the metrics dictionary.
  return metrics


# Define the RunCompletePipeline function (deep-learning version).
def RunCompletePipeline(
  datasetCfg: Dict[str, Any], architecture: str, optimizerName: str, lossFunction: str,
  scalerName: str, imbalanceMethod: str, outlierMethod: str, outputDir: str, config: Dict[str, Any],
) -> Dict[str, Any]:
  # Record the start time for the runtime report.
  startTime = time.time()
  # Print the pipeline initialization message.
  fprint(f"\n{'=' * 70}")
  fprint(f"Starting Experiment: Dataset={datasetCfg['Name']} | Architecture={architecture} | "
         f"Optimizer={optimizerName} | Loss={lossFunction} | Scaler={scalerName} | "
         f"Imbalance={imbalanceMethod} | Outlier={outlierMethod}")
  fprint(f"Output Directory: {outputDir}")
  fprint(f"{'=' * 70}")
  # Create the output directory structure.
  Path(outputDir).mkdir(parents=True, exist_ok=True)

  # Determine the compute device.
  devicePref = config.get("Device", "auto")
  device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") if (devicePref == "auto") \
    else torch.device(devicePref)

  # ---- Steps 1-7: identical to the classical-ML pipeline, imported directly from it ----
  fprint("Building dataset...")
  df, targetCol, extraDrop, catCols = ResolveDataset(datasetCfg)
  fprint(f"Raw dataset shape: {df.shape}")

  df = df.dropna(subset=[targetCol]).reset_index(drop=True)
  labelEncoder = LabelEncoder()
  df[targetCol] = labelEncoder.fit_transform(df[targetCol].astype(str))
  df["RowId"] = numpy.arange(len(df))

  rawTargetDistribution = {str(k): int(v) for k, v in df[targetCol].value_counts().to_dict().items()}

  fprint("Step 1: DropUnnecessaryColumns")
  df, unnecessaryReport = DropUnnecessaryColumns(df, targetCol, explicitDropCols=extraDrop)

  fprint(f"Step 2: RemoveOutliers ({outlierMethod})")
  df, outlierReport = RemoveOutliers(
    df, targetCol, method=outlierMethod,
    iqrMultiplier=config.get("OutlierIqrMultiplier", 3.0),
    zScoreThreshold=config.get("OutlierZScoreThreshold", 3.0),
    contamination=config.get("OutlierContamination", 0.05),
    maxRowDropFrac=config.get("OutlierMaxRowDropFrac", 0.15),
    randomState=config.get("RandomState", 42),
  )

  fprint("Step 3: SplitData")
  trainDf, valDf, testDf, splitReport = SplitData(
    df, targetCol, testSize=config.get("TestSize", 0.10), valSize=config.get("ValSize", 0.10),
    randomState=config.get("RandomState", 42),
  )

  fprint("Step 4: DropRedundantColumns")
  trainDf, valDf, testDf, redundantReport = DropRedundantColumns(
    trainDf, valDf, testDf, targetCol, correlationThreshold=config.get("CorrelationThreshold", 0.95))

  trainDf, valDf, testDf, newCatCols = EncodeCategoricalColumns(trainDf, valDf, testDf, catCols)

  featureCols = [
    c for c in trainDf.columns
    if (c not in (targetCol, "RowId") and pandas.api.types.is_numeric_dtype(trainDf[c]))
  ]
  # Fill any remaining missing feature values with the *training-fold* median, applied
  # identically to Val/Test - computing each split's median from its own rows (the previous
  # behavior here) leaks that split's own distribution into its imputed values, which a model
  # would never have access to at real prediction time. Only the training fold's statistics
  # may be used to fill any split, exactly like NormalizeFeatures already does for scaling.
  trainMedians = trainDf[featureCols].median().fillna(0)
  for splitDf in (trainDf, valDf, testDf):
    splitDf[featureCols] = splitDf[featureCols].fillna(trainMedians)

  xTrain, yTrain = trainDf[featureCols], trainDf[targetCol]
  xVal, yVal = valDf[featureCols], valDf[targetCol]
  xTest, yTest = testDf[featureCols], testDf[targetCol]
  rowIdsTrain, rowIdsVal, rowIdsTest = trainDf["RowId"], valDf["RowId"], testDf["RowId"]

  fprint(f"Step 5: HandleClassImbalance ({imbalanceMethod})")
  xTrainBal, yTrainBal, classWeights, imbalanceReport = HandleClassImbalance(
    xTrain, yTrain, imbalanceMethod, randomState=config.get("RandomState", 42))
  xTrainBal = pandas.DataFrame(xTrainBal, columns=featureCols).reset_index(drop=True)
  yTrainBal = pandas.Series(yTrainBal).reset_index(drop=True)
  # A resampler invalidates the original row ids (it can duplicate/remove rows), so
  # the resampled training fold gets fresh synthetic ids purely for bookkeeping.
  rowIdsTrainBal = pandas.Series(numpy.arange(len(yTrainBal)))

  fprint(f"Step 6: NormalizeFeatures ({scalerName})")
  xTrainScaled, xValScaled, xTestScaled, scaler = NormalizeFeatures(
    xTrainBal, xVal, xTest, scalerName, randomState=config.get("RandomState", 42))

  fprint("Step 7: SelectFeatures")
  selectedFeatures, selectionReport = SelectFeatures(
    xTrainScaled, yTrainBal,
    topKFraction=config.get("FeatureSelectionTopKFrac", 0.6),
    minFeaturesSelected=config.get("MinFeaturesSelected", 4),
    randomState=config.get("RandomState", 42),
  )
  xTrainFs = xTrainScaled[selectedFeatures]
  xValFs = xValScaled[selectedFeatures]
  xTestFs = xTestScaled[selectedFeatures]

  # ---- Steps 8-9: either a from-scratch PyTorch model, or a pretrained in-context model ----
  nClasses = int(pandas.concat([pandas.Series(yTrainBal), pandas.Series(yVal), pandas.Series(yTest)]).nunique())
  classNames = [str(c) for c in labelEncoder.classes_]

  # Save the class distribution plot across the three splits (reusing the classical-ML pipeline's
  # plotter) - identical regardless of which kind of architecture is used below.
  PlotClassDistribution({"Train": yTrainBal, "Val": yVal, "Test": yTest}, classNames, outputDir)

  if (architecture in PRETRAINED_ARCHITECTURES):
    # ---- Pretrained in-context model path (TabPFN / TabICL / SapRpt1Oss) ----
    # "ClassWeight" has no effect here - none of these models' .fit() accepts sample weights
    # (they're pretrained, not being fit via a weighted loss) - warn rather than silently ignore.
    if (imbalanceMethod == "ClassWeight"):
      fprint(
        f"  Warning: ImbalanceMethod=ClassWeight has no effect with {architecture} - this "
        f"pretrained model's .fit() doesn't accept sample weights. The training data is used "
        f"as-is (equivalent to ImbalanceMethod=None for this architecture).")

    # Step 8: hyperparameter tuning via Optuna (cheap - each trial is one fit+predict, no epochs).
    fprint(f"Step 8: TunePretrainedHyperparametersWithOptuna "
           f"({architecture}, {config.get('PretrainedNOptunaTrials', config.get('NOptunaTrials', 20))} trials)")
    bestParams, tuningReport = TunePretrainedHyperparametersWithOptuna(
      xTrainFs, yTrainBal, xValFs, yVal, architecture, config)
    from TabularMLPipeline import PlotOptunaHistory
    PlotOptunaHistory(tuningReport["TrialValues"], outputDir)

    # Step 9: fit the final model once on the (subsampled, if needed) training context and evaluate.
    # Val is NOT folded into this final fit either - it's kept fully held out for the
    # Step9ValMetrics report, on equal footing with Test, exactly like the from-scratch path.
    fprint("Step 9: FitFinalPretrainedModel + EvaluateModel")
    xTrainCtx, yTrainCtxBal = SubsampleForPretrainedContext(xTrainFs, yTrainBal, architecture, config)
    finalModel = BuildPretrainedModel(architecture, bestParams, config.get("RandomState", 42))
    fitStartTime = time.time()
    finalModel.fit(xTrainCtx, yTrainCtxBal)
    fprint(f"  {architecture} fit() (context of {len(xTrainCtx)} rows) took {time.time() - fitStartTime:.2f}s")

    # Evaluate the final model on every split (rowIdsTrainBal covers the FULL balanced training
    # fold, not just the subsampled context used for fitting - Train metrics still reflect
    # performance across all of it).
    testMetrics = EvaluatePretrainedModel(finalModel, xTestFs, yTest, rowIdsTest, classNames, outputDir, prefix="Test")
    valMetrics = EvaluatePretrainedModel(finalModel, xValFs, yVal, rowIdsVal, classNames, outputDir, prefix="Val")
    trainMetrics = EvaluatePretrainedModel(
      finalModel, xTrainFs, yTrainBal, rowIdsTrainBal, classNames, outputDir, prefix="Train")

    # There's no epoch-by-epoch training history for a pretrained model, and no dedicated
    # checkpoint file - the "weights" are the shared, already-downloaded pretrained checkpoint,
    # not something unique to this run, so only the lightweight metadata needed to refit is saved.
    history, checkpointPath = None, None

  else:
    # ---- From-scratch PyTorch model path (MLP / TabTransformer) ----
    # Build the class-weight tensor for the loss function, if the ClassWeight method was chosen.
    classWeightsTensor = None
    if (classWeights is not None):
      classWeightsTensor = torch.tensor(
        [classWeights.get(i, 1.0) for i in range(nClasses)], dtype=torch.float32, device=device)

    # Wrap every split in a TabularDataset.
    trainDataset = TabularDataset(xTrainFs, yTrainBal, rowIdsTrainBal)
    valDataset = TabularDataset(xValFs, yVal, rowIdsVal)
    testDataset = TabularDataset(xTestFs, yTest, rowIdsTest)

    # Step 8: hyperparameter tuning via Optuna.
    fprint(f"Step 8: TuneDlHyperparametersWithOptuna ({architecture}, {config.get('NOptunaTrials', 20)} trials)")
    bestParams, tuningReport = TuneDlHyperparametersWithOptuna(
      trainDataset, valDataset, len(selectedFeatures), nClasses, architecture, optimizerName, lossFunction,
      config.get("SchedulerName", "CosineWarmup"), classWeightsTensor, device, config,
    )
    from TabularMLPipeline import PlotOptunaHistory
    PlotOptunaHistory(tuningReport["TrialValues"], outputDir)

    # Step 9: train the final model (train fold only, early-stopped on the validation fold) and evaluate.
    # Unlike the classical-ML pipeline, train+val are NOT merged here: a held-out validation
    # fold is what tells a neural net when to stop, so folding it into training would remove
    # the only signal early stopping has to work with.
    fprint("Step 9: TrainFinalModel + EvaluateModel")
    batchSize = config.get("BatchSize", 64)
    trainLoader = BuildTrainLoader(trainDataset, batchSize)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    finalModel = BuildModel(architecture, len(selectedFeatures), nClasses, bestParams)
    criterion = GetLossFunction(
      lossFunction, config.get("LabelSmoothing", 0.1), config.get("FocalGamma", 2.0), classWeightsTensor)
    checkpointPath = Path(outputDir) / "BestModel.pt"
    finalModel, history, bestValLoss = RunTrainingLoop(
      finalModel, trainLoader, valLoader, criterion, optimizerName, bestParams["learningRate"],
      bestParams["weightDecay"], config.get("SchedulerName", "CosineWarmup"),
      numEpochs=config.get("NumEpochs", 100), patience=config.get("Patience", 15),
      device=device, useAmp=config.get("UseAmp", False), useEma=config.get("UseEma", False),
      checkpointPath=checkpointPath, showProgress=True,
    )
    PlotTrainingHistory(history, outputDir)

    # Evaluate the final model on every split.
    testMetrics = EvaluateModel(finalModel, testLoader, device, classNames, outputDir, prefix="Test")
    valMetrics = EvaluateModel(finalModel, valLoader, device, classNames, outputDir, prefix="Val")
    trainMetrics = EvaluateModel(finalModel, trainLoader, device, classNames, outputDir, prefix="Train")

  # Persist the scaler and label encoder alongside the model's hyperparameters for later inference
  # (for a pretrained architecture, "reloading" this model means re-fitting it on the training
  # data with these exact BestParams, which is cheap - there are no unique-to-this-run weights to
  # save, unlike BestModel.pt for the from-scratch architectures above).
  joblib.dump(
    {"Features": selectedFeatures, "LabelEncoder": labelEncoder, "Scaler": scaler,
     "Architecture": architecture, "BestParams": bestParams},
    Path(outputDir) / "ModelMetadata.joblib",
  )

  # Assemble the full experiment result dictionary covering every pipeline step.
  result = {
    "Dataset"                  : datasetCfg["Name"],
    "Architecture"             : architecture,
    "OptimizerName"            : optimizerName,
    "LossFunction"             : lossFunction,
    "ScalerName"               : scalerName,
    "ImbalanceMethod"          : imbalanceMethod,
    "OutlierMethod"            : outlierMethod,
    "TargetColumn"             : targetCol,
    "RawRows"                  : len(df),
    "RawTargetDistribution"    : rawTargetDistribution,
    "ClassNames"               : classNames,
    "Step1UnnecessaryColumns"  : unnecessaryReport,
    "Step2Outliers"            : outlierReport,
    "Step3Split"               : splitReport,
    "Step4RedundantColumns"    : redundantReport,
    "Step5ClassImbalance"      : imbalanceReport,
    "Step7FeatureSelection"    : selectionReport,
    "Step8HyperparameterTuning": tuningReport,
    "Step9TestMetrics"         : testMetrics,
    "Step9ValMetrics"          : valMetrics,
    "Step9TrainMetrics"        : trainMetrics,
    "TrainingHistory"          : history,
    "ModelCheckpointPath"      : str(checkpointPath) if (checkpointPath is not None) else None,
    "RuntimeSeconds"           : round(time.time() - startTime, 2),
  }
  # Save the full experiment result to a single JSON file for downstream aggregation.
  with open(Path(outputDir) / "FullExperimentResult.json", "w") as f:
    json.dump(result, f, indent=2, cls=NumpyEncoder)

  # Print the pipeline completion summary for user confirmation.
  fprint(
    f"Pipeline Complete | {datasetCfg['Name']}/{architecture}/{optimizerName}/{lossFunction} "
    f"| Test Accuracy={testMetrics['Accuracy']:.4f} | Test F1Macro={testMetrics['F1Macro']:.4f} "
    f"| {result['RuntimeSeconds']}s | Results saved to {outputDir}"
  )
  # Return the full result dictionary.
  return result


# Define the ValidateGridDimension function: catches a common YAML mistake early, with a
# clear error message, instead of letting a bad config value fail deep inside an Optuna trial.
def ValidateGridDimension(dimensionName: str, values: List[str], supportedValues: List[str]) -> None:
  # Check every value configured for this grid dimension.
  for value in values:
    # Values already on the supported list need no further checking.
    if (value in supportedValues):
      continue
    # Detect the specific mistake of listing two options on one YAML line joined by " - "
    # (e.g. "- MLP - TabTransformer") instead of as two separate "- " list items, and
    # name it explicitly since the generic "unsupported value" message hides the real cause.
    dashSplitParts = [p.strip() for p in value.split(" - ")]
    if (len(dashSplitParts) > 1 and all(p in supportedValues for p in dashSplitParts)):
      raise ValueError(
        f"Config error in '{dimensionName}': got a single value {value!r}, which looks like "
        f"{len(dashSplitParts)} options ({', '.join(dashSplitParts)}) written on one YAML line. "
        f"Each option needs its own '- ' list item, e.g.:\n"
        f"{dimensionName}:\n" + "\n".join(f"  - {p}" for p in dashSplitParts))
    # Otherwise, raise the generic unsupported-value error.
    raise ValueError(
      f"Config error in '{dimensionName}': {value!r} is not one of the supported options: {supportedValues}.")


# Define the main execution block.
if (__name__ == "__main__"):
  # Import the argparse module for command-line interface.
  import argparse

  # Initialize the argument parser for command-line interface.
  parser = argparse.ArgumentParser(description="Deep Learning Tabular Pipeline (PyTorch MLP / Tabular Transformer)")
  # Add the command-line argument for the configuration file.
  parser.add_argument("--config", type=str, default="config_dl.yaml", help="Path to the YAML or JSON configuration file")
  # Parse the command-line arguments into a namespace object.
  args = parser.parse_args()
  # Load the configuration from the file.
  config = LoadConfig(args.config)

  # Extract the base output directory for the experiment grid.
  baseOutputDir = str(Path(config.get("OutputDir", "ResultsDL")))

  # Extract the datasets list and grid dimensions from the configuration.
  datasets = config.get("Datasets", [])
  architectures = config.get("Architecture", ["MLP"])
  optimizerNames = config.get("OptimizerName", ["AdamW"])
  lossFunctions = config.get("LossFunction", ["CrossEntropy"])
  scalers = config.get("Scalers", ["Standard"])
  imbalanceMethods = config.get("ImbalanceMethod", ["SMOTE"])
  outlierMethods = config.get("OutlierMethod", ["IQR"])

  # Ensure every grid dimension is a list, for uniform iteration (a single string in the
  # config, e.g. "Architecture: MLP" instead of "Architecture: [MLP]", still works).
  if (isinstance(architectures, str)):
    architectures = [architectures]
  if (isinstance(optimizerNames, str)):
    optimizerNames = [optimizerNames]
  if (isinstance(lossFunctions, str)):
    lossFunctions = [lossFunctions]
  if (isinstance(scalers, str)):
    scalers = [scalers]
  if (isinstance(imbalanceMethods, str)):
    imbalanceMethods = [imbalanceMethods]
  if (isinstance(outlierMethods, str)):
    outlierMethods = [outlierMethods]

  # Validate every grid dimension up front, so a config typo fails fast with a clear
  # message instead of surfacing deep inside an Optuna trial's traceback.
  ValidateGridDimension("Architecture", architectures, SUPPORTED_ARCHITECTURES)
  ValidateGridDimension("OptimizerName", optimizerNames, SUPPORTED_OPTIMIZERS)
  ValidateGridDimension("LossFunction", lossFunctions, SUPPORTED_LOSS_FUNCTIONS)
  ValidateGridDimension("Scalers", scalers, SUPPORTED_SCALERS)
  ValidateGridDimension("ImbalanceMethod", imbalanceMethods, SUPPORTED_IMBALANCE_METHODS)
  ValidateGridDimension("OutlierMethod", outlierMethods, SUPPORTED_OUTLIER_METHODS)

  # Bail out early with a clear message if no datasets were configured.
  if (len(datasets) == 0):
    fprint("No datasets configured. Add at least one entry under 'Datasets:' in your config_dl.yaml.")
  else:
    # Loop through every combination of dataset, architecture, optimizer, loss function, scaler,
    # imbalance method, and outlier method.
    for datasetCfg in datasets:
      for architecture in architectures:
        for optimizerName in optimizerNames:
          for lossFunction in lossFunctions:
            for scalerName in scalers:
              for imbalanceMethod in imbalanceMethods:
                for outlierMethod in outlierMethods:
                  # Construct the dynamic experiment output directory name.
                  experimentName = (
                    f"Exp-{datasetCfg['Name']}-{architecture}-{optimizerName}-{lossFunction}-"
                    f"{scalerName}-{imbalanceMethod}-{outlierMethod}")
                  currentOutputDir = str(Path(baseOutputDir) / experimentName)

                  # Run the experiment inside a try/except so one failure does not abort the whole grid.
                  try:
                    RunCompletePipeline(
                      datasetCfg=datasetCfg, architecture=architecture, optimizerName=optimizerName,
                      lossFunction=lossFunction, scalerName=scalerName, imbalanceMethod=imbalanceMethod,
                      outlierMethod=outlierMethod, outputDir=currentOutputDir, config=config,
                    )
                    # Save a copy of the configuration used for this run, for reproducibility.
                    configCopy = dict(config)
                    configCopy.update({
                      "Dataset": datasetCfg["Name"], "Architecture": architecture, "OptimizerName": optimizerName,
                      "LossFunction": lossFunction, "ScalerName": scalerName,
                      "ImbalanceMethod": imbalanceMethod, "OutlierMethod": outlierMethod,
                    })
                    with open(Path(currentOutputDir) / "ConfigUsed.yaml", "w") as f:
                      yaml.dump(configCopy, f)
                  except Exception as e:
                    fprint(f"ERROR in experiment {experimentName}: {e}")
                    import traceback
                    traceback.print_exc()
                    fprint("Continuing to the next experiment...")
                    continue

    # Print a separator line and the final completion message.
    fprint("\n" + "=" * 70)
    fprint("All experiments completed.")
