import os  # Import the operating system module.
import csv  # Import the csv module for writing CSV files.
import json  # Import the json module for data serialization.
import yaml  # Import the yaml module for configuration parsing.
import time  # Import the time module for timing operations.
import numpy  # Import the numpy module for numerical operations.
import pandas  # Import the pandas module for tabular data handling.
import optuna  # Import optuna for hyperparameter tuning.
import xgboost  # Import xgboost for the gradient-boosted tree model.
import warnings  # Import the warnings module to silence noisy library warnings.
import seaborn  # Import the seaborn module for statistical data visualization.
import joblib  # Import joblib for model persistence.
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib.
from pathlib import Path  # Import the Path class from pathlib.
from typing import Dict, List, Tuple, Optional, Union, Any  # Import typing utilities.

# Import the class-imbalance resamplers from imbalanced-learn.
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Import the scalers used by the normalization step.
from sklearn.preprocessing import (
  StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, LabelEncoder,
)
# Import the outlier-detection estimators used by the outlier-removal step.
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Import every classification model family this pipeline can build.
from sklearn.ensemble import (
  RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
  AdaBoostClassifier, BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Import LightGBM as an optional extra gradient-boosting model family.
try:
  import lightgbm
  LIGHTGBM_AVAILABLE = True
except ImportError:
  LIGHTGBM_AVAILABLE = False

# Import the train/val/test splitting utility.
from sklearn.model_selection import train_test_split
# Import the class-weight balancing utility.
from sklearn.utils.class_weight import compute_class_weight
# Import the feature-selection score functions.
from sklearn.feature_selection import mutual_info_classif, f_classif
# Import metric functions from sklearn.
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score,
  confusion_matrix, classification_report, roc_auc_score, log_loss,
)

# Import the sport-specific dataset builders (still available as "Builtin" datasets).
from UFCFeatures import BuildUFCDataset
from BasketballFeatures import BuildBasketballDataset

# Silence noisy library warnings so the console log stays readable.
warnings.filterwarnings("ignore")
# Silence optuna's per-trial console spam (we print our own summary lines instead).
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Define the fprint helper for flushed, timestamp-free console logging.
def fprint(msg, *args, **kwargs):
  # Print the message immediately, flushing the output buffer.
  print(msg, flush=True, *args, **kwargs)


# Define the NumpyEncoder class.
class NumpyEncoder(json.JSONEncoder):
  """
  Custom JSON encoder to handle numpy arrays and scalar types.
  """

  # Define the default serialization method.
  def default(self, obj):
    # Check if the object is a numpy array.
    if (isinstance(obj, numpy.ndarray)):
      # Convert the numpy array to a native Python list.
      return obj.tolist()
    # Check if the object is a numpy scalar (integer or float).
    if (isinstance(obj, (numpy.integer, numpy.floating))):
      # Convert the numpy scalar to a native Python int or float.
      return obj.item()
    # Check if the object is a pandas Timestamp.
    if (isinstance(obj, pandas.Timestamp)):
      # Convert the timestamp to an ISO-formatted string.
      return obj.isoformat()
    # Fallback to the default encoder for other types.
    return super().default(obj)


# Define the LoadConfig function.
def LoadConfig(
  configPath: str,
) -> Dict[str, Any]:
  """
  Load configuration from a YAML or JSON file.
  """
  # Check if the file exists.
  if (not Path(configPath).exists()):
    # Raise an error if the file is not found.
    raise FileNotFoundError(f"Configuration file not found: {configPath}")
  # Open the file for reading.
  with open(configPath, "r") as f:
    # Check the file extension.
    if (configPath.endswith(".json")):
      # Load the JSON configuration.
      config = json.load(f)
    elif (configPath.endswith(".yaml") or configPath.endswith(".yml")):
      # Load the YAML configuration.
      config = yaml.safe_load(f)
    else:
      # Raise an error for unsupported file formats.
      raise ValueError("Unsupported configuration file format. Use .json, .yaml, or .yml.")
  # Return the loaded configuration dictionary.
  return config


# Define the registry of built-in datasets (the UFC / basketball pipelines built earlier).
# A config Dataset entry can reference one of these by name via "Builtin: <name>",
# OR point at any arbitrary CSV via "CsvPath" + "TargetColumn" instead - see ResolveDataset.
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
  "UfcRound": {
    "Builder"  : lambda: BuildUFCDataset(),
    "TargetCol": "roundClass",
    "ExtraDrop": ["methodClass"],
    "CatCols"  : ["f1stance", "f2stance", "weightClassClean", "stanceMatchup"],
  },
  "UfcMethod": {
    "Builder"  : lambda: BuildUFCDataset(),
    "TargetCol": "methodClass",
    "ExtraDrop": ["roundClass"],
    "CatCols"  : ["f1stance", "f2stance", "weightClassClean", "stanceMatchup"],
  },
  "NbaWin": {
    "Builder"  : lambda: BuildBasketballDataset("nba"),
    "TargetCol": "targetHomeWin",
    "ExtraDrop": [],
    "CatCols"  : ["season_type"],
  },
  "WnbaWin": {
    "Builder"  : lambda: BuildBasketballDataset("wnba"),
    "TargetCol": "targetHomeWin",
    "ExtraDrop": [],
    "CatCols"  : ["season_type"],
  },
  "NcaaMbbWin": {
    "Builder"  : lambda: BuildBasketballDataset("ncaa_mbb"),
    "TargetCol": "targetHomeWin",
    "ExtraDrop": [],
    "CatCols"  : ["season_type"],
  },
  "NcaaWbbWin": {
    "Builder"  : lambda: BuildBasketballDataset("ncaa_wbb"),
    "TargetCol": "targetHomeWin",
    "ExtraDrop": [],
    "CatCols"  : ["season_type"],
  },
}

# Define the set of model names this pipeline knows how to build. Comment models
# out of your config.yaml's "Models" list to skip them - nothing else needs to change.
SUPPORTED_MODELS = [
  "RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost", "DecisionTree",
  "KNN", "LogisticRegression", "SVC", "GaussianNB", "SGD", "MLP", "Bagging",
  "XGBoost",
] + (["LightGBM"] if (LIGHTGBM_AVAILABLE) else [])

# Define which model families accept a "class_weight" constructor argument directly.
CLASS_WEIGHT_PARAM_MODELS = {"RandomForest", "ExtraTrees", "DecisionTree", "LogisticRegression", "SVC", "SGD"}
# Define which model families accept a "sample_weight" argument to .fit() instead.
SAMPLE_WEIGHT_FIT_MODELS = {"GradientBoosting", "GaussianNB", "Bagging", "XGBoost", "LightGBM"}
# KNN and MLP support neither mechanism in scikit-learn, so the "ClassWeight" imbalance
# method has no effect for those two models (documented here so it isn't a silent surprise).

# Define the set of class-imbalance method names this pipeline supports.
SUPPORTED_IMBALANCE_METHODS = [
  "SMOTE", "ADASYN", "BorderlineSMOTE", "RandomOverSampler", "RandomUnderSampler",
  "SMOTETomek", "ClassWeight", "None",
]

# Define the set of outlier-removal method names this pipeline supports.
SUPPORTED_OUTLIER_METHODS = ["IQR", "ZScore", "IsolationForest", "LocalOutlierFactor", "EllipticEnvelope", "None"]

# Define the set of scaler names this pipeline supports.
SUPPORTED_SCALERS = ["Standard", "MinMax", "Robust", "MaxAbs", "Normalizer", "QuantileTransformer"]


# Define the ResolveDataset function, translating one config "Dataset" entry into a raw dataframe.
def ResolveDataset(
  datasetCfg: Dict[str, Any],
) -> Tuple[pandas.DataFrame, str, List[str], List[str]]:
  # Check whether this entry points at one of the built-in sport pipelines.
  if (datasetCfg.get("Builtin")):
    # Look up the built-in task definition.
    taskDef = TASK_REGISTRY[datasetCfg["Builtin"]]
    # Build the raw dataset using the task's registered builder function.
    df = taskDef["Builder"]()
    # Return the dataframe, target column, sibling-target drops, and categorical columns.
    return df, taskDef["TargetCol"], taskDef["ExtraDrop"], taskDef["CatCols"]

  # Otherwise, this is a generic CSV dataset - read the CSV from the configured path.
  csvPath = datasetCfg["CsvPath"]
  # Read the CSV file into a dataframe.
  df = pandas.read_csv(csvPath)
  # Optionally drop the first column (a common convention for an unnamed index/ID column).
  if (datasetCfg.get("DropFirstColumn", False)):
    # Drop the first column by position.
    df = df.drop(columns=[df.columns[0]])
  # Extract the target column name (required for a generic CSV dataset).
  targetCol = datasetCfg["TargetColumn"]
  # Extract any explicit columns to drop before modeling.
  explicitDrop = list(datasetCfg.get("DropColumns", []))
  # Determine the categorical columns: either explicitly configured, or auto-detected.
  catCols = datasetCfg.get("CategoricalColumns")
  # Auto-detect object/category dtype columns (excluding the target) if not explicitly given.
  if (catCols is None):
    # Build the auto-detected categorical column list.
    catCols = [c for c in df.columns if (c != targetCol and df[c].dtype == object)]
  # Return the dataframe, target column, explicit drops, and categorical columns.
  return df, targetCol, explicitDrop, catCols


# Define the DropUnnecessaryColumns function (Step 1 of the pipeline).
def DropUnnecessaryColumns(
  df: pandas.DataFrame,
  targetCol: str,
  explicitDropCols: List[str],
  idLikeMaxUniqueRatio: float = 0.98,
  constantMaxUnique: int = 1,
  highMissingThreshold: float = 0.6,
) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
  # Work on a copy to avoid mutating the caller's frame.
  df = df.copy()
  # Initialize the report dictionary tracking every column dropped and why.
  report = {"ExplicitDrops": [], "AutoIdLikeDrops": [], "AutoConstantDrops": [], "AutoHighMissingDrops": []}

  # Drop the explicitly requested unnecessary columns (e.g. the sibling target column).
  explicit = [c for c in explicitDropCols if (c in df.columns)]
  # Apply the explicit drop.
  df = df.drop(columns=explicit)
  # Record the explicit drops in the report.
  report["ExplicitDrops"] = explicit

  # Get the total row count for missing/unique ratio calculations.
  nRows = len(df)
  # Iterate through every remaining column to apply the automatic cleaning heuristics.
  for c in list(df.columns):
    # Never auto-drop the target column.
    if (c == targetCol):
      # Skip to the next column.
      continue
    # Count the number of unique non-null values in this column.
    nUnique = df[c].nunique(dropna=True)
    # Check for a near-unique text/id-like column (e.g. a raw identifier) that carries no signal.
    if (df[c].dtype == object and nUnique / max(nRows, 1) >= idLikeMaxUniqueRatio):
      # Drop the id-like column.
      df = df.drop(columns=[c])
      # Record the drop in the report.
      report["AutoIdLikeDrops"].append(c)
      # Continue to the next column.
      continue
    # Check for a constant (or near-constant) column that carries no signal.
    if (nUnique <= constantMaxUnique):
      # Drop the constant column.
      df = df.drop(columns=[c])
      # Record the drop in the report.
      report["AutoConstantDrops"].append(c)
      # Continue to the next column.
      continue
    # Check for a column with too much missing data to be useful.
    missingFrac = df[c].isna().mean()
    # Check if the missing fraction exceeds the configured threshold.
    if (missingFrac >= highMissingThreshold):
      # Drop the high-missingness column.
      df = df.drop(columns=[c])
      # Record the drop in the report.
      report["AutoHighMissingDrops"].append(c)

  # Return the cleaned dataframe and the drop report.
  return df, report


# Define the RemoveOutliers function (Step 2 of the pipeline): dispatches to one of several
# interchangeable outlier-detection techniques, selected via config.
def RemoveOutliers(
  df: pandas.DataFrame,
  targetCol: str,
  method: str,
  iqrMultiplier: float = 3.0,
  zScoreThreshold: float = 3.0,
  contamination: float = 0.05,
  maxRowDropFrac: float = 0.15,
  randomState: int = 42,
) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
  # Work on a copy to avoid mutating the caller's frame.
  df = df.copy()
  # Record the row count before any removal.
  nBefore = len(df)

  # Handle the "no outlier removal" option first.
  if (method == "None"):
    # Return the dataframe unchanged.
    return df, {"Method": "None", "RowsBefore": nBefore, "RowsAfter": nBefore, "RowsRemoved": 0,
                "RemovedFraction": 0.0}

  # Identify the numeric columns eligible for outlier checks (never the target or the internal row id).
  numericCols = [c for c in df.select_dtypes(include=[numpy.number]).columns if (c not in (targetCol, "RowId"))]
  # Restrict to columns with meaningfully continuous values (skip binary/one-hot-like flags).
  numericCols = [c for c in numericCols if (df[c].nunique(dropna=True) > 5)]
  # Bail out if there are no eligible numeric columns.
  if (len(numericCols) == 0):
    # Return the dataframe unchanged.
    return df, {"Method": method, "RowsBefore": nBefore, "RowsAfter": nBefore, "RowsRemoved": 0,
                "RemovedFraction": 0.0, "Note": "No eligible numeric columns."}

  # Build a NaN-filled matrix (median imputed) purely for the outlier detectors below,
  # which cannot natively handle missing values. The original dataframe values are untouched.
  detectorMatrix = df[numericCols].apply(lambda s: s.fillna(s.median()))

  # Dispatch to the requested outlier-detection technique.
  if (method == "IQR"):
    # Initialize the boolean keep-mask covering every row.
    keepMask = pandas.Series(True, index=df.index)
    # Initialize the dictionary of per-column outlier bounds.
    bounds = {}
    # Iterate through each numeric column to compute IQR-based bounds.
    for c in numericCols:
      # Compute the first and third quartiles.
      q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
      # Compute the interquartile range.
      iqr = q3 - q1
      # Skip columns with a degenerate (zero or NaN) IQR.
      if (iqr == 0 or pandas.isna(iqr)):
        # Continue to the next column.
        continue
      # Compute the lower and upper bounds.
      lowerBound, upperBound = q1 - iqrMultiplier * iqr, q3 + iqrMultiplier * iqr
      # Record the bounds for this column.
      bounds[c] = (float(lowerBound), float(upperBound))
      # Update the keep-mask, treating missing values as "not an outlier".
      keepMask &= df[c].between(lowerBound, upperBound) | df[c].isna()
    # Count the rows flagged as outliers.
    nFlagged = int((~keepMask).sum())
    # Apply the safety cap if too many rows were flagged.
    if (nFlagged / max(nBefore, 1) > maxRowDropFrac):
      # Count, per row, how many columns it is extreme on.
      outlierCounts = pandas.Series(0, index=df.index)
      # Iterate through the computed bounds.
      for c, (lowerBound, upperBound) in bounds.items():
        # Increment the per-row outlier count for this column.
        outlierCounts += (~(df[c].between(lowerBound, upperBound) | df[c].isna())).astype(int)
      # Only drop rows extreme on an unusually high number of columns.
      cutoff = max(2, int(outlierCounts.quantile(0.99)))
      # Rebuild the keep-mask using the relaxed criterion.
      keepMask = outlierCounts < cutoff

  elif (method == "ZScore"):
    # Compute the absolute z-score of every numeric column.
    zScores = (detectorMatrix - detectorMatrix.mean()) / detectorMatrix.std(ddof=0).replace(0, numpy.nan)
    # Keep rows where every column's z-score is within the threshold (or the column was degenerate).
    keepMask = (zScores.abs() <= zScoreThreshold).fillna(True).all(axis=1)

  elif (method == "IsolationForest"):
    # Instantiate the Isolation Forest outlier detector.
    detector = IsolationForest(contamination=contamination, random_state=randomState, n_jobs=-1)
    # Fit and predict in one step (-1 = outlier, 1 = inlier).
    predictions = detector.fit_predict(detectorMatrix)
    # Build the keep-mask from the inlier predictions.
    keepMask = pandas.Series(predictions == 1, index=df.index)

  elif (method == "LocalOutlierFactor"):
    # Instantiate the Local Outlier Factor detector (unsupervised, no separate predict step).
    detector = LocalOutlierFactor(n_neighbors=min(20, max(2, len(df) - 1)), contamination=contamination)
    # Fit and predict in one step (-1 = outlier, 1 = inlier).
    predictions = detector.fit_predict(detectorMatrix)
    # Build the keep-mask from the inlier predictions.
    keepMask = pandas.Series(predictions == 1, index=df.index)

  elif (method == "EllipticEnvelope"):
    # Try the Elliptic Envelope detector, which assumes a roughly Gaussian feature distribution
    # and can fail on highly collinear data - fall back to keeping everything if it errors out.
    try:
      # Instantiate the Elliptic Envelope detector.
      detector = EllipticEnvelope(contamination=contamination, random_state=randomState)
      # Fit and predict in one step (-1 = outlier, 1 = inlier).
      predictions = detector.fit_predict(detectorMatrix)
      # Build the keep-mask from the inlier predictions.
      keepMask = pandas.Series(predictions == 1, index=df.index)
    except Exception as e:
      # Keep every row and note the failure in the report.
      keepMask = pandas.Series(True, index=df.index)
      # Print a warning so the failure is visible in the console log.
      fprint(f"  Warning: EllipticEnvelope failed ({e}); skipping outlier removal for this run.")

  else:
    # Raise an error for an unrecognized outlier method name.
    raise ValueError(f"Unsupported outlier method: {method}. Choose from {SUPPORTED_OUTLIER_METHODS}.")

  # Apply the final safety cap (shared by every method): never remove more than the configured fraction.
  nFlaggedFinal = int((~keepMask).sum())
  # Check if the safety cap was exceeded.
  if (nFlaggedFinal / max(nBefore, 1) > maxRowDropFrac):
    # Print a warning explaining the fallback.
    fprint(f"  Warning: {method} flagged {nFlaggedFinal}/{nBefore} rows as outliers "
           f"(> {maxRowDropFrac:.0%} cap); keeping only the most extreme rows up to the cap.")
    # Fall back to dropping only the configured fraction of the most extreme rows, ranked by
    # summed absolute z-score across the numeric columns (a method-agnostic "how extreme" score).
    extremeness = (detectorMatrix - detectorMatrix.mean()).abs().div(
      detectorMatrix.std(ddof=0).replace(0, numpy.nan)).fillna(0).sum(axis=1)
    # Compute how many rows the cap allows removing.
    nAllowedToRemove = int(nBefore * maxRowDropFrac)
    # Identify the row labels of the most extreme rows, up to the allowed count.
    rowsToRemove = extremeness.sort_values(ascending=False).head(nAllowedToRemove).index
    # Rebuild the keep-mask excluding only those capped rows.
    keepMask = ~df.index.isin(rowsToRemove)

  # Apply the keep-mask and reset the index.
  dfClean = df[keepMask].reset_index(drop=True)
  # Build the report describing the outlier removal step.
  report = {
    "Method"         : method,
    "RowsBefore"     : nBefore,
    "RowsAfter"      : len(dfClean),
    "RowsRemoved"    : nBefore - len(dfClean),
    "RemovedFraction": round((nBefore - len(dfClean)) / max(nBefore, 1), 4),
    "ColumnsChecked" : numericCols,
  }
  # Return the cleaned dataframe and the report.
  return dfClean, report


# Define the SplitData function (Step 3 of the pipeline).
def SplitData(
  df: pandas.DataFrame,
  targetCol: str,
  testSize: float,
  valSize: float,
  randomState: int,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, Dict[str, Any]]:
  # Extract the target series for stratification.
  y = df[targetCol]
  # Only stratify if every class has at least 2 members.
  stratify = y if (y.value_counts().min() >= 2) else None
  # Perform the first split, separating the training set from a combined val+test set.
  trainDf, tempDf = train_test_split(
    df, test_size=(testSize + valSize), random_state=randomState, stratify=stratify,
  )
  # Compute the relative test fraction within the combined val+test set.
  relativeTestSize = testSize / (testSize + valSize)
  # Only stratify the second split if every class still has at least 2 members.
  stratify2 = tempDf[targetCol] if (
    stratify is not None and tempDf[targetCol].value_counts().min() >= 2
  ) else None
  # Perform the second split, separating validation from test.
  valDf, testDf = train_test_split(
    tempDf, test_size=relativeTestSize, random_state=randomState, stratify=stratify2,
  )
  # Build the report describing the resulting split sizes.
  report = {
    "TrainRows": len(trainDf), "ValRows": len(valDf), "TestRows": len(testDf),
    "TrainFraction": round(len(trainDf) / len(df), 4),
    "ValFraction"  : round(len(valDf) / len(df), 4),
    "TestFraction" : round(len(testDf) / len(df), 4),
  }
  # Return the three splits and the report, with indices reset.
  return trainDf.reset_index(drop=True), valDf.reset_index(drop=True), testDf.reset_index(drop=True), report


# Define the DropRedundantColumns function (Step 4 of the pipeline).
def DropRedundantColumns(
  trainDf: pandas.DataFrame,
  valDf: pandas.DataFrame,
  testDf: pandas.DataFrame,
  targetCol: str,
  correlationThreshold: float,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, Dict[str, Any]]:
  # Identify the numeric columns eligible for correlation checks (never the target or the internal row id).
  numericCols = [c for c in trainDf.select_dtypes(include=[numpy.number]).columns if (c not in (targetCol, "RowId"))]
  # Initialize the set of columns to drop as redundant.
  toDrop = set()
  # Only compute correlations if there is more than one numeric column.
  if (len(numericCols) > 1):
    # Compute the absolute pairwise correlation matrix on the training fold only.
    corr = trainDf[numericCols].corr().abs()
    # Mask to the upper triangle to avoid double-counting each pair.
    upper = corr.where(numpy.triu(numpy.ones(corr.shape), k=1).astype(bool))
    # Iterate through each column in the upper-triangular matrix.
    for col in upper.columns:
      # Skip columns already marked for removal.
      if (col in toDrop):
        # Continue to the next column.
        continue
      # Find every column highly correlated with this one.
      correlatedWith = upper.index[upper[col] > correlationThreshold].tolist()
      # Mark each correlated partner for removal.
      for other in correlatedWith:
        # Only add the partner if it is not already tracked.
        if (other not in toDrop):
          # Add the redundant column to the drop set.
          toDrop.add(other)

  # Apply the redundant-column drop to all three splits.
  trainDf = trainDf.drop(columns=list(toDrop))
  # Apply the same drop to the validation split.
  valDf = valDf.drop(columns=list(toDrop))
  # Apply the same drop to the test split.
  testDf = testDf.drop(columns=list(toDrop))
  # Build the report describing which columns were dropped as redundant.
  report = {"RedundantColumnsDropped": sorted(toDrop), "CorrelationThreshold": correlationThreshold}
  # Return the cleaned splits and the report.
  return trainDf, valDf, testDf, report


# Define the EncodeCategoricalColumns function.
def EncodeCategoricalColumns(
  trainDf: pandas.DataFrame,
  valDf: pandas.DataFrame,
  testDf: pandas.DataFrame,
  categoricalCols: List[str],
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, List[str]]:
  # Restrict to the categorical columns that actually survived the earlier cleaning steps.
  catCols = [c for c in categoricalCols if (c in trainDf.columns)]
  # Return unchanged if there are no categorical columns to encode.
  if (len(catCols) == 0):
    # Return the original splits and an empty new-columns list.
    return trainDf, valDf, testDf, []

  # One-hot encode the categorical columns on each split independently.
  trainEnc = pandas.get_dummies(trainDf, columns=catCols, dummy_na=True)
  valEnc = pandas.get_dummies(valDf, columns=catCols, dummy_na=True)
  testEnc = pandas.get_dummies(testDf, columns=catCols, dummy_na=True)

  # Align the validation and test columns to the training columns, filling any gaps with 0.
  trainEnc, valEnc = trainEnc.align(valEnc, join="left", axis=1, fill_value=0)
  trainEnc, testEnc = trainEnc.align(testEnc, join="left", axis=1, fill_value=0)
  # Re-order the validation and test columns to exactly match the training column order.
  valEnc = valEnc[trainEnc.columns]
  testEnc = testEnc[trainEnc.columns]
  # Determine which columns were newly created by the one-hot encoding.
  newCols = [c for c in trainEnc.columns if (c not in trainDf.columns)]
  # Return the encoded splits and the list of new columns.
  return trainEnc, valEnc, testEnc, newCols


# Define the HandleClassImbalance function (Step 5 of the pipeline): dispatches to one of
# several interchangeable resamplers, or to class weighting, selected via config.
def HandleClassImbalance(
  xTrain: pandas.DataFrame,
  yTrain: pandas.Series,
  imbalanceMethod: str,
  randomState: int,
) -> Tuple[pandas.DataFrame, pandas.Series, Optional[Dict[int, float]], Dict[str, Any]]:
  # Record the class distribution before any balancing.
  before = yTrain.value_counts().to_dict()
  # Initialize the class-weights dictionary (only populated for the ClassWeight method).
  classWeights = None
  # Determine the smallest class size, used to size neighborhood-based resamplers safely.
  minClassCount = yTrain.value_counts().min()
  # Clamp the neighbor count so the resamplers never request more neighbors than available.
  kNeighbors = max(1, min(5, minClassCount - 1))
  # Only the resampling methods (as opposed to ClassWeight/None) require more than 1 sample
  # per class and more than 1 class to run safely.
  canResample = (minClassCount > 1 and yTrain.nunique() > 1)

  # Dispatch to the requested imbalance-handling method.
  if (imbalanceMethod == "SMOTE" and canResample):
    # Instantiate and apply the SMOTE oversampler.
    xResampled, yResampled = SMOTE(random_state=randomState, k_neighbors=kNeighbors).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "ADASYN" and canResample):
    # Instantiate and apply the ADASYN oversampler.
    xResampled, yResampled = ADASYN(random_state=randomState, n_neighbors=kNeighbors).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "BorderlineSMOTE" and canResample):
    # Instantiate and apply the Borderline-SMOTE oversampler.
    xResampled, yResampled = BorderlineSMOTE(
      random_state=randomState, k_neighbors=kNeighbors).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "RandomOverSampler" and canResample):
    # Instantiate and apply simple random oversampling of the minority class(es).
    xResampled, yResampled = RandomOverSampler(random_state=randomState).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "RandomUnderSampler" and canResample):
    # Instantiate and apply simple random undersampling of the majority class(es).
    xResampled, yResampled = RandomUnderSampler(random_state=randomState).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "SMOTETomek" and canResample):
    # Instantiate and apply the hybrid SMOTE + Tomek-link cleaning resampler.
    xResampled, yResampled = SMOTETomek(random_state=randomState).fit_resample(xTrain, yTrain)
  elif (imbalanceMethod == "ClassWeight"):
    # Keep the training data unresampled when using class weights instead.
    xResampled, yResampled = xTrain, yTrain
    # Determine the unique classes present in the training fold.
    classes = numpy.unique(yTrain)
    # Compute the balanced class weights.
    weights = compute_class_weight("balanced", classes=classes, y=yTrain)
    # Build the class-to-weight mapping.
    classWeights = dict(zip(classes.tolist(), weights.tolist()))
  elif (imbalanceMethod == "None" or not canResample):
    # Leave the training data untouched (also the safe fallback if resampling isn't possible).
    xResampled, yResampled = xTrain, yTrain
  else:
    # Raise an error for an unrecognized imbalance method name.
    raise ValueError(f"Unsupported imbalance method: {imbalanceMethod}. "
                      f"Choose from {SUPPORTED_IMBALANCE_METHODS}.")

  # Record the class distribution after balancing.
  after = pandas.Series(yResampled).value_counts().to_dict()
  # Build the report describing the imbalance-handling step.
  report = {
    "Method"                 : imbalanceMethod,
    "ClassDistributionBefore": {str(k): int(v) for k, v in before.items()},
    "ClassDistributionAfter" : {str(k): int(v) for k, v in after.items()},
    "ClassWeights"           : classWeights,
  }
  # Return the (possibly resampled) training data, the class weights, and the report.
  return xResampled, yResampled, classWeights, report


# Define the GetScalerObject function, dispatching a scaler name to its scikit-learn instance.
def GetScalerObject(scalerName: str, randomState: int = 42):
  # Check which scaler was requested.
  if (scalerName == "Standard"):
    # Standardize features by removing the mean and scaling to unit variance.
    return StandardScaler()
  elif (scalerName == "MinMax"):
    # Scale features to a fixed [0, 1] range.
    return MinMaxScaler()
  elif (scalerName == "Robust"):
    # Scale features using statistics robust to outliers (median / IQR).
    return RobustScaler()
  elif (scalerName == "MaxAbs"):
    # Scale each feature by its maximum absolute value.
    return MaxAbsScaler()
  elif (scalerName == "Normalizer"):
    # Scale each individual row (sample) to unit norm.
    return Normalizer()
  elif (scalerName == "QuantileTransformer"):
    # Map features to a (roughly) normal output distribution via quantiles.
    return QuantileTransformer(output_distribution="normal", random_state=randomState)
  else:
    # Raise an error for an unrecognized scaler name.
    raise ValueError(f"Unsupported scaler: {scalerName}. Choose from {SUPPORTED_SCALERS}.")


# Define the NormalizeFeatures function (Step 6 of the pipeline).
def NormalizeFeatures(
  xTrain: pandas.DataFrame,
  xVal: pandas.DataFrame,
  xTest: pandas.DataFrame,
  scalerName: str,
  randomState: int,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, Any]:
  # Instantiate the requested scaler.
  scaler = GetScalerObject(scalerName, randomState)
  # Fit the scaler on the training fold and transform it.
  xTrainScaled = pandas.DataFrame(scaler.fit_transform(xTrain), columns=xTrain.columns, index=xTrain.index)
  # Transform the validation fold using the training-fit scaler.
  xValScaled = pandas.DataFrame(scaler.transform(xVal), columns=xVal.columns, index=xVal.index)
  # Transform the test fold using the training-fit scaler.
  xTestScaled = pandas.DataFrame(scaler.transform(xTest), columns=xTest.columns, index=xTest.index)
  # Return the three scaled splits and the fitted scaler.
  return xTrainScaled, xValScaled, xTestScaled, scaler


# Define the SelectFeatures function (Step 7 of the pipeline): 3 methods, 2-of-3 vote.
def SelectFeatures(
  xTrain: pandas.DataFrame,
  yTrain: pandas.Series,
  topKFraction: float,
  minFeaturesSelected: int,
  randomState: int,
) -> Tuple[List[str], Dict[str, Any]]:
  # Determine the total number of candidate features.
  nFeatures = xTrain.shape[1]
  # Compute how many features each method should keep, respecting the configured minimum.
  topK = max(minFeaturesSelected, int(numpy.ceil(nFeatures * topKFraction)))
  # Clamp topK to the total number of available features.
  topK = min(topK, nFeatures)

  # Method 1: mutual information between each feature and the target.
  miScores = mutual_info_classif(xTrain, yTrain, random_state=randomState)
  # Rank the features by mutual information score, descending.
  miRanked = pandas.Series(miScores, index=xTrain.columns).sort_values(ascending=False)
  # Keep the top-K mutual-information features.
  miSelected = set(miRanked.head(topK).index)

  # Method 2: ANOVA F-statistic between each feature and the target.
  fScores, _ = f_classif(xTrain, yTrain)
  # Replace any NaN F-scores (constant features within a class) with 0.
  fScores = numpy.nan_to_num(fScores, nan=0.0)
  # Rank the features by F-score, descending.
  fRanked = pandas.Series(fScores, index=xTrain.columns).sort_values(ascending=False)
  # Keep the top-K ANOVA F-statistic features.
  fSelected = set(fRanked.head(topK).index)

  # Method 3: Random Forest feature importance (used purely as a feature-ranking tool here,
  # independent of whatever model family is actually being trained in Step 9).
  rf = RandomForestClassifier(n_estimators=300, random_state=randomState, n_jobs=-1)
  # Fit the Random Forest on the training fold.
  rf.fit(xTrain, yTrain)
  # Rank the features by Random Forest importance, descending.
  rfRanked = pandas.Series(rf.feature_importances_, index=xTrain.columns).sort_values(ascending=False)
  # Keep the top-K Random Forest importance features.
  rfSelected = set(rfRanked.head(topK).index)

  # Initialize the per-feature vote counter.
  votes = pandas.Series(0, index=xTrain.columns)
  # Iterate through each method's selected-feature set.
  for selectedSet in (miSelected, fSelected, rfSelected):
    # Increment the vote count for every feature this method selected.
    votes.loc[list(selectedSet)] += 1

  # Keep only the features chosen by at least 2 of the 3 methods.
  finalFeatures = votes[votes >= 2].index.tolist()
  # Fall back to the highest-vote features if the 2-of-3 rule is too strict for this dataset.
  if (len(finalFeatures) < minFeaturesSelected):
    # Take the top-voted features up to the configured minimum.
    finalFeatures = votes.sort_values(ascending=False).head(minFeaturesSelected).index.tolist()

  # Build the report describing every method's ranking and the final vote outcome.
  report = {
    "NFeaturesBefore"                  : nFeatures,
    "TopKPerMethod"                    : topK,
    "MutualInformationTopFeatures"     : miRanked.head(topK).round(5).to_dict(),
    "AnovaFTopFeatures"                : fRanked.head(topK).round(5).to_dict(),
    "RandomForestImportanceTopFeatures": rfRanked.head(topK).round(5).to_dict(),
    "VotesPerFeature"                  : votes.to_dict(),
    "FinalSelectedFeatures"            : finalFeatures,
    "NFeaturesSelected"                : len(finalFeatures),
  }
  # Return the final selected feature list and the report.
  return finalFeatures, report


# Define the SuggestHyperparameters function: the Optuna search space for each model family.
def SuggestHyperparameters(trial: optuna.Trial, modelName: str) -> Dict[str, Any]:
  # Check which model family is being tuned and return its search space.
  if (modelName in ("RandomForest", "ExtraTrees")):
    # Shared search space for the two bagged-tree ensembles.
    return {
      "n_estimators"     : trial.suggest_int("n_estimators", 100, 600),
      "max_depth"        : trial.suggest_int("max_depth", 3, 30),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
      "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
      "max_features"     : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
  elif (modelName == "GradientBoosting"):
    # Search space for scikit-learn's Gradient Boosting classifier.
    return {
      "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
      "max_depth"        : trial.suggest_int("max_depth", 2, 8),
      "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
      "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
      "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
    }
  elif (modelName == "AdaBoost"):
    # Search space for AdaBoost.
    return {
      "n_estimators" : trial.suggest_int("n_estimators", 50, 500),
      "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
    }
  elif (modelName == "DecisionTree"):
    # Search space for a single Decision Tree.
    return {
      "max_depth"        : trial.suggest_int("max_depth", 2, 30),
      "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
      "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
      "criterion"        : trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }
  elif (modelName == "KNN"):
    # Search space for K-Nearest Neighbors.
    return {
      "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
      "weights"    : trial.suggest_categorical("weights", ["uniform", "distance"]),
      "p"          : trial.suggest_categorical("p", [1, 2]),
    }
  elif (modelName == "LogisticRegression"):
    # Search space for L2-regularized Logistic Regression.
    return {
      "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
    }
  elif (modelName == "SVC"):
    # Search space for the Support Vector Classifier.
    return {
      "C"     : trial.suggest_float("C", 1e-2, 100.0, log=True),
      "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
      "gamma" : trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
  elif (modelName == "GaussianNB"):
    # Search space for Gaussian Naive Bayes.
    return {
      "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True),
    }
  elif (modelName == "SGD"):
    # Search space for a linear model fit via Stochastic Gradient Descent.
    # The loss is restricted to log_loss/modified_huber so predict_proba stays available.
    return {
      "loss"    : trial.suggest_categorical("loss", ["log_loss", "modified_huber"]),
      "alpha"   : trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
      "penalty" : trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
    }
  elif (modelName == "MLP"):
    # Search space for a Multi-Layer Perceptron.
    return {
      "hidden_layer_sizes": trial.suggest_categorical(
        "hidden_layer_sizes", ["(64,)", "(128,)", "(64,32)", "(128,64)"]),
      "alpha"             : trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
      "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
    }
  elif (modelName == "Bagging"):
    # Search space for a Bagging ensemble of decision trees.
    return {
      "n_estimators": trial.suggest_int("n_estimators", 10, 200),
      "max_samples" : trial.suggest_float("max_samples", 0.5, 1.0),
      "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }
  elif (modelName == "XGBoost"):
    # Search space for XGBoost.
    return {
      "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
      "max_depth"       : trial.suggest_int("max_depth", 2, 10),
      "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
      "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
      "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
      "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
      "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
      "gamma"           : trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
  elif (modelName == "LightGBM"):
    # Search space for LightGBM.
    return {
      "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
      "max_depth"       : trial.suggest_int("max_depth", 2, 12),
      "num_leaves"      : trial.suggest_int("num_leaves", 15, 255),
      "learning_rate"   : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
      "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
      "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
      "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
  else:
    # Raise an error for an unrecognized model name.
    raise ValueError(f"Unsupported model: {modelName}. Choose from {SUPPORTED_MODELS}.")


# Define the BuildModelFromParams function: instantiates an unfitted model of the requested family.
def BuildModelFromParams(
  modelName: str,
  params: Dict[str, Any],
  randomState: int,
  useClassWeight: bool,
  nClasses: int,
) -> Any:
  # Copy the params so we can safely add fixed settings without mutating the caller's dict.
  params = dict(params)
  # Determine the class_weight argument to pass to constructors that support it directly.
  classWeightArg = "balanced" if (useClassWeight) else None

  # Check which model family is being built.
  if (modelName == "RandomForest"):
    # Build the Random Forest classifier.
    return RandomForestClassifier(**params, random_state=randomState, n_jobs=-1, class_weight=classWeightArg)
  elif (modelName == "ExtraTrees"):
    # Build the Extra Trees classifier.
    return ExtraTreesClassifier(**params, random_state=randomState, n_jobs=-1, class_weight=classWeightArg)
  elif (modelName == "GradientBoosting"):
    # Build the Gradient Boosting classifier (weighted via sample_weight at fit time instead).
    return GradientBoostingClassifier(**params, random_state=randomState)
  elif (modelName == "AdaBoost"):
    # Build the AdaBoost classifier.
    return AdaBoostClassifier(**params, random_state=randomState)
  elif (modelName == "DecisionTree"):
    # Build the Decision Tree classifier.
    return DecisionTreeClassifier(**params, random_state=randomState, class_weight=classWeightArg)
  elif (modelName == "KNN"):
    # Build the K-Nearest Neighbors classifier (no weighting support in scikit-learn).
    return KNeighborsClassifier(**params, n_jobs=-1)
  elif (modelName == "LogisticRegression"):
    # Build the Logistic Regression classifier.
    return LogisticRegression(**params, random_state=randomState, max_iter=2000, class_weight=classWeightArg)
  elif (modelName == "SVC"):
    # Build the Support Vector Classifier (probability=True is required for predict_proba).
    return SVC(**params, probability=True, random_state=randomState, class_weight=classWeightArg)
  elif (modelName == "GaussianNB"):
    # Build the Gaussian Naive Bayes classifier (weighted via sample_weight at fit time instead).
    return GaussianNB(**params)
  elif (modelName == "SGD"):
    # Build the SGD-trained linear classifier.
    return SGDClassifier(**params, random_state=randomState, class_weight=classWeightArg)
  elif (modelName == "MLP"):
    # Parse the hidden_layer_sizes string (Optuna categoricals must be hashable, hence the string form).
    hiddenLayerSizes = eval(params.pop("hidden_layer_sizes"))
    # Build the Multi-Layer Perceptron classifier (no weighting support in scikit-learn).
    return MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, **params, random_state=randomState, max_iter=500)
  elif (modelName == "Bagging"):
    # Build the Bagging ensemble (weighted via sample_weight at fit time instead, if the base estimator supports it).
    return BaggingClassifier(**params, random_state=randomState, n_jobs=-1)
  elif (modelName == "XGBoost"):
    # Add the fixed XGBoost settings that are not tuned by Optuna. device="cpu" is forced
    # explicitly here (rather than left to XGBoost's own auto-detection) because recent XGBoost
    # wheels bundle CUDA support and probe for a GPU even when one isn't actually usable - on a
    # machine with an NVIDIA GPU present but an outdated/mismatched driver, that probe throws
    # "cudaErrorInsufficientDriver" and crashes training outright instead of falling back to CPU.
    # Forcing CPU keeps this pipeline running reliably on any machine regardless of GPU/driver
    # state; XGBoost on CPU is still fast enough for this pipeline's dataset sizes.
    params.update({
      "random_state": randomState, "n_jobs": -1, "device": "cpu",
      "eval_metric" : "mlogloss" if (nClasses > 2) else "logloss",
    })
    # Configure the multiclass objective if there are more than 2 classes.
    if (nClasses > 2):
      # Set the multiclass softmax-probability objective and class count.
      params["objective"], params["num_class"] = "multi:softprob", nClasses
    # Build the XGBoost classifier.
    return xgboost.XGBClassifier(**params)
  elif (modelName == "LightGBM"):
    # Build the LightGBM classifier (weighting is supported directly via class_weight here).
    return lightgbm.LGBMClassifier(**params, random_state=randomState, n_jobs=-1,
                                    class_weight=classWeightArg, verbosity=-1)
  else:
    # Raise an error for an unrecognized model name.
    raise ValueError(f"Unsupported model: {modelName}. Choose from {SUPPORTED_MODELS}.")


# Define the FitModel function: fits a model, routing class weights to sample_weight when
# the model family doesn't accept a class_weight constructor argument.
def FitModel(
  model: Any,
  modelName: str,
  xTrain: pandas.DataFrame,
  yTrain: pandas.Series,
  classWeights: Optional[Dict[int, float]],
) -> Any:
  # Check whether this model family should receive per-sample weights derived from classWeights.
  if (classWeights is not None and modelName in SAMPLE_WEIGHT_FIT_MODELS):
    # Compute the per-row sample weight from the class-weight mapping.
    sampleWeight = yTrain.map(classWeights).values
    # Try fitting with sample_weight; fall back silently if this particular estimator rejects it.
    try:
      # Fit the model with sample weights.
      model.fit(xTrain, yTrain, sample_weight=sampleWeight)
    except TypeError:
      # Fall back to an unweighted fit.
      model.fit(xTrain, yTrain)
  else:
    # Fit the model without any sample weighting.
    model.fit(xTrain, yTrain)
  # Return the fitted model.
  return model


# Define the TuneHyperparametersWithOptuna function (Step 8 of the pipeline).
def TuneHyperparametersWithOptuna(
  xTrain: pandas.DataFrame,
  yTrain: pandas.Series,
  xVal: pandas.DataFrame,
  yVal: pandas.Series,
  classWeights: Optional[Dict[int, float]],
  modelType: str,
  nTrials: int,
  randomState: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  # Determine the number of distinct classes in the target.
  nClasses = int(pandas.Series(yTrain).nunique())
  # Determine whether this model family should use the class_weight constructor argument.
  useClassWeight = (classWeights is not None and modelType in CLASS_WEIGHT_PARAM_MODELS)

  # Define the Optuna objective function, maximizing macro-F1 on the validation fold.
  def Objective(trial: optuna.Trial) -> float:
    # Sample this trial's hyperparameters for the requested model family.
    params = SuggestHyperparameters(trial, modelType)
    # Build the unfitted model.
    model = BuildModelFromParams(modelType, params, randomState, useClassWeight, nClasses)
    # Fit the model, applying sample weights if this model family needs them.
    model = FitModel(model, modelType, xTrain, yTrain, classWeights)
    # Generate predictions on the validation fold.
    valPreds = model.predict(xVal)
    # Return the macro-averaged F1 score as the objective value to maximize.
    return f1_score(yVal, valPreds, average="macro")

  # Create the Optuna study using the TPE sampler for reproducibility.
  study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=randomState))
  # Run the optimization for the configured number of trials.
  study.optimize(Objective, n_trials=nTrials, show_progress_bar=False)

  # Build the report summarizing the tuning run.
  report = {
    "NTrials"       : len(study.trials),
    "BestValMacroF1": study.best_value,
    "BestParams"    : study.best_params,
    "TrialValues"   : [t.value for t in study.trials],
  }
  # Return the best hyperparameters and the tuning report.
  return study.best_params, report


# Define the TrainFinalModel function (Step 9a of the pipeline).
def TrainFinalModel(
  xTrainFull: pandas.DataFrame,
  yTrainFull: pandas.Series,
  bestParams: Dict[str, Any],
  classWeights: Optional[Dict[int, float]],
  modelType: str,
  nClasses: int,
  randomState: int,
) -> Any:
  # Determine whether this model family should use the class_weight constructor argument.
  useClassWeight = (classWeights is not None and modelType in CLASS_WEIGHT_PARAM_MODELS)
  # Build the unfitted final model with the tuned hyperparameters.
  model = BuildModelFromParams(modelType, bestParams, randomState, useClassWeight, nClasses)
  # Fit the final model on the combined train+val data.
  model = FitModel(model, modelType, xTrainFull, yTrainFull, classWeights)
  # Return the trained model.
  return model


# Define the ExtractFeatureImportances function: works across every supported model family.
def ExtractFeatureImportances(model: Any, featureNames: List[str]) -> Dict[str, float]:
  # Check for tree-based models that expose feature_importances_ directly.
  if (hasattr(model, "feature_importances_")):
    # Use the native feature importances.
    values = model.feature_importances_
  elif (hasattr(model, "coef_")):
    # Use the mean absolute coefficient magnitude for linear models (handles multiclass coef_ shape).
    coef = model.coef_
    # Average across classes if this is a multiclass coefficient matrix.
    values = numpy.mean(numpy.abs(coef), axis=0) if (coef.ndim > 1) else numpy.abs(coef)
  else:
    # Fall back to zeros for model families with no native importance/coefficient (KNN, GaussianNB, MLP, kernel SVC).
    values = numpy.zeros(len(featureNames))
  # Return the feature-name-to-importance mapping.
  return dict(zip(featureNames, [float(x) for x in values]))


# Define the CalculatePerformanceMetrics function.
def CalculatePerformanceMetrics(
  confMatrix: numpy.ndarray,
  eps: float = 1e-10,
  addWeightedAverage: bool = False,
  addPerClass: bool = False,
) -> Dict[str, Any]:
  # Convert the confusion matrix to a NumPy array for easier manipulation.
  confMatrix = numpy.array(confMatrix)
  # Get the number of classes from the shape of the confusion matrix.
  noOfClasses = confMatrix.shape[0]
  # Check if the confusion matrix is for binary classification or multiclass.
  if (noOfClasses > 2):
    # Calculate True Positives (TP) as the diagonal elements of the confusion matrix.
    truePositives = numpy.diag(confMatrix)
    # Calculate False Positives (FP) as the sum of each column minus the TP.
    falsePositives = numpy.sum(confMatrix, axis=0) - truePositives
    # Calculate False Negatives (FN) as the sum of each row minus the TP.
    falseNegatives = numpy.sum(confMatrix, axis=1) - truePositives
    # Calculate True Negatives (TN) as the total sum of the matrix minus TP, FP, and FN.
    trueNegatives = numpy.sum(confMatrix) - (truePositives + falsePositives + falseNegatives)
  else:
    # For binary classification, the confusion matrix is a 2x2 matrix.
    # Unravel the confusion matrix to get the TN, FP, FN, and TP.
    trueNegatives, falsePositives, falseNegatives, truePositives = confMatrix.ravel()
    # Wrap the scalars into length-2 arrays so the per-class math below stays uniform.
    truePositives = numpy.array([truePositives, truePositives])
    falsePositives = numpy.array([falsePositives, falsePositives])
    falseNegatives = numpy.array([falseNegatives, falseNegatives])
    trueNegatives = numpy.array([trueNegatives, trueNegatives])

  # Add a small epsilon value to avoid division by zero in metric calculations.
  truePositives = truePositives + eps
  falsePositives = falsePositives + eps
  falseNegatives = falseNegatives + eps
  trueNegatives = trueNegatives + eps

  # Create a dictionary to hold the calculated performance metrics and the TP, FP, FN, TN vectors.
  metrics = {
    "TruePositives" : truePositives.tolist(),
    "FalsePositives": falsePositives.tolist(),
    "FalseNegatives": falseNegatives.tolist(),
    "TrueNegatives" : trueNegatives.tolist(),
  }

  # If requested, calculate per-class precision, recall, F1, accuracy, and specificity.
  if (addPerClass):
    # Calculate per-class precision, recall, F1, accuracy, specificity, and balanced accuracy.
    precisionPerClass = truePositives / (truePositives + falsePositives)
    recallPerClass = truePositives / (truePositives + falseNegatives)
    f1PerClass = 2.0 * precisionPerClass * recallPerClass / (precisionPerClass + recallPerClass)
    accuracyPerClass = (truePositives + trueNegatives) / (
      truePositives + trueNegatives + falsePositives + falseNegatives)
    specificityPerClass = trueNegatives / (trueNegatives + falsePositives)
    bacPerClass = 0.5 * (recallPerClass + specificityPerClass)
    # Iterate through each class to add its metrics to the dictionary.
    for i in range(len(precisionPerClass)):
      # Update the metrics dictionary with this class's metrics.
      metrics.update({
        f"Class {i} Precision"  : float(precisionPerClass[i]),
        f"Class {i} Recall"     : float(recallPerClass[i]),
        f"Class {i} F1"         : float(f1PerClass[i]),
        f"Class {i} Accuracy"   : float(accuracyPerClass[i]),
        f"Class {i} Specificity": float(specificityPerClass[i]),
        f"Class {i} BAC"        : float(bacPerClass[i]),
      })

  # Calculate macro-averaged precision, recall, F1, specificity, balanced accuracy, and MCC.
  precision = numpy.mean(truePositives / (truePositives + falsePositives))
  recall = numpy.mean(truePositives / (truePositives + falseNegatives))
  f1 = 2.0 * precision * recall / (precision + recall)
  specificity = numpy.mean(trueNegatives / (trueNegatives + falsePositives))
  bac = 0.5 * (recall + specificity)
  mcc = numpy.mean((truePositives * trueNegatives - falsePositives * falseNegatives) / numpy.sqrt(
    (truePositives + falsePositives) * (truePositives + falseNegatives) *
    (trueNegatives + falsePositives) * (trueNegatives + falseNegatives)))

  # Add macro metrics to the dictionary.
  metrics.update({
    "Macro Precision"  : float(precision),
    "Macro Recall"     : float(recall),
    "Macro F1"         : float(f1),
    "Macro Specificity": float(specificity),
    "Macro BAC"        : float(bac),
    "Macro MCC"        : float(mcc),
  })

  # If requested, calculate the macro average of the main metrics.
  if (addWeightedAverage):
    # Calculate the average of the 5 main macro metrics.
    avg = (precision + recall + f1 + specificity + bac) / 5.0
    # Update the metrics dictionary with the macro average.
    metrics.update({"Macro Average": float(avg)})

  # Calculate the number of samples per class by summing the rows of the confusion matrix.
  samples = numpy.sum(confMatrix, axis=1) if (noOfClasses > 2) else numpy.array(
    [trueNegatives[0] + falsePositives[0], truePositives[0] + falseNegatives[0]])
  # Calculate the weights for each class as the proportion of samples in that class.
  weights = samples / numpy.sum(samples)

  # Calculate weighted-averaged precision, recall, and F1.
  precisionW = numpy.sum(truePositives / (truePositives + falsePositives) * weights)
  recallW = numpy.sum(truePositives / (truePositives + falseNegatives) * weights)
  f1W = 2.0 * precisionW * recallW / (precisionW + recallW)

  # Add weighted metrics to the dictionary.
  metrics.update({
    "Weighted Precision": float(precisionW),
    "Weighted Recall"   : float(recallW),
    "Weighted F1"       : float(f1W),
  })

  # Return the dictionary containing all calculated metrics.
  return metrics


# Define the PlotConfusionMatrix function.
def PlotConfusionMatrix(
  confMatrix: numpy.ndarray,
  classNames: List[str],
  outputDir: Union[str, Path],
  fileName: str = "CM.png",
) -> None:
  # Create a matplotlib figure with appropriate size for readability.
  plt.figure(figsize=(8, 6))
  # Generate the heatmap with annotations using the seaborn library.
  seaborn.heatmap(
    confMatrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=classNames, yticklabels=classNames, cbar_kws={"label": "Count"},
  )
  # Configure the axis labels and title.
  plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
  plt.ylabel("True Label", fontsize=12, fontweight="bold")
  plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
  # Adjust the layout to prevent overlap.
  plt.tight_layout()
  # Save the figure to the specified output directory.
  plt.savefig(Path(outputDir) / fileName, dpi=200, bbox_inches="tight")
  # Close the plot to free memory.
  plt.close()
  # Print a confirmation message for user feedback.
  fprint(f"Confusion matrix saved to {Path(outputDir) / fileName}")


# Define the PlotClassDistribution function.
def PlotClassDistribution(
  splitLabels: Dict[str, pandas.Series],
  classNames: List[str],
  outputDir: Union[str, Path],
) -> None:
  # Determine the number of classes.
  numClasses = len(classNames)
  # Bail out early if there are no classes to plot.
  if (numClasses == 0):
    # Return without plotting.
    return
  # Create a figure for the histograms.
  plt.figure(figsize=(max(6, numClasses * 0.8), 5))
  # Define the bar width and base positions.
  barWidth = 0.25
  basePositions = numpy.arange(numClasses)
  # Initialize the split index used to offset each split's bars.
  splitIdx = 0
  # Iterate through each split's label series.
  for splitName, labels in splitLabels.items():
    # Count the occurrences of each class in this split.
    counts = numpy.bincount(labels.astype(int), minlength=numClasses)
    # Compute this split's bar positions.
    positions = [x + splitIdx * barWidth for x in basePositions]
    # Plot the bars for this split, labeling with the split's total sample count.
    plt.bar(positions, counts, width=barWidth, edgecolor="grey", label=f"{splitName} (N={len(labels)})")
    # Advance the split index.
    splitIdx += 1
  # Configure the axis ticks, labels, title, and legend.
  plt.xticks([r + barWidth for r in range(numClasses)], classNames, rotation=45, ha="right")
  plt.xlabel("Class", fontsize=12, fontweight="bold")
  plt.ylabel("Count", fontsize=12, fontweight="bold")
  plt.title("Class Distribution Across Splits", fontsize=14, fontweight="bold")
  plt.legend()
  # Adjust the layout and save the figure.
  plt.tight_layout()
  plt.savefig(Path(outputDir) / "ClassDistribution.png", dpi=200, bbox_inches="tight")
  plt.close()
  # Print a confirmation message.
  fprint(f"Class distribution plot saved to {Path(outputDir) / 'ClassDistribution.png'}")


# Define the PlotOptunaHistory function.
def PlotOptunaHistory(trialValues: List[float], outputDir: Union[str, Path]) -> None:
  # Create a matplotlib figure for the optimization history.
  plt.figure(figsize=(8, 5))
  # Plot the validation macro-F1 achieved by each trial.
  plt.plot(trialValues, marker="o", linewidth=1.5, markersize=4)
  # Plot the running best-so-far value as a reference line.
  runningBest = numpy.maximum.accumulate(trialValues)
  plt.plot(runningBest, linewidth=2, linestyle="--", label="Best so far")
  # Configure the axis labels, title, legend, and grid.
  plt.xlabel("Trial", fontsize=12)
  plt.ylabel("Validation Macro F1", fontsize=12)
  plt.title("Optuna Hyperparameter Search History", fontsize=14, fontweight="bold")
  plt.legend()
  plt.grid(True, alpha=0.3)
  # Adjust the layout and save the figure.
  plt.tight_layout()
  plt.savefig(Path(outputDir) / "OptunaHistory.png", dpi=200, bbox_inches="tight")
  plt.close()
  # Print a confirmation message.
  fprint(f"Optuna history plot saved to {Path(outputDir) / 'OptunaHistory.png'}")


# Define the PlotFeatureImportance function.
def PlotFeatureImportance(featureImportances: Dict[str, float], outputDir: Union[str, Path], topN: int = 20) -> None:
  # Sort the feature importances descending and keep the top N.
  sortedItems = sorted(featureImportances.items(), key=lambda kv: -kv[1])[:topN]
  # Split the sorted items into separate name and value lists (reversed for a top-to-bottom barh).
  names = [k for k, _ in sortedItems][::-1]
  values = [v for _, v in sortedItems][::-1]
  # Create a matplotlib figure sized to the number of bars.
  plt.figure(figsize=(8, max(4, len(names) * 0.3)))
  # Plot the horizontal bar chart.
  plt.barh(names, values, color="steelblue")
  # Configure the axis label and title.
  plt.xlabel("Importance", fontsize=12)
  plt.title(f"Top {len(names)} Feature Importances", fontsize=14, fontweight="bold")
  # Adjust the layout and save the figure.
  plt.tight_layout()
  plt.savefig(Path(outputDir) / "FeatureImportance.png", dpi=200, bbox_inches="tight")
  plt.close()
  # Print a confirmation message.
  fprint(f"Feature importance plot saved to {Path(outputDir) / 'FeatureImportance.png'}")


# Define the EvaluateModel function.
def EvaluateModel(
  model: Any,
  xSplit: pandas.DataFrame,
  ySplit: pandas.Series,
  rowIds: pandas.Series,
  classNames: List[str],
  outputDir: Union[str, Path],
  prefix: str = "Test",
  savePlots: bool = True,
) -> Dict[str, Any]:
  # Print the evaluation start message.
  fprint(f"Starting evaluation: prefix={prefix} | outputDir={outputDir}")
  # Create a folder for this split's evaluation results.
  evalOutputDir = Path(outputDir) / prefix
  evalOutputDir.mkdir(parents=True, exist_ok=True)

  # Generate the predicted class labels and probabilities.
  preds = model.predict(xSplit)
  predProbs = model.predict_proba(xSplit)

  # Build the sorted list of label ids present in this split, and the confusion matrix.
  labelsSorted = sorted(pandas.Series(ySplit).unique().tolist())
  confMatrix = confusion_matrix(ySplit, preds, labels=labelsSorted)
  targetNames = [classNames[i] for i in labelsSorted]
  # Calculate the extended performance metrics from the confusion matrix.
  pmMetrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True, addPerClass=True)
  # Generate the detailed sklearn classification report.
  classReport = classification_report(ySplit, preds, labels=labelsSorted, target_names=targetNames, zero_division=0)

  # Compute the standard summary metrics used for cross-model comparison.
  summaryMetrics = {
    "Accuracy"         : float(accuracy_score(ySplit, preds)),
    "PrecisionMacro"   : float(precision_score(ySplit, preds, average="macro", zero_division=0)),
    "RecallMacro"      : float(recall_score(ySplit, preds, average="macro", zero_division=0)),
    "F1Macro"          : float(f1_score(ySplit, preds, average="macro", zero_division=0)),
    "PrecisionWeighted": float(precision_score(ySplit, preds, average="weighted", zero_division=0)),
    "RecallWeighted"   : float(recall_score(ySplit, preds, average="weighted", zero_division=0)),
    "F1Weighted"       : float(f1_score(ySplit, preds, average="weighted", zero_division=0)),
    "LogLoss"          : float(log_loss(ySplit, predProbs, labels=model.classes_)),
  }
  # Attempt to compute the ROC-AUC score (binary vs. multiclass one-vs-rest).
  try:
    # Check whether this is a binary or multiclass problem.
    if (len(labelsSorted) == 2):
      # Compute the binary ROC-AUC using the positive class probability.
      summaryMetrics["RocAuc"] = float(roc_auc_score(ySplit, predProbs[:, 1]))
    else:
      # Compute the multiclass macro-averaged one-vs-rest ROC-AUC.
      summaryMetrics["RocAucOvrMacro"] = float(roc_auc_score(ySplit, predProbs, multi_class="ovr", average="macro"))
  except Exception as e:
    # Record the reason ROC-AUC could not be computed for this split.
    summaryMetrics["RocAucError"] = str(e)

  # Save the confusion matrix plot if visualization is requested.
  if (savePlots):
    # Call the function to plot the confusion matrix.
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
    for rowId, actual, pred, predProb in zip(rowIds, ySplit, preds, predProbs):
      csvWriter.writerow([prefix, rowId, int(actual), int(pred), predProb.tolist()])

  # Print a one-line summary for quick console feedback.
  fprint(f"  {prefix}: accuracy={summaryMetrics['Accuracy']:.4f} f1Macro={summaryMetrics['F1Macro']:.4f}")
  # Return the metrics dictionary.
  return metrics


# Define the RunCompletePipeline function.
def RunCompletePipeline(
  datasetCfg: Dict[str, Any],
  modelType: str,
  scalerName: str,
  imbalanceMethod: str,
  outlierMethod: str,
  outputDir: str,
  config: Dict[str, Any],
) -> Dict[str, Any]:
  # Record the start time for the runtime report.
  startTime = time.time()
  # Print the pipeline initialization message.
  fprint(f"\n{'=' * 70}")
  fprint(f"Starting Experiment: Dataset={datasetCfg['Name']} | Model={modelType} | "
         f"Scaler={scalerName} | Imbalance={imbalanceMethod} | Outlier={outlierMethod}")
  fprint(f"Output Directory: {outputDir}")
  fprint(f"{'=' * 70}")
  # Create the output directory structure.
  Path(outputDir).mkdir(parents=True, exist_ok=True)

  # Resolve the dataset (either a built-in sport pipeline or an arbitrary CSV) into a raw dataframe.
  fprint("Building dataset...")
  df, targetCol, extraDrop, catCols = ResolveDataset(datasetCfg)
  fprint(f"Raw dataset shape: {df.shape}")

  # Drop rows with a missing target value.
  df = df.dropna(subset=[targetCol]).reset_index(drop=True)
  # Always label-encode the target to contiguous 0..n-1 integers (required by XGBoost's multiclass objective).
  labelEncoder = LabelEncoder()
  df[targetCol] = labelEncoder.fit_transform(df[targetCol].astype(str))
  # Assign a stable row id (used later for the detailed predictions CSV) before any filtering.
  df["RowId"] = numpy.arange(len(df))

  # Record the raw target distribution for the report.
  rawTargetDistribution = {str(k): int(v) for k, v in df[targetCol].value_counts().to_dict().items()}

  # Step 1: drop unnecessary columns.
  fprint("Step 1: DropUnnecessaryColumns")
  df, unnecessaryReport = DropUnnecessaryColumns(df, targetCol, explicitDropCols=extraDrop)

  # Step 2: remove outliers (method selected via config).
  fprint(f"Step 2: RemoveOutliers ({outlierMethod})")
  df, outlierReport = RemoveOutliers(
    df, targetCol, method=outlierMethod,
    iqrMultiplier=config.get("OutlierIqrMultiplier", 3.0),
    zScoreThreshold=config.get("OutlierZScoreThreshold", 3.0),
    contamination=config.get("OutlierContamination", 0.05),
    maxRowDropFrac=config.get("OutlierMaxRowDropFrac", 0.15),
    randomState=config.get("RandomState", 42),
  )

  # Step 3: split into train/val/test.
  fprint("Step 3: SplitData")
  trainDf, valDf, testDf, splitReport = SplitData(
    df, targetCol,
    testSize=config.get("TestSize", 0.10), valSize=config.get("ValSize", 0.10),
    randomState=config.get("RandomState", 42),
  )

  # Step 4: drop redundant columns.
  fprint("Step 4: DropRedundantColumns")
  trainDf, valDf, testDf, redundantReport = DropRedundantColumns(
    trainDf, valDf, testDf, targetCol, correlationThreshold=config.get("CorrelationThreshold", 0.95))

  # Encode the categorical columns (one-hot, fit on train, aligned to val/test).
  trainDf, valDf, testDf, newCatCols = EncodeCategoricalColumns(trainDf, valDf, testDf, catCols)

  # Determine the final numeric feature columns (excluding the target and the row id).
  featureCols = [
    c for c in trainDf.columns
    if (c not in (targetCol, "RowId") and pandas.api.types.is_numeric_dtype(trainDf[c]))
  ]
  # Fill any remaining missing feature values with the *training-fold* median, applied
  # identically to Val/Test (falling back to 0 for a column that's entirely missing in
  # training). Each split computing its own median here was a leakage bug - a split's own
  # distribution should never inform how its own missing values get filled - so this now
  # mirrors NormalizeFeatures, which already correctly fits only on the training fold.
  trainMedians = trainDf[featureCols].median().fillna(0)
  for splitDf in (trainDf, valDf, testDf):
    splitDf[featureCols] = splitDf[featureCols].fillna(trainMedians)

  # Extract the feature matrices, target vectors, and row ids for each split.
  xTrain, yTrain = trainDf[featureCols], trainDf[targetCol]
  xVal, yVal = valDf[featureCols], valDf[targetCol]
  xTest, yTest = testDf[featureCols], testDf[targetCol]
  rowIdsTrain, rowIdsVal, rowIdsTest = trainDf["RowId"], valDf["RowId"], testDf["RowId"]

  # Step 5: handle class imbalance (training fold only, method selected via config).
  fprint(f"Step 5: HandleClassImbalance ({imbalanceMethod})")
  xTrainBal, yTrainBal, classWeights, imbalanceReport = HandleClassImbalance(
    xTrain, yTrain, imbalanceMethod, randomState=config.get("RandomState", 42))
  # Reset the resampled training indices (a resampler returns a fresh 0..n-1 index).
  xTrainBal = pandas.DataFrame(xTrainBal, columns=featureCols).reset_index(drop=True)
  yTrainBal = pandas.Series(yTrainBal).reset_index(drop=True)

  # Step 6: normalize the features (scaler selected via config).
  fprint(f"Step 6: NormalizeFeatures ({scalerName})")
  xTrainScaled, xValScaled, xTestScaled, scaler = NormalizeFeatures(
    xTrainBal, xVal, xTest, scalerName, randomState=config.get("RandomState", 42))

  # Step 7: feature selection via the 3-method vote.
  fprint("Step 7: SelectFeatures")
  selectedFeatures, selectionReport = SelectFeatures(
    xTrainScaled, yTrainBal,
    topKFraction=config.get("FeatureSelectionTopKFrac", 0.6),
    minFeaturesSelected=config.get("MinFeaturesSelected", 4),
    randomState=config.get("RandomState", 42),
  )
  # Restrict every split to the selected features.
  xTrainFs, xValFs, xTestFs = xTrainScaled[selectedFeatures], xValScaled[selectedFeatures], xTestScaled[
    selectedFeatures]

  # Step 8: hyperparameter tuning via Optuna (model family selected via config).
  fprint(f"Step 8: TuneHyperparametersWithOptuna ({modelType}, {config.get('NOptunaTrials', 25)} trials)")
  bestParams, tuningReport = TuneHyperparametersWithOptuna(
    xTrainFs, yTrainBal, xValFs, yVal, classWeights, modelType,
    nTrials=config.get("NOptunaTrials", 25), randomState=config.get("RandomState", 42),
  )
  # Save the Optuna trial-history plot.
  PlotOptunaHistory(tuningReport["TrialValues"], outputDir)

  # Step 9: train the final model on train+val and evaluate on every split.
  fprint("Step 9: TrainFinalModel + EvaluateModel")
  xTrainFull = pandas.concat([xTrainFs, xValFs], axis=0).reset_index(drop=True)
  yTrainFull = pandas.concat(
    [pandas.Series(yTrainBal), pandas.Series(yVal).reset_index(drop=True)], axis=0).reset_index(drop=True)
  nClasses = int(pandas.concat([pandas.Series(yTrainBal), pandas.Series(yVal), pandas.Series(yTest)]).nunique())
  finalModel = TrainFinalModel(
    xTrainFull, yTrainFull, bestParams, classWeights, modelType, nClasses,
    randomState=config.get("RandomState", 42))

  # Build the list of human-readable class names in label-encoded order.
  classNames = [str(c) for c in labelEncoder.classes_]

  # Save the class distribution plot across the three splits.
  PlotClassDistribution({"Train": yTrainBal, "Val": yVal, "Test": yTest}, classNames, outputDir)

  # Evaluate the final model on every split.
  testMetrics = EvaluateModel(finalModel, xTestFs, yTest, rowIdsTest, classNames, outputDir, prefix="Test")
  valMetrics = EvaluateModel(finalModel, xValFs, yVal, rowIdsVal, classNames, outputDir, prefix="Val")
  trainMetrics = EvaluateModel(finalModel, xTrainFs, yTrainBal, rowIdsTrain, classNames, outputDir, prefix="Train")

  # Extract the final model's feature importances (works across every supported model family).
  featureImportances = ExtractFeatureImportances(finalModel, selectedFeatures)
  # Save the feature importance plot.
  PlotFeatureImportance(featureImportances, outputDir)

  # Persist the trained model, its feature list, scaler, and label encoder for later inference.
  modelPath = Path(outputDir) / "BestModel.joblib"
  joblib.dump(
    {"Model": finalModel, "Features": selectedFeatures, "LabelEncoder": labelEncoder, "Scaler": scaler}, modelPath)

  # Assemble the full experiment result dictionary covering every pipeline step.
  result = {
    "Dataset"                  : datasetCfg["Name"],
    "ModelType"                : modelType,
    "ScalerName"                : scalerName,
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
    "FeatureImportances"       : featureImportances,
    "ModelCheckpointPath"      : str(modelPath),
    "RuntimeSeconds"           : round(time.time() - startTime, 2),
  }
  # Save the full experiment result to a single JSON file for downstream aggregation.
  with open(Path(outputDir) / "FullExperimentResult.json", "w") as f:
    json.dump(result, f, indent=2, cls=NumpyEncoder)

  # Print the pipeline completion summary for user confirmation.
  fprint(
    f"Pipeline Complete | {datasetCfg['Name']}/{modelType}/{scalerName}/{imbalanceMethod}/{outlierMethod} "
    f"| Test Accuracy={testMetrics['Accuracy']:.4f} | Test F1Macro={testMetrics['F1Macro']:.4f} "
    f"| {result['RuntimeSeconds']}s | Results saved to {outputDir}"
  )
  # Return the full result dictionary.
  return result


# Define the main execution block.
if (__name__ == "__main__"):
  # Import the argparse module for command-line interface.
  import argparse

  # Initialize the argument parser for command-line interface.
  parser = argparse.ArgumentParser(description="Tabular ML Pipeline (bring your own CSV, or use the built-in UFC/basketball datasets)")
  # Add the command-line argument for the configuration file.
  parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML or JSON configuration file")
  # Parse the command-line arguments into a namespace object.
  args = parser.parse_args()
  # Load the configuration from the file.
  config = LoadConfig(args.config)

  # Extract the base output directory for the experiment grid.
  baseOutputDir = str(Path(config.get("OutputDir", "Results")))

  # Extract the datasets list from the configuration (each entry is a dict; see config.yaml).
  datasets = config.get("Datasets", [])
  # Extract the grid dimensions, defaulting to one sensible value each if omitted.
  modelTypes = config.get("Models", ["XGBoost"])
  scalers = config.get("Scalers", ["Standard"])
  imbalanceMethods = config.get("ImbalanceMethod", ["SMOTE"])
  outlierMethods = config.get("OutlierMethod", ["IQR"])

  # Ensure every grid dimension is a list, for uniform iteration.
  if (isinstance(modelTypes, str)):
    modelTypes = [modelTypes]
  if (isinstance(scalers, str)):
    scalers = [scalers]
  if (isinstance(imbalanceMethods, str)):
    imbalanceMethods = [imbalanceMethods]
  if (isinstance(outlierMethods, str)):
    outlierMethods = [outlierMethods]

  # Bail out early with a clear message if no datasets were configured.
  if (len(datasets) == 0):
    # Print the guidance message.
    fprint("No datasets configured. Add at least one entry under 'Datasets:' in your config.yaml.")
  else:
    # Loop through every combination of dataset, model, scaler, imbalance method, and outlier method.
    for datasetCfg in datasets:
      for modelType in modelTypes:
        for scalerName in scalers:
          for imbalanceMethod in imbalanceMethods:
            for outlierMethod in outlierMethods:
              # Construct the dynamic experiment output directory name.
              experimentName = (
                f"Exp-{datasetCfg['Name']}-{modelType}-{scalerName}-{imbalanceMethod}-{outlierMethod}")
              # Build the full output directory path.
              currentOutputDir = str(Path(baseOutputDir) / experimentName)

              # Run the experiment inside a try/except so one failure does not abort the whole grid.
              try:
                # Execute the complete pipeline for this combination.
                RunCompletePipeline(
                  datasetCfg=datasetCfg, modelType=modelType, scalerName=scalerName,
                  imbalanceMethod=imbalanceMethod, outlierMethod=outlierMethod,
                  outputDir=currentOutputDir, config=config,
                )
                # Save a copy of the configuration used for this run, for reproducibility.
                configCopy = dict(config)
                configCopy.update({
                  "Dataset": datasetCfg["Name"], "ModelType": modelType, "ScalerName": scalerName,
                  "ImbalanceMethod": imbalanceMethod, "OutlierMethod": outlierMethod,
                })
                with open(Path(currentOutputDir) / "ConfigUsed.yaml", "w") as f:
                  yaml.dump(configCopy, f)
              except Exception as e:
                # Print the error message for this failed experiment.
                fprint(f"ERROR in experiment {experimentName}: {e}")
                # Import the traceback module for detailed error reporting.
                import traceback
                traceback.print_exc()
                fprint("Continuing to the next experiment...")
                continue

    # Print a separator line and the final completion message.
    fprint("\n" + "=" * 70)
    fprint("All experiments completed.")
