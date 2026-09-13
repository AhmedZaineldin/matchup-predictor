import json
from pathlib import Path

import joblib
import numpy
import pandas
import yaml

# These two are the actual pipeline modules one directory up - the predictor app is meant to run
# from inside LocalPipeline/PredictorApp/ with LocalPipeline/ on the Python path (see app.py).
import TabularMLPipeline as tp
import DeepLearningPipeline as dl


def fprint(msg):
  # Print immediately, matching the rest of the pipeline's logging style.
  print(msg, flush=True)


class LoadedModel:
  """A uniform wrapper around a classical, from-scratch-DL, or pretrained-DL model, so the Flask
  app can call .Predict(rawFeatureRow) the same way regardless of which kind of model it is."""

  def __init__(self, kind, features, labelEncoder, scaler, catCols, predictProbaFn):
    self.kind = kind
    self.features = features
    self.labelEncoder = labelEncoder
    self.scaler = scaler
    self.catCols = catCols
    self.predictProbaFn = predictProbaFn

  # Encode + scale + select a single raw feature dict, exactly reproducing what training did to
  # every row, WITHOUT needing a saved one-hot encoder object: pandas.get_dummies is deterministic
  # given the same categorical columns and dummy_na=True, and scaler.feature_names_in_ (available
  # on every scikit-learn scaler fit on a DataFrame, since scikit-learn>=1.0) records the exact
  # full column set + order the scaler was fit on - so reindexing onto that list and filling
  # missing dummy columns with 0 reconstructs training-time one-hot encoding exactly.
  #
  # IMPORTANT: do NOT force categorical columns to object dtype here. pandas.get_dummies(...,
  # dummy_na=True) silently coerces a numeric-dtype column to float before naming its dummy
  # columns - e.g. an int64 "season_type" column with values 1/2/3/5 produces columns named
  # "season_type_1.0", not "season_type_1" - even though training's data had NO missing values
  # in that column at all. TASK_REGISTRY's CatCols list includes some numeric-coded columns
  # (season_type), so training's trainDf hit this exact quirk and its scaler was fit on the
  # "_1.0"-style names. Forcing this single row's value to object dtype beforehand (an earlier
  # version of this function did that, to make a lone NaN one-hot correctly) skips that
  # coercion and produces "season_type_1" instead - which then NEVER matches any trained
  # dummy column, silently zeroing out that feature's signal on every prediction. Leaving each
  # column at whatever dtype pandas naturally infers from the raw Python value (str -> object,
  # int/float -> numeric, None/NaN -> float64) reproduces training's dtype exactly, and
  # pandas.get_dummies(..., columns=[...]) one-hot-encodes a listed column correctly regardless
  # of its dtype - including a lone-NaN numeric column, which still yields a "..._nan" column.
  def EncodeRow(self, rawRow: dict) -> pandas.DataFrame:
    df = pandas.DataFrame([rawRow])
    presentCatCols = [c for c in self.catCols if (c in df.columns)]
    dfEnc = pandas.get_dummies(df, columns=presentCatCols, dummy_na=True)
    fullCols = list(self.scaler.feature_names_in_)
    dfEnc = dfEnc.reindex(columns=fullCols, fill_value=0)
    scaledArr = self.scaler.transform(dfEnc)
    scaledDf = pandas.DataFrame(scaledArr, columns=fullCols)
    return scaledDf[self.features]

  def Predict(self, rawRow: dict):
    encoded = self.EncodeRow(rawRow)
    probs = numpy.asarray(self.predictProbaFn(encoded)).reshape(-1)
    classes = [str(c) for c in self.labelEncoder.classes_]
    probsByClass = {cls: float(p) for cls, p in zip(classes, probs)}
    predictedLabel = classes[int(numpy.argmax(probs))]
    return predictedLabel, probsByClass


# Define the function that refits a pretrained (TabPFN/TabICL/SapRpt1Oss) model at app startup.
# These models have no unique-to-this-run weights to reload (see DeepLearningPipeline.py's own
# comment on this) - "reloading" one means re-running Steps 1-7 of the pipeline with the exact
# settings that experiment used (recorded in ConfigUsed.yaml + FullExperimentResult.json) and
# fitting fresh, which is cheap since these models do no gradient training.
def RefitPretrainedModel(datasetName: str, architecture: str, bestParams: dict, folder: Path):
  configPath, resultPath = folder / "ConfigUsed.yaml", folder / "FullExperimentResult.json"
  if (not configPath.exists() or not resultPath.exists()):
    raise FileNotFoundError(
      f"Reloading the pretrained model for {datasetName} needs ConfigUsed.yaml AND "
      f"FullExperimentResult.json inside {folder} (both are produced automatically alongside "
      f"ModelMetadata.joblib) - copy that experiment's WHOLE output folder, not individual files.")
  config = yaml.safe_load(open(configPath))
  result = json.load(open(resultPath))
  scalerName, imbalanceMethod, outlierMethod = (
    result["ScalerName"], result["ImbalanceMethod"], result["OutlierMethod"])
  randomState = config.get("RandomState", 42)

  fprint(f"  Refitting pretrained model ({architecture}) for {datasetName} - this happens once at startup...")
  datasetCfg = {"Name": datasetName, "Builtin": datasetName}
  df, targetCol, extraDrop, catCols = tp.ResolveDataset(datasetCfg)
  df = df.dropna(subset=[targetCol]).reset_index(drop=True)
  labelEncoder = tp.LabelEncoder()
  df[targetCol] = labelEncoder.fit_transform(df[targetCol].astype(str))
  df["RowId"] = numpy.arange(len(df))

  df, _ = tp.DropUnnecessaryColumns(df, targetCol, explicitDropCols=extraDrop)
  df, _ = tp.RemoveOutliers(
    df, targetCol, method=outlierMethod,
    iqrMultiplier=config.get("OutlierIqrMultiplier", 3.0),
    zScoreThreshold=config.get("OutlierZScoreThreshold", 3.0),
    contamination=config.get("OutlierContamination", 0.05),
    maxRowDropFrac=config.get("OutlierMaxRowDropFrac", 0.15), randomState=randomState,
  )
  trainDf, valDf, testDf, _ = tp.SplitData(
    df, targetCol, testSize=config.get("TestSize", 0.10), valSize=config.get("ValSize", 0.10),
    randomState=randomState)
  trainDf, valDf, testDf, _ = tp.DropRedundantColumns(
    trainDf, valDf, testDf, targetCol, correlationThreshold=config.get("CorrelationThreshold", 0.95))
  trainDf, valDf, testDf, _ = tp.EncodeCategoricalColumns(trainDf, valDf, testDf, catCols)

  featureCols = [
    c for c in trainDf.columns
    if (c not in (targetCol, "RowId") and pandas.api.types.is_numeric_dtype(trainDf[c]))
  ]
  trainMedians = trainDf[featureCols].median().fillna(0)
  for splitDf in (trainDf, valDf, testDf):
    splitDf[featureCols] = splitDf[featureCols].fillna(trainMedians)

  xTrain, yTrain = trainDf[featureCols], trainDf[targetCol]
  xVal = valDf[featureCols]
  xTrainBal, yTrainBal, _, _ = tp.HandleClassImbalance(xTrain, yTrain, imbalanceMethod, randomState=randomState)
  xTrainBal = pandas.DataFrame(xTrainBal, columns=featureCols).reset_index(drop=True)
  yTrainBal = pandas.Series(yTrainBal).reset_index(drop=True)

  xTrainScaled, _, _, scaler = tp.NormalizeFeatures(xTrainBal, xVal, xVal, scalerName, randomState=randomState)
  selectedFeatures, _ = tp.SelectFeatures(
    xTrainScaled, yTrainBal, topKFraction=config.get("FeatureSelectionTopKFrac", 0.6),
    minFeaturesSelected=config.get("MinFeaturesSelected", 4), randomState=randomState)
  xTrainFs = xTrainScaled[selectedFeatures]

  xTrainCtx, yTrainCtx = dl.SubsampleForPretrainedContext(xTrainFs, yTrainBal, architecture, config)
  model = dl.BuildPretrainedModel(architecture, bestParams, randomState)
  model.fit(xTrainCtx, yTrainCtx)
  fprint(f"  Refit complete for {datasetName} ({architecture}).")
  return model, selectedFeatures, scaler, labelEncoder, catCols


# Define the function that loads whichever "best model" folder the user has placed for one
# dataset. See PredictorApp/README.md for the exact expected folder layout.
def LoadModelForDataset(datasetName: str, bestModelsDir: Path) -> LoadedModel:
  folder = Path(bestModelsDir) / datasetName
  if (not folder.is_dir()):
    raise FileNotFoundError(f"No folder found at {folder} - see PredictorApp/README.md for the expected layout.")

  catCols = tp.TASK_REGISTRY[datasetName]["CatCols"] if (datasetName in tp.TASK_REGISTRY) else []
  classicalPath = folder / "BestModel.joblib"
  metadataPath = folder / "ModelMetadata.joblib"

  if (classicalPath.exists()):
    # ---- Classical model: everything needed is in this single self-contained file. ----
    bundle = joblib.load(classicalPath)
    model = bundle["Model"]
    return LoadedModel(
      kind="classical", features=bundle["Features"], labelEncoder=bundle["LabelEncoder"],
      scaler=bundle["Scaler"], catCols=catCols, predictProbaFn=model.predict_proba)

  if (metadataPath.exists()):
    meta = joblib.load(metadataPath)
    architecture = meta["Architecture"]

    if (architecture in ("MLP", "TabTransformer")):
      # ---- From-scratch deep-learning model: reconstruct the network, load its weights. ----
      import torch
      ptPath = folder / "BestModel.pt"
      if (not ptPath.exists()):
        raise FileNotFoundError(f"{metadataPath} says Architecture={architecture}, which needs a "
                                 f"BestModel.pt in {folder} too, but none was found.")
      numClasses = len(meta["LabelEncoder"].classes_)
      model = dl.BuildModel(architecture, len(meta["Features"]), numClasses, meta["BestParams"])
      model.load_state_dict(torch.load(ptPath, map_location="cpu"))
      model.eval()

      def PredictProba(xDf, _model=model):
        with torch.no_grad():
          logits = _model(torch.tensor(xDf.values, dtype=torch.float32))
          return torch.softmax(logits, dim=1).numpy()

      return LoadedModel(
        kind="dl", features=meta["Features"], labelEncoder=meta["LabelEncoder"],
        scaler=meta["Scaler"], catCols=catCols, predictProbaFn=PredictProba)

    if (architecture in dl.PRETRAINED_ARCHITECTURES):
      # ---- Pretrained foundation model: refit once, now, at startup. ----
      model, selectedFeatures, scaler, labelEncoder, refitCatCols = RefitPretrainedModel(
        datasetName, architecture, meta["BestParams"], folder)
      return LoadedModel(
        kind="pretrained", features=selectedFeatures, labelEncoder=labelEncoder,
        scaler=scaler, catCols=refitCatCols, predictProbaFn=model.predict_proba)

    raise ValueError(f"Unrecognized architecture '{architecture}' in {metadataPath}.")

  raise FileNotFoundError(
    f"Neither BestModel.joblib nor ModelMetadata.joblib found in {folder} - "
    f"see PredictorApp/README.md for the expected folder layout.")
