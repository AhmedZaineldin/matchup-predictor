import numpy  # Import numpy for numeric helpers.
import pandas  # Import pandas for the snapshot medians.

# This module answers "why did the model say that?" for a single live prediction.
#
# THE METHOD: occlusion (also called ablation). Take the real matchup row, replace one group of
# related features with league-typical values, re-predict, and see how far the probability of the
# originally-predicted class moved. A big drop means that group was doing a lot of the work; a
# rise means it was actually arguing against the prediction. The contribution is reported as a
# straight percentage-point swing, so the numbers are in the same units as the bars above them.
#
# WHY OCCLUSION AND NOT SHAP: this app deliberately supports three very different kinds of model
# behind one .Predict() interface - a classical scikit-learn estimator, a from-scratch torch MLP
# or TabTransformer, and a refit pretrained foundation model. SHAP would need a different
# explainer for each (TreeExplainer / DeepExplainer / KernelExplainer), a new dependency, and in
# the kernel case far more model calls than this. Occlusion goes through the exact same
# .Predict(row) path the real prediction used, so it works identically for every model type and
# can never disagree with the model about what the model actually does.
#
# WHY GROUPS AND NOT SINGLE COLUMNS: the raw feature row is deliberately redundant - a fighter's
# reach appears as f1reachIn, f2reachIn AND reachDiff. Blanking any one of those alone barely
# moves the prediction, because the other two still carry the same signal, so per-column
# occlusion would report that reach "doesn't matter" for every model that uses it. Neutralizing
# the whole group at once measures what that piece of information is worth, and collapses ~96
# raw columns into ~17 drivers a person can actually read.

# How many drivers the UI shows by default.
TOP_DRIVER_COUNT = 6
# Contributions smaller than this (in percentage points) are noise, not signal, and are dropped.
MIN_CONTRIBUTION_PP = 0.05


# ---- UFC driver groups -------------------------------------------------------------------
# Each entry is (driver label, how to describe the underlying values, [raw feature name...]).
# The feature names must match feature_builder.BuildUfcMatchupRow()'s output exactly.
def _UfcStatGroup(label, statLabel, unitSuffix=""):
  # Build the group for one in-fight performance stat: both fighters' career and last-5
  # averages plus the two difference columns the model also sees.
  lowerFirst = f"{statLabel[0].lower()}{statLabel[1:]}"
  return (label, ("f1", f"{statLabel}Entering", unitSuffix), [
    f"f1{statLabel}Entering", f"f2{statLabel}Entering",
    f"f1{statLabel}Roll5", f"f2{statLabel}Roll5",
    f"{lowerFirst}Diff", f"{lowerFirst}Roll5Diff",
  ])


UFC_DRIVER_GROUPS = [
  ("Reach", ("f1", "reachIn", " in"), ["f1reachIn", "f2reachIn", "reachDiff"]),
  ("Height", ("f1", "heightIn", " in"), ["f1heightIn", "f2heightIn", "heightDiff"]),
  # Weight has no diff column in the matchup row (weight class carries that information), but the
  # two absolute values still reach the model and must be attributed to something - without this
  # group they would be the only raw features no driver accounts for, quietly leaking influence.
  ("Weight", ("f1", "weightLbs", " lbs"), ["f1weightLbs", "f2weightLbs"]),
  ("Age", ("f1", "AgeAtFight", " yrs"), ["f1AgeAtFight", "f2AgeAtFight", "ageDiff"]),
  ("Win rate", ("f1", "WinPctEntering", "%"), [
    "f1WinPctEntering", "f2WinPctEntering", "winPctDiff",
    "f1WinsEntering", "f2WinsEntering", "f1LossesEntering", "f2LossesEntering"]),
  ("UFC experience", ("f1", "FightsEntering", " fights"), [
    "f1FightsEntering", "f2FightsEntering", "experienceDiff"]),
  ("Current streak", ("f1", "StreakEntering", ""), [
    "f1StreakEntering", "f2StreakEntering", "streakDiff"]),
  ("Finishing tendency", ("f1", "FinishRateEntering", "%"), [
    "f1FinishRateEntering", "f2FinishRateEntering", "finishRateDiff",
    "f1KoWinRateEntering", "f2KoWinRateEntering", "koWinRateDiff",
    "f1SubWinRateEntering", "f2SubWinRateEntering", "subWinRateDiff",
    "f1AvgRoundEntering", "f2AvgRoundEntering", "avgRoundDiff"]),
  ("Layoff & activity", ("f1", "DaysSinceLastFightEntering", " days"), [
    "f1DaysSinceLastFightEntering", "f2DaysSinceLastFightEntering", "layoffDiff",
    "f1ActivityRateEntering", "f2ActivityRateEntering", "activityRateDiff"]),
  _UfcStatGroup("Striking volume", "SigStrLanded", "/fight"),
  _UfcStatGroup("Striking accuracy", "SigStrAcc", "%"),
  _UfcStatGroup("Strikes absorbed", "SigStrAbsorbed", "/fight"),
  _UfcStatGroup("Takedowns landed", "TdLanded", "/fight"),
  _UfcStatGroup("Takedown accuracy", "TdAcc", "%"),
  _UfcStatGroup("Control time", "CtrlSeconds", "s/fight"),
  _UfcStatGroup("Knockdowns", "Kd", "/fight"),
  _UfcStatGroup("Submission attempts", "SubAtt", "/fight"),
  ("Stance matchup", ("f1", "stance", ""), ["f1stance", "f2stance", "stanceMatchup"]),
  ("Bout context", None, ["weightClassClean", "isTitleFight", "scheduledRounds"]),
]

# The basketball box-score stats worth surfacing as named drivers, in the order they read best.
# Everything not named here is swept into the catch-all group at the end.
# The third element names the stat whose last-10-game average is shown as the driver's evidence,
# so every basketball driver carries real numbers instead of a bare label. The unit suffix says
# what the reader is looking at.
BASKETBALL_NAMED_STATS = [
  ("Scoring", ["team_score"], ("team_scoreRoll10", " pts")),
  ("Points allowed", ["opponent_team_score"], ("opponent_team_scoreRoll10", " pts")),
  ("Shooting %", ["field_goal_pct", "three_point_field_goal_pct", "free_throw_pct"],
   ("field_goal_pctRoll10", "% FG")),
  ("Shot volume", ["field_goals_attempted", "three_point_field_goals_attempted",
                   "free_throws_attempted"], ("field_goals_attemptedRoll10", " FGA")),
  ("Rebounding", ["total_rebounds", "offensive_rebounds", "defensive_rebounds"],
   ("total_reboundsRoll10", " reb")),
  ("Playmaking", ["assists"], ("assistsRoll10", " ast")),
  ("Defensive plays", ["steals", "blocks"], ("stealsRoll10", " stl")),
  ("Turnovers", ["turnovers", "turnover_points"], ("turnoversRoll10", " TO")),
  ("Fouls", ["fouls", "technical_fouls"], ("foulsRoll10", " fouls")),
  ("Inside & transition", ["points_in_paint", "fast_break_points"],
   ("points_in_paintRoll10", " pts in paint")),
  ("Margin of control", ["largest_lead"], ("largest_leadRoll10", " largest lead")),
]


# Define the function that builds the basketball driver groups, which depend on the box-stat
# column list and therefore have to be assembled rather than written out literally.
def BuildBasketballDriverGroups(boxStatCols):
  groups = []
  covered = set()
  for label, stats, evidence in BASKETBALL_NAMED_STATS:
    keys = []
    for stat in stats:
      if (stat not in boxStatCols):
        continue
      covered.add(stat)
      # Each box stat reaches the model as home/away last-10 and season averages plus the diffs.
      for window in ("Roll10", "Season"):
        column = f"{stat}{window}"
        upper = f"{column[0].upper()}{column[1:]}"
        keys.extend([f"home{upper}", f"away{upper}", f"diff{upper}"])
    if (keys):
      evidenceColumn, unitSuffix = evidence
      valueSpec = ("home", evidenceColumn, unitSuffix) if (f"home{evidenceColumn[0].upper()}"
                                                          f"{evidenceColumn[1:]}" in
                                                          set(keys)) else None
      groups.append((label, valueSpec, keys))
  # Record, streak and rest are not box stats, so they are their own groups.
  groups.append(("Record", ("home", "winPctEntering", "%"), [
    "homeWinPctEntering", "awayWinPctEntering", "diffWinPctEntering",
    "homeWinsEntering", "awayWinsEntering", "diffWinsEntering",
    "homeGamesEntering", "awayGamesEntering", "diffGamesEntering"]))
  groups.append(("Current streak", ("home", "streakEntering", ""), [
    "homeStreakEntering", "awayStreakEntering", "diffStreakEntering"]))
  groups.append(("Rest days", ("home", "restDays", " days"), [
    "homeRestDays", "awayRestDays", "diffRestDays"]))
  groups.append(("Game context", None, ["neutral_site", "conference_competition", "season_type"]))
  # Anything left over still gets measured, just under a generic label.
  leftover = []
  for stat in boxStatCols:
    if (stat in covered):
      continue
    for window in ("Roll10", "Season"):
      column = f"{stat}{window}"
      upper = f"{column[0].upper()}{column[1:]}"
      leftover.extend([f"home{upper}", f"away{upper}", f"diff{upper}"])
  if (leftover):
    groups.append(("Other box stats", None, leftover))
  return groups


# Define the function that builds the "league-typical" row that occlusion neutralizes toward.
# A difference feature's neutral value is 0 (neither side has an edge); an absolute feature's is
# the median over the population; a categorical feature's is its most common value. Passing the
# snapshot lets this reflect the same population the two entities were drawn from.
def BuildNeutralRow(realRow: dict, snapshot: pandas.DataFrame, sport: str) -> dict:
  neutral = {}
  # Precompute the medians of every numeric snapshot column, weighted so that entities with more
  # games/fights count more - the same weighting used for debut fighters, and for the same
  # reason: the training rows the model learned from are per-appearance, not per-entity.
  weightCol = "FightsEntering" if (sport == "ufc") else "gamesEntering"
  weights = snapshot[weightCol] if (weightCol in snapshot.columns) else None
  medians = {}
  for column in snapshot.columns:
    if (not pandas.api.types.is_numeric_dtype(snapshot[column])):
      continue
    if (weights is not None):
      medians[column] = _WeightedMedian(snapshot[column], weights)
    else:
      medians[column] = float(snapshot[column].median())

  for key, value in realRow.items():
    # A difference feature is neutral at zero - that is literally "no advantage either way".
    if (key.startswith("diff") or key.endswith("Diff")):
      neutral[key] = 0.0
      continue
    # A prefixed absolute feature maps back to its snapshot column.
    baseKey = _StripEntityPrefix(key, sport)
    if (baseKey is not None and baseKey in medians):
      neutral[key] = medians[baseKey]
      continue
    # Everything else (context fields, categoricals, unmapped columns) keeps its real value,
    # except where a group explicitly lists it - see NeutralizeGroup below.
    neutral[key] = value
  return neutral


# Define the helper that turns a prefixed feature name back into its snapshot column name.
def _StripEntityPrefix(key: str, sport: str):
  prefixes = ("f1", "f2") if (sport == "ufc") else ("home", "away")
  for prefix in prefixes:
    if (not key.startswith(prefix)):
      continue
    remainder = key[len(prefix):]
    if (not remainder):
      continue
    if (sport == "ufc"):
      # UFC snapshot columns are capitalized ("SigStrLandedEntering") except the physical ones,
      # which the matchup row carries through unchanged ("f1reachIn" -> "reachIn").
      return remainder
    # Basketball snapshot columns are lower-camel ("winPctEntering"), and the matchup row
    # upper-cases the first letter when prefixing ("homeWinPctEntering").
    return f"{remainder[0].lower()}{remainder[1:]}"
  return None


# Define the fight-count-weighted median, mirroring UFCFeatures.WeightedMedian.
def _WeightedMedian(values, weights) -> float:
  valueArr = numpy.asarray(values, dtype=float)
  weightArr = numpy.asarray(weights, dtype=float)
  validMask = ~numpy.isnan(valueArr) & ~numpy.isnan(weightArr) & (weightArr > 0)
  valueArr, weightArr = valueArr[validMask], weightArr[validMask]
  if (len(valueArr) == 0):
    plain = numpy.asarray(values, dtype=float)
    return float(numpy.nanmedian(plain)) if (len(plain) > 0 and not numpy.all(numpy.isnan(plain))) else 0.0
  sortOrder = numpy.argsort(valueArr)
  valueArr, weightArr = valueArr[sortOrder], weightArr[sortOrder]
  cumulative = numpy.cumsum(weightArr)
  return float(valueArr[numpy.searchsorted(cumulative, cumulative[-1] / 2.0)])


# Define the function that produces one row with a single driver group neutralized.
def NeutralizeGroup(realRow: dict, neutralRow: dict, groupKeys) -> dict:
  occluded = dict(realRow)
  for key in groupKeys:
    if (key in occluded):
      occluded[key] = neutralRow.get(key, occluded[key])
  return occluded


# Define the function that ranks driver groups by how much each one moved this prediction.
# Returns a list of dicts ready for the frontend, most influential first.
def ExplainPrediction(predictFn, realRow: dict, snapshot: pandas.DataFrame, sport: str,
                      driverGroups, entity1Name: str, entity2Name: str,
                      topCount: int = TOP_DRIVER_COUNT):
  # Establish the baseline: the class the model actually predicted, and how sure it was.
  # predictFn is whatever the app used to produce the headline number - for UFC that is the
  # order-averaged predictor, so the explanation describes the SAME quantity the user is
  # looking at rather than a different one computed from the raw model.
  predictedLabel, baselineProbs = predictFn(realRow)
  baselineProb = baselineProbs[predictedLabel]
  neutralRow = BuildNeutralRow(realRow, snapshot, sport)

  drivers = []
  for label, valueSpec, groupKeys in driverGroups:
    presentKeys = [k for k in groupKeys if (k in realRow)]
    if (not presentKeys):
      continue
    # Re-predict with this group of features reset to league-typical, and read the probability
    # of the SAME class the real prediction chose (not the new argmax - we are measuring how
    # much this group supported that specific call).
    occludedRow = NeutralizeGroup(realRow, neutralRow, presentKeys)
    _, occludedProbs = predictFn(occludedRow)
    contribution = (baselineProb - occludedProbs.get(predictedLabel, 0.0)) * 100.0
    if (abs(contribution) < MIN_CONTRIBUTION_PP):
      continue
    drivers.append({
      "label": label,
      "contributionPp": round(contribution, 2),
      "detail": _DescribeGroup(valueSpec, realRow, sport, entity1Name, entity2Name),
    })

  drivers.sort(key=lambda d: abs(d["contributionPp"]), reverse=True)
  return {
    "predictedLabel": predictedLabel,
    "baselineProbability": baselineProb,
    "drivers": drivers[:topCount],
    "supporting": [d for d in drivers[:topCount] if (d["contributionPp"] > 0)],
    "opposing": [d for d in drivers[:topCount] if (d["contributionPp"] < 0)],
  }


# Define the helper that renders a short "what the numbers actually are" string for a driver.
def _DescribeGroup(valueSpec, realRow: dict, sport: str, entity1Name: str, entity2Name: str):
  if (valueSpec is None):
    return ""
  _, baseColumn, unitSuffix = valueSpec
  prefix1, prefix2 = ("f1", "f2") if (sport == "ufc") else ("home", "away")
  if (sport == "ufc"):
    key1, key2 = f"{prefix1}{baseColumn}", f"{prefix2}{baseColumn}"
  else:
    upper = f"{baseColumn[0].upper()}{baseColumn[1:]}"
    key1, key2 = f"{prefix1}{upper}", f"{prefix2}{upper}"
  if (key1 not in realRow or key2 not in realRow):
    return ""
  value1, value2 = realRow[key1], realRow[key2]
  # A categorical value (stance) is shown as-is; a missing one reads as "unknown" rather than
  # being silently dropped, since "we don't know this fighter's stance" is itself informative.
  if (not isinstance(value1, (int, float)) or not isinstance(value2, (int, float))):
    text1 = str(value1) if (value1 is not None and not _IsMissing(value1)) else "unknown"
    text2 = str(value2) if (value2 is not None and not _IsMissing(value2)) else "unknown"
    return f"{entity1Name} {text1} vs {entity2Name} {text2}"
  if (pandas.isna(value1) or pandas.isna(value2)):
    return ""
  # A unit suffix of "%" means the stored value is a 0-1 rate that reads far better as a
  # percentage - "Jon Jones 100% vs Ciryl Gane 85%" rather than "Jon Jones 1 vs Ciryl Gane 0.85".
  if (unitSuffix == "%"):
    return f"{entity1Name} {value1 * 100:.0f}% vs {entity2Name} {value2 * 100:.0f}%"
  return (f"{entity1Name} {_FormatNumber(value1)}{unitSuffix} vs "
          f"{entity2Name} {_FormatNumber(value2)}{unitSuffix}")


# Define the helper that treats NaN/None as missing without tripping over non-scalar values.
def _IsMissing(value) -> bool:
  try:
    return bool(pandas.isna(value))
  except (TypeError, ValueError):
    return False


# Define the helper that formats a number without trailing noise.
def _FormatNumber(value) -> str:
  value = float(value)
  if (abs(value - round(value)) < 0.05):
    return str(int(round(value)))
  return f"{value:.1f}" if (abs(value) >= 1) else f"{value:.2f}"
