import numpy  # Import numpy for numeric helpers.
import pandas  # Import pandas for building the single-row feature frames.

# This module turns "two fighters" or "two teams" into the exact raw feature dict that
# UFCFeatures.BuildUFCDataset() / BasketballFeatures.BuildBasketballDataset() would have produced
# for that matchup, using each side's live snapshot row (see BuildCurrentFighterSnapshot() /
# BuildCurrentTeamSnapshot() in those modules). Column names here MUST match those modules'
# featureCols exactly - model_loader.py's encoder then one-hot encodes and scales this dict the
# same way training did.

ROLL_SUFFIX_UFC = "Roll5"  # Must match UFCFeatures.ROLL_WINDOW_FIGHTS.


# Define the function that builds one UFC matchup's raw feature row from two fighter snapshots.
def BuildUfcMatchupRow(
  fighter1Snap: pandas.Series, fighter2Snap: pandas.Series,
  weightClass: str, isTitleFight: bool, scheduledRounds: int,
  eventDate: pandas.Timestamp = None,
) -> dict:
  # Default to today if no hypothetical event date was given.
  if (eventDate is None):
    eventDate = pandas.Timestamp.now().normalize()

  row = {}
  # Static physical attributes, straight from each fighter's snapshot.
  for prefix, snap in (("f1", fighter1Snap), ("f2", fighter2Snap)):
    row[f"{prefix}heightIn"] = float(snap["heightIn"])
    row[f"{prefix}reachIn"] = float(snap["reachIn"])
    row[f"{prefix}weightLbs"] = float(snap["weightLbs"])
    # Age is computed fresh at request time (not baked into the snapshot), so it's always
    # correct relative to the hypothetical event date, however long ago the snapshot was built.
    dob = snap.get("dob")
    row[f"{prefix}AgeAtFight"] = (
      (eventDate - dob).days / 365.25 if (pandas.notna(dob)) else 30.0  # league-typical fallback
    )
    row[f"{prefix}stance"] = snap["stance"] if (pandas.notna(snap.get("stance"))) else numpy.nan

    # Career record + finishing tendency + layoff + activity, straight from the snapshot.
    for col in [
      "WinsEntering", "LossesEntering", "FightsEntering", "WinPctEntering", "StreakEntering",
      "FinishRateEntering", "KoWinRateEntering", "SubWinRateEntering", "AvgRoundEntering",
      "DaysSinceLastFightEntering", "ActivityRateEntering",
    ]:
      row[f"{prefix}{col}"] = float(snap[col])

    # Career and recent-form (last-5) in-fight performance stats.
    for statLabel in [
      "SigStrLanded", "SigStrAcc", "SigStrAbsorbed", "TdLanded", "TdAcc", "CtrlSeconds", "Kd", "SubAtt",
    ]:
      row[f"{prefix}{statLabel}Entering"] = float(snap[f"{statLabel}Entering"])
      row[f"{prefix}{statLabel}{ROLL_SUFFIX_UFC}"] = float(snap[f"{statLabel}{ROLL_SUFFIX_UFC}"])

  # Symmetric fighter1-minus-fighter2 difference features.
  row["reachDiff"] = row["f1reachIn"] - row["f2reachIn"]
  row["heightDiff"] = row["f1heightIn"] - row["f2heightIn"]
  row["ageDiff"] = row["f1AgeAtFight"] - row["f2AgeAtFight"]
  row["winPctDiff"] = row["f1WinPctEntering"] - row["f2WinPctEntering"]
  row["experienceDiff"] = row["f1FightsEntering"] - row["f2FightsEntering"]
  row["streakDiff"] = row["f1StreakEntering"] - row["f2StreakEntering"]
  row["finishRateDiff"] = row["f1FinishRateEntering"] - row["f2FinishRateEntering"]
  row["koWinRateDiff"] = row["f1KoWinRateEntering"] - row["f2KoWinRateEntering"]
  row["subWinRateDiff"] = row["f1SubWinRateEntering"] - row["f2SubWinRateEntering"]
  row["avgRoundDiff"] = row["f1AvgRoundEntering"] - row["f2AvgRoundEntering"]
  row["layoffDiff"] = row["f1DaysSinceLastFightEntering"] - row["f2DaysSinceLastFightEntering"]
  row["activityRateDiff"] = row["f1ActivityRateEntering"] - row["f2ActivityRateEntering"]
  for statLabel, diffName in [
    ("SigStrLanded", "sigStrLandedDiff"), ("SigStrAcc", "sigStrAccDiff"),
    ("SigStrAbsorbed", "sigStrAbsorbedDiff"), ("TdLanded", "tdLandedDiff"),
    ("TdAcc", "tdAccDiff"), ("CtrlSeconds", "ctrlSecondsDiff"), ("Kd", "kdDiff"), ("SubAtt", "subAttDiff"),
  ]:
    row[diffName] = row[f"f1{statLabel}Entering"] - row[f"f2{statLabel}Entering"]
    row[f"{statLabel[0].lower()}{statLabel[1:]}{ROLL_SUFFIX_UFC}Diff"] = (
      row[f"f1{statLabel}{ROLL_SUFFIX_UFC}"] - row[f"f2{statLabel}{ROLL_SUFFIX_UFC}"]
    )

  # Bout-level context (chosen in the UI, not derived from either fighter alone).
  row["weightClassClean"] = weightClass
  row["isTitleFight"] = int(bool(isTitleFight))
  row["scheduledRounds"] = float(scheduledRounds)
  row["stanceMatchup"] = f"{row['f1stance'] if pandas.notna(row['f1stance']) else 'Unknown'}_" \
                          f"{row['f2stance'] if pandas.notna(row['f2stance']) else 'Unknown'}"
  return row


# Define the function that returns the same UFC matchup with the two fighters swapped.
#
# WHY THIS EXISTS: "which round does this fight end in" and "how does it end" are questions about
# the BOUT, so the answer should not depend on which fighter you happen to type first. The model
# has no idea of that, though - it only ever saw rows where fighter1 came from the left-hand side
# of a ufcstats BOUT string, and ufcstats lists the winner first (5579 "W/L" vs 3138 "L/W"), so
# "fighter 1" carries a real positional signal that the model learns. Measured on a test model,
# swapping the two names changed the predicted round in 36% of random matchups, shifting
# probabilities by a median of 12.5 points. Averaging the model's answer over both orderings
# removes that artifact; app.py does this for UFC only. It is deliberately NOT done for
# basketball, where home vs away is a genuine asymmetry the model SHOULD be using.
def SwapUfcMatchupRow(row: dict) -> dict:
  swapped = {}
  for key, value in row.items():
    if (key.startswith("f1")):
      swapped[f"f2{key[2:]}"] = value
    elif (key.startswith("f2")):
      swapped[f"f1{key[2:]}"] = value
    elif (key.endswith("Diff")):
      # Every difference feature is defined as fighter1 minus fighter2, so swapping negates it.
      swapped[key] = -value if (isinstance(value, (int, float)) and not pandas.isna(value)) else value
    else:
      # Bout-level context (weight class, title fight, scheduled rounds) is unaffected by order.
      swapped[key] = value
  # stanceMatchup is a composed string, so rebuild it from the now-swapped stances.
  if ("f1stance" in swapped and "f2stance" in swapped):
    stance1 = swapped["f1stance"] if (pandas.notna(swapped["f1stance"])) else "Unknown"
    stance2 = swapped["f2stance"] if (pandas.notna(swapped["f2stance"])) else "Unknown"
    swapped["stanceMatchup"] = f"{stance1}_{stance2}"
  return swapped


# Define the function that builds one basketball matchup's raw feature row from two team snapshots.
def BuildBasketballMatchupRow(
  homeSnap: pandas.Series, awaySnap: pandas.Series,
  neutralSite: bool, conferenceCompetition: bool, seasonType: str,
) -> dict:
  # Import here (not at module load) to avoid a hard dependency if only UFC is used.
  from BasketballFeatures import BOX_STAT_COLS

  row = {}
  featureCols = (
    [f"{c}Roll10" for c in BOX_STAT_COLS] + [f"{c}Season" for c in BOX_STAT_COLS] +
    ["winsEntering", "gamesEntering", "winPctEntering", "streakEntering", "restDays"]
  )
  for prefix, snap in (("home", homeSnap), ("away", awaySnap)):
    for col in featureCols:
      key = f"{prefix}{col[0].upper()}{col[1:]}"
      row[key] = float(snap[col])
  # Home-minus-away difference for every one of those features.
  for col in featureCols:
    key = f"diff{col[0].upper()}{col[1:]}"
    row[key] = row[f"home{col[0].upper()}{col[1:]}"] - row[f"away{col[0].upper()}{col[1:]}"]

  row["neutral_site"] = int(bool(neutralSite))
  row["conference_competition"] = int(bool(conferenceCompetition))
  row["season_type"] = seasonType
  return row
