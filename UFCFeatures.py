import os  # Import the os module to read the optional environment variable override.
import re  # Import the re module for regular expression parsing.
import numpy  # Import the numpy module for numerical operations.
import pandas  # Import the pandas module for tabular data handling.
from typing import Dict, List, Tuple  # Import typing utilities.

# Define the default directory containing the raw UFC CSV files, relative to
# wherever the pipeline is run from. Copy your 6 ufc_*.csv files into this
# folder (see README.md), or override it per-run via the UFC_DATA_DIR
# environment variable, or by passing dataDir=... into BuildUFCDataset().
UFC_DATA_DIR = os.environ.get("UFC_DATA_DIR", os.path.join("Data", "UFC"))

# Define the recent-form window (in fights) used for the rolling in-fight-performance averages,
# mirroring the basketball pipeline's ROLL_WINDOW (there measured in games, here in fights).
ROLL_WINDOW_FIGHTS = 5

# Define the set of METHOD values that do not represent a clean, decisive win.
BAD_METHODS = {"Overturned", "Could Not Continue", "DQ", "Other"}
# Define the set of OUTCOME values that do not represent a clean win/loss.
BAD_OUTCOMES = {"NC/NC", "D/D"}

# Define the mapping from raw METHOD strings to the 3 canonical method classes.
METHOD_MAP = {
  "KO/TKO"                  : "KO/TKO",
  "TKO - Doctor's Stoppage" : "KO/TKO",
  "Submission"              : "Submission",
  "Decision - Unanimous"    : "Decision",
  "Decision - Split"        : "Decision",
  "Decision - Majority"     : "Decision",
}

# Define the ordered list of (substring, canonical label) pairs used to collapse
# ~130 raw WEIGHTCLASS strings (tournament names, TUF seasons, interim/title
# fights, catchweights, etc.) down to a small set of canonical divisions.
WEIGHTCLASS_ORDER = [
  ("Women's Strawweight", "Women's Strawweight"),
  ("Women's Flyweight", "Women's Flyweight"),
  ("Women's Bantamweight", "Women's Bantamweight"),
  ("Women's Featherweight", "Women's Featherweight"),
  ("Flyweight", "Flyweight"),
  ("Bantamweight", "Bantamweight"),
  ("Featherweight", "Featherweight"),
  ("Lightweight", "Lightweight"),
  ("Welterweight", "Welterweight"),
  ("Middleweight", "Middleweight"),
  ("Light Heavyweight", "Light Heavyweight"),
  ("Heavyweight", "Heavyweight"),
  ("Catch Weight", "Catch Weight"),
  ("Open Weight", "Open Weight"),
  ("Superfight", "Open Weight"),
]


# Define the fprint helper so this module logs the same way as the rest of the pipeline.
def fprint(msg, *args, **kwargs):
  # Print the message immediately, flushing the output buffer.
  print(msg, flush=True, *args, **kwargs)


# Define the function that parses a HEIGHT string like "5' 11\"" into total inches.
def ParseHeightToInches(heightStr) -> float:
  # Check for missing or placeholder values.
  if (pandas.isna(heightStr) or heightStr == "--"):
    # Return NaN for missing height.
    return numpy.nan
  # Match the feet and inches components with a regular expression.
  match = re.match(r"(\d+)'\s*(\d+)", str(heightStr))
  # Check if the pattern matched.
  if (not match):
    # Return NaN when the string could not be parsed.
    return numpy.nan
  # Extract the feet component.
  feet = int(match.group(1))
  # Extract the inches component.
  inches = int(match.group(2))
  # Convert feet and inches into total inches.
  return feet * 12 + inches


# Define the function that parses a REACH string like "72\"" into inches.
def ParseReachToInches(reachStr) -> float:
  # Check for missing or placeholder values.
  if (pandas.isna(reachStr) or reachStr == "--"):
    # Return NaN for missing reach.
    return numpy.nan
  # Match the leading numeric component with a regular expression.
  match = re.match(r"([\d.]+)", str(reachStr))
  # Return the parsed float, or NaN if unparseable.
  return float(match.group(1)) if (match) else numpy.nan


# Define the function that parses a WEIGHT string like "155 lbs." into pounds.
def ParseWeightToPounds(weightStr) -> float:
  # Check for missing or placeholder values.
  if (pandas.isna(weightStr) or weightStr == "--"):
    # Return NaN for missing weight.
    return numpy.nan
  # Match the leading numeric component with a regular expression.
  match = re.match(r"([\d.]+)", str(weightStr))
  # Return the parsed float, or NaN if unparseable.
  return float(match.group(1)) if (match) else numpy.nan


# Define the function that collapses a raw WEIGHTCLASS string to a canonical division.
def CanonicalizeWeightClass(weightClass) -> str:
  # Check for a missing weight class.
  if (pandas.isna(weightClass)):
    # Return the "Unknown" label.
    return "Unknown"
  # Iterate through the canonical substrings in priority order.
  for needle, label in WEIGHTCLASS_ORDER:
    # Check if the needle substring is present (case-insensitively).
    if (needle.lower() in str(weightClass).lower()):
      # Return the canonical label.
      return label
  # Return the "Other" label for tournament/unusual weight classes.
  return "Other"


# Define the function that parses a TIME FORMAT string like "3 Rnd (5-5-5)" into the number of
# scheduled rounds. This is known before the fight happens (the promotion sets the fight's
# round format when the bout is scheduled/announced) - it is NOT the actual round the fight
# ended in, so using it is not leakage, unlike ROUND itself.
def ParseScheduledRounds(timeFormatStr) -> float:
  # Check for a missing time-format string.
  if (pandas.isna(timeFormatStr)):
    # Return NaN for missing data.
    return numpy.nan
  # Match the leading round count (e.g. the "3" in "3 Rnd (5-5-5)" or "5" in "5 Rnd (...)").
  match = re.match(r"(\d+)\s*Rnd", str(timeFormatStr))
  # Return the parsed round count, or NaN for an unparseable/unusual format
  # (e.g. "No Time Limit"), which the pipeline's own NaN handling covers downstream.
  return float(match.group(1)) if (match) else numpy.nan


# Define the function that splits a BOUT string like "A vs. B" into two fighter names.
def SplitBoutString(bout: str) -> pandas.Series:
  # Split on the "vs" / "vs." separator, allowing for whitespace variation.
  parts = re.split(r"\s+vs\.?\s+", str(bout), maxsplit=1)
  # Check that exactly two fighter names were found.
  if (len(parts) != 2):
    # Return a pair of NaN values when the bout string is malformed.
    return pandas.Series([numpy.nan, numpy.nan])
  # Return the two stripped fighter names.
  return pandas.Series([parts[0].strip(), parts[1].strip()])


# Define the function that loads and cleans the static fighter attribute table.
def LoadFighterAttributes(dataDir: str = UFC_DATA_DIR) -> pandas.DataFrame:
  # Read the fighter physical attributes CSV file.
  tott = pandas.read_csv(os.path.join(dataDir, "ufc_fighter_tott.csv"))
  # Drop duplicate fighter name rows, keeping the first occurrence.
  tott = tott.drop_duplicates(subset=["FIGHTER"], keep="first")
  # Parse the height column into inches.
  tott["heightIn"] = tott["HEIGHT"].apply(ParseHeightToInches)
  # Parse the reach column into inches.
  tott["reachIn"] = tott["REACH"].apply(ParseReachToInches)
  # Parse the weight column into pounds.
  tott["weightLbs"] = tott["WEIGHT"].apply(ParseWeightToPounds)
  # Parse the date of birth column into a proper datetime.
  tott["dob"] = pandas.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce")
  # Replace empty stance strings with NaN.
  tott["stance"] = tott["STANCE"].replace({"": numpy.nan})
  # Return only the columns needed downstream.
  return tott[["FIGHTER", "heightIn", "reachIn", "weightLbs", "dob", "stance"]]


# Define the function that parses an "X of Y" landed/attempted string (e.g. "14 of 31") into
# a (landed, attempted) pair of floats, used for SIG.STR./TOTAL STR./TD.
def ParseLandedOfAttempted(valueStr) -> Tuple[float, float]:
  # Check for a missing value.
  if (pandas.isna(valueStr)):
    # Return NaN for both landed and attempted.
    return numpy.nan, numpy.nan
  # Match the "landed of attempted" pattern.
  match = re.match(r"(\d+)\s+of\s+(\d+)", str(valueStr))
  # Check if the pattern matched.
  if (not match):
    # Return NaN for both landed and attempted when unparseable.
    return numpy.nan, numpy.nan
  # Return the parsed landed and attempted counts.
  return float(match.group(1)), float(match.group(2))


# Define the function that parses a CTRL time string like "4:20" (minutes:seconds) into
# total seconds of control time.
def ParseControlSeconds(ctrlStr) -> float:
  # Check for a missing or placeholder control-time value (some early events lack this stat).
  if (pandas.isna(ctrlStr) or ctrlStr == "--"):
    # Return NaN for missing control time.
    return numpy.nan
  # Match the minutes:seconds pattern.
  match = re.match(r"(\d+):(\d+)", str(ctrlStr))
  # Check if the pattern matched.
  if (not match):
    # Return NaN when the string could not be parsed.
    return numpy.nan
  # Convert minutes and seconds into total seconds.
  return int(match.group(1)) * 60 + int(match.group(2))


# Define the function that loads and aggregates the round-by-round fight-stats CSV down to
# one row per (EVENT, BOUT, FIGHTER) - i.e. that fighter's own totals for that single fight.
# These are still "this fight's own stats" at this point (not yet safe to use as a feature for
# predicting that same fight's outcome) - BuildUFCDataset() below turns them into career
# averages *entering* each fight, the same way it already does for win/loss record.
def LoadFightStats(dataDir: str = UFC_DATA_DIR) -> pandas.DataFrame:
  # Read the round-by-round fight-stats CSV file.
  roundStats = pandas.read_csv(os.path.join(dataDir, "ufc_fight_stats.csv"))
  # Parse the significant-strikes landed/attempted pair.
  roundStats[["sigStrLanded", "sigStrAttempted"]] = roundStats["SIG.STR."].apply(
    lambda s: pandas.Series(ParseLandedOfAttempted(s)))
  # Parse the takedown landed/attempted pair.
  roundStats[["tdLanded", "tdAttempted"]] = roundStats["TD"].apply(
    lambda s: pandas.Series(ParseLandedOfAttempted(s)))
  # Parse the control-time string into total seconds.
  roundStats["ctrlSeconds"] = roundStats["CTRL"].apply(ParseControlSeconds)
  # Rename the remaining per-round counting stats to lowerCamelCase for consistency.
  roundStats = roundStats.rename(columns={"KD": "kd", "SUB.ATT": "subAtt"})
  # Aggregate every round of a fight into that fighter's single fight-level total (sum across
  # rounds - e.g. a 3-round fight's significant strikes landed is the sum of all 3 rounds).
  fightLevel = roundStats.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[[
    "sigStrLanded", "sigStrAttempted", "tdLanded", "tdAttempted", "ctrlSeconds", "kd", "subAtt",
  ]].sum(min_count=1)
  # Return the fight-level per-fighter stats table.
  return fightLevel


# Define the function that computes a fighter's chronological win/loss streak entering each fight.
def ComputeStreakEntering(subFrame: pandas.DataFrame) -> pandas.Series:
  # Initialize the list that accumulates the streak value before each fight.
  streakValues = []
  # Initialize the running streak counter.
  currentStreak = 0
  # Iterate through the fighter's results in chronological order.
  for result in subFrame["result"]:
    # Record the streak value entering this fight, before updating it.
    streakValues.append(currentStreak)
    # Check if the fighter won this fight.
    if (result == "W"):
      # Extend a positive streak, or start a new one.
      currentStreak = currentStreak + 1 if (currentStreak >= 0) else 1
    elif (result == "L"):
      # Extend a negative streak, or start a new one.
      currentStreak = currentStreak - 1 if (currentStreak <= 0) else -1
    else:
      # Reset the streak on a draw/no-contest.
      currentStreak = 0
  # Return the streak values aligned to the original index.
  return pandas.Series(streakValues, index=subFrame.index)


# Define the function that computes a fighter's career-to-date average of some per-fight
# performance stat (e.g. significant strikes landed), entering each fight - i.e. the average
# over that fighter's EARLIER fights only, never including the fight the row itself represents.
# This is what makes in-fight performance stats usable: the current fight's own strikes/takedowns
# would be leakage (they don't exist until after the fight), but "how this fighter has performed
# on average across their career so far" is known before the fight and is legitimate signal.
#
# window=None gives the all-time career average (an expanding mean). Passing an integer window
# instead gives a RECENT-FORM average over only that fighter's last `window` fights - the same
# "shift first, so the current fight is never included" rule still applies. Recent form matters
# because fighters aren't static: a 15-fight veteran's all-time average blends together how they
# fought 8 years ago and how they fight now, while a last-5-fights average tracks where they are
# TODAY (improving, declining, coming off an injury, etc.) - exactly the same idea as the
# basketball pipeline's Roll10-vs-season-average pair, applied to fighters instead of teams.
def ComputeEnteringMean(subFrame: pandas.DataFrame, col: str, window: int = None) -> pandas.Series:
  # Shift the stat column down by one fight so every row only ever sees strictly earlier fights.
  shiftedVals = subFrame[col].shift(1)
  # Recent-form path: a plain rolling mean over the last `window` shifted (already-past) values.
  if (window is not None):
    return shiftedVals.rolling(window, min_periods=1).mean()
  # Career path: running total of that stat across all earlier fights (skipping missing ones).
  cumulativeSum = shiftedVals.cumsum()
  # Running count of earlier fights where that stat was actually recorded (some older events are
  # missing a few stats, e.g. control time) - used as the averaging denominator.
  cumulativeCount = shiftedVals.notna().cumsum()
  # Return the running mean (NaN before the fighter has any recorded prior fight with this stat -
  # left for the pipeline's own train-fold-median imputation to fill in, same as every other
  # pre-fight feature with no history yet).
  return cumulativeSum / cumulativeCount.replace(0, numpy.nan)


# Define the function that builds the full pre-fight UFC modeling dataset.
def BuildUFCDataset(dataDir: str = UFC_DATA_DIR) -> pandas.DataFrame:
  # Read the event details CSV file (used to get chronological event dates).
  events = pandas.read_csv(os.path.join(dataDir, "ufc_event_details.csv"))
  # Parse the event date column into a proper datetime.
  events["eventDate"] = pandas.to_datetime(events["DATE"], errors="coerce")
  # Read the fight results CSV file.
  results = pandas.read_csv(os.path.join(dataDir, "ufc_fight_results.csv"))

  # Merge the fight results with the event dates and status.
  fights = results.merge(events[["EVENT", "eventDate", "STATUS"]], on="EVENT", how="left")
  # Filter out events that have not happened yet.
  fights = fights[fights["STATUS"] != "UPCOMING"].copy()
  # Drop fights that could not be matched to a valid event date.
  fights = fights.dropna(subset=["eventDate"]).copy()

  # Split the BOUT column into two separate fighter name columns.
  fights[["fighter1", "fighter2"]] = fights["BOUT"].apply(SplitBoutString)
  # Drop fights where the bout string could not be parsed.
  fights = fights.dropna(subset=["fighter1", "fighter2"])

  # Split the OUTCOME column (e.g. "W/L") into per-corner result columns.
  outcomeParts = fights["OUTCOME"].str.split("/", expand=True)
  # Store the fighter 1 result.
  fights["f1Result"] = outcomeParts[0]
  # Store the fighter 2 result.
  fights["f2Result"] = outcomeParts[1]

  # Filter out fights with an ambiguous outcome (no-contest / draw).
  fights = fights[~fights["OUTCOME"].isin(BAD_OUTCOMES)]
  # Filter out fights with a non-standard method (overturned, DQ, etc.).
  fights = fights[~fights["METHOD"].isin(BAD_METHODS)]
  # Map the raw method string to the canonical 3-class method target.
  fights["methodClass"] = fights["METHOD"].map(METHOD_MAP)
  # Drop any fights whose method could not be mapped.
  fights = fights.dropna(subset=["methodClass"])
  # Convert the ROUND column to a numeric round-ended-in target.
  fights["roundClass"] = pandas.to_numeric(fights["ROUND"], errors="coerce")
  # Drop any fights with a missing round value.
  fights = fights.dropna(subset=["roundClass"])

  # Sort all fights chronologically so pre-fight stats can be computed without leakage.
  fights = fights.sort_values("eventDate").reset_index(drop=True)
  # Assign a stable integer index used to re-merge long-format records back onto fights.
  fights["fightIdx"] = fights.index

  # Build a long-format table with one row per (fighter, fight) appearance.
  longRows = pandas.concat([
    fights[["fightIdx", "eventDate", "fighter1", "f1Result"]].rename(
      columns={"fighter1": "fighter", "f1Result": "result"}),
    fights[["fightIdx", "eventDate", "fighter2", "f2Result"]].rename(
      columns={"fighter2": "fighter", "f2Result": "result"}),
  ], ignore_index=True)
  # Bring in how each of the fighter's own past fights ended (method/round) - needed below to
  # build "how this fighter tends to finish fights" features. This is the SAME leakage-safe
  # pattern as everything else here: methodClass/roundClass are only ever read from a fighter's
  # STRICTLY EARLIER fights via the "entering" shift, never from the fight being predicted.
  longRows = longRows.merge(fights[["fightIdx", "methodClass", "roundClass"]], on="fightIdx", how="left")
  # Sort the long-format table chronologically within each fighter.
  longRows = longRows.sort_values(["fighter", "eventDate", "fightIdx"]).reset_index(drop=True)
  # Flag rows where the fighter won.
  longRows["winFlag"] = (longRows["result"] == "W").astype(int)
  # Flag rows where the fighter lost.
  longRows["lossFlag"] = (longRows["result"] == "L").astype(int)
  # Group the long-format table by fighter for cumulative calculations.
  fighterGroup = longRows.groupby("fighter")
  # Compute the number of wins entering each fight (cumulative sum, excluding the current row).
  longRows["winsEntering"] = fighterGroup["winFlag"].cumsum() - longRows["winFlag"]
  # Compute the number of losses entering each fight.
  longRows["lossesEntering"] = fighterGroup["lossFlag"].cumsum() - longRows["lossFlag"]
  # Compute the number of prior fights entering each fight.
  longRows["fightsEntering"] = fighterGroup.cumcount()
  # Compute the win percentage entering each fight, defaulting to a neutral 0.5 prior with no history.
  longRows["winPctEntering"] = (
    longRows["winsEntering"] / longRows["fightsEntering"].replace(0, numpy.nan)
  ).fillna(0.5)
  # Compute the win/loss streak entering each fight.
  longRows["streakEntering"] = longRows.groupby("fighter", group_keys=False).apply(ComputeStreakEntering)

  # ---- Career finishing tendency (how THIS fighter's past fights have ended) ----
  # Distinct from the in-fight performance stats below (which measure output like strikes/
  # takedowns): this captures a fighter's finishing STYLE and durability from their own fight
  # outcomes, which is direct pre-fight signal for both targets - a fighter who finishes (or gets
  # finished) often is informative for roundClass, and KO-heavy vs. submission-heavy tendencies
  # are directly informative for methodClass.
  longRows["finishFlag"] = (longRows["methodClass"] != "Decision").astype(int)
  longRows["koWinFlag"] = ((longRows["result"] == "W") & (longRows["methodClass"] == "KO/TKO")).astype(int)
  longRows["subWinFlag"] = ((longRows["result"] == "W") & (longRows["methodClass"] == "Submission")).astype(int)
  # Fraction of this fighter's past fights (win or lose) that ended before the final bell.
  longRows["finishRateEntering"] = longRows.groupby("fighter", group_keys=False).apply(
    lambda g: ComputeEnteringMean(g, "finishFlag"))
  # Fraction of this fighter's past fights won specifically by KO/TKO.
  longRows["koWinRateEntering"] = longRows.groupby("fighter", group_keys=False).apply(
    lambda g: ComputeEnteringMean(g, "koWinFlag"))
  # Fraction of this fighter's past fights won specifically by submission.
  longRows["subWinRateEntering"] = longRows.groupby("fighter", group_keys=False).apply(
    lambda g: ComputeEnteringMean(g, "subWinFlag"))
  # Average round this fighter's own past fights have ended in (win or lose) - a fighter who's
  # historically in fast fights vs. long grinding ones, independent of who wins this one.
  longRows["avgRoundEntering"] = longRows.groupby("fighter", group_keys=False).apply(
    lambda g: ComputeEnteringMean(g, "roundClass"))

  # ---- Layoff and activity level ----
  # Days since this fighter's last fight, entering this one - ring rust from a long layoff (injury,
  # suspension, contract dispute) vs. a fighter who's fought recently and is battle-sharp. A debut
  # fighter has no prior fight to measure from, so that case gets a neutral typical-layoff default
  # (180 days) rather than 0 (which would wrongly say "just fought yesterday").
  longRows["daysSinceLastFightEntering"] = (
    longRows.groupby("fighter")["eventDate"].diff().dt.days.fillna(180).clip(lower=0, upper=2000)
  )
  # Fights per year of active career entering this fight - how busy this fighter has been kept,
  # as opposed to fightsEntering's raw career-length count (a 10-fight veteran over 3 years reads
  # very differently from a 10-fight veteran over 10 years).
  firstFightDateEntering = longRows.groupby("fighter")["eventDate"].transform("min")
  yearsSinceDebutEntering = (longRows["eventDate"] - firstFightDateEntering).dt.days / 365.25
  longRows["activityRateEntering"] = longRows["fightsEntering"] / yearsSinceDebutEntering.replace(0, numpy.nan)

  # Define the list of pre-fight record columns to carry back onto the wide fight table.
  recordCols = [
    "winsEntering", "lossesEntering", "fightsEntering", "winPctEntering", "streakEntering",
    "finishRateEntering", "koWinRateEntering", "subWinRateEntering", "avgRoundEntering",
    "daysSinceLastFightEntering", "activityRateEntering",
  ]

  # Re-merge the fighter-1 perspective of the long-format table back onto the fights table.
  f1Records = longRows.merge(
    fights[["fightIdx", "fighter1"]], left_on=["fightIdx", "fighter"], right_on=["fightIdx", "fighter1"]
  )[["fightIdx"] + recordCols]
  # Rename the columns with an "f1" prefix.
  f1Records.columns = ["fightIdx"] + [f"f1{c[0].upper()}{c[1:]}" for c in recordCols]

  # Re-merge the fighter-2 perspective of the long-format table back onto the fights table.
  f2Records = longRows.merge(
    fights[["fightIdx", "fighter2"]], left_on=["fightIdx", "fighter"], right_on=["fightIdx", "fighter2"]
  )[["fightIdx"] + recordCols]
  # Rename the columns with an "f2" prefix.
  f2Records.columns = ["fightIdx"] + [f"f2{c[0].upper()}{c[1:]}" for c in recordCols]

  # Merge the fighter-1 pre-fight record columns onto the fights table.
  fights = fights.merge(f1Records, on="fightIdx", how="left")
  # Merge the fighter-2 pre-fight record columns onto the fights table.
  fights = fights.merge(f2Records, on="fightIdx", how="left")

  # ---- In-fight performance stats (strikes, takedowns, control time, knockdowns) ----
  # These CANNOT be used as-is (that fight's own stats only exist after the fight happens -
  # using them to predict that same fight would be leakage). Instead, the same "entering" pattern
  # used above for win/loss record is applied to them: each fighter's CAREER-TO-DATE average
  # performance, computed only from their earlier fights, is a legitimate pre-fight signal - it's
  # "how this fighter typically performs," exactly analogous to the basketball pipeline's rolling
  # pre-game team averages.
  statCols = ["sigStrLanded", "sigStrAttempted", "tdLanded", "tdAttempted", "ctrlSeconds", "kd", "subAtt"]
  fightStats = LoadFightStats(dataDir)

  # Merge in fighter 1's own raw totals for this specific fight (still not a usable feature yet -
  # only the entering averages computed below are).
  f1OwnStatCols = {c: f"f1Own{c[0].upper()}{c[1:]}" for c in statCols}
  fights = fights.merge(
    fightStats.rename(columns={"FIGHTER": "fighter1", **f1OwnStatCols}), on=["EVENT", "BOUT", "fighter1"], how="left")
  # Merge in fighter 2's own raw totals for this specific fight.
  f2OwnStatCols = {c: f"f2Own{c[0].upper()}{c[1:]}" for c in statCols}
  fights = fights.merge(
    fightStats.rename(columns={"FIGHTER": "fighter2", **f2OwnStatCols}), on=["EVENT", "BOUT", "fighter2"], how="left")

  # Build a long-format table of each fighter's own performance AND what their opponent landed
  # against them that fight ("absorbed" - a simple defensive proxy), one row per (fighter, fight).
  f1StatLong = fights[["fightIdx", "eventDate", "fighter1"] + list(f1OwnStatCols.values())].copy()
  f1StatLong.columns = ["fightIdx", "eventDate", "fighter"] + statCols
  f1StatLong["sigStrAbsorbed"] = fights["f2OwnSigStrLanded"]
  f2StatLong = fights[["fightIdx", "eventDate", "fighter2"] + list(f2OwnStatCols.values())].copy()
  f2StatLong.columns = ["fightIdx", "eventDate", "fighter"] + statCols
  f2StatLong["sigStrAbsorbed"] = fights["f1OwnSigStrLanded"]
  statLongRows = pandas.concat([f1StatLong, f2StatLong], ignore_index=True)
  # Sort the long-format table chronologically within each fighter, exactly like the win/loss one.
  statLongRows = statLongRows.sort_values(["fighter", "eventDate", "fightIdx"]).reset_index(drop=True)

  # Compute each fighter's career-to-date entering average for every performance stat, AND a
  # recent-form version over just their last ROLL_WINDOW_FIGHTS fights. The career average answers
  # "how has this fighter performed across their whole career"; the recent-form average answers
  # "how is this fighter performing lately" - a fighter whose career average looks modest but whose
  # last 5 fights show sharply improving output (or the reverse - decline, wear and tear) is
  # exactly the case where these two disagree and both versions earn their place as separate
  # features (the same career-vs-recent-form pairing the basketball pipeline already uses).
  allStatCols = statCols + ["sigStrAbsorbed"]
  for statCol in allStatCols:
    statLongRows[f"{statCol}Entering"] = statLongRows.groupby("fighter", group_keys=False).apply(
      lambda g, c=statCol: ComputeEnteringMean(g, c))
    statLongRows[f"{statCol}Roll{ROLL_WINDOW_FIGHTS}"] = statLongRows.groupby("fighter", group_keys=False).apply(
      lambda g, c=statCol: ComputeEnteringMean(g, c, window=ROLL_WINDOW_FIGHTS))
  enteringStatCols = [f"{c}Entering" for c in allStatCols] + [f"{c}Roll{ROLL_WINDOW_FIGHTS}" for c in allStatCols]

  # Re-merge the fighter-1 perspective of the entering-stats table back onto the fights table.
  f1EnteringStats = statLongRows.merge(
    fights[["fightIdx", "fighter1"]], left_on=["fightIdx", "fighter"], right_on=["fightIdx", "fighter1"]
  )[["fightIdx"] + enteringStatCols]
  f1EnteringStats.columns = ["fightIdx"] + [f"f1{c[0].upper()}{c[1:]}" for c in enteringStatCols]
  # Re-merge the fighter-2 perspective of the entering-stats table back onto the fights table.
  f2EnteringStats = statLongRows.merge(
    fights[["fightIdx", "fighter2"]], left_on=["fightIdx", "fighter"], right_on=["fightIdx", "fighter2"]
  )[["fightIdx"] + enteringStatCols]
  f2EnteringStats.columns = ["fightIdx"] + [f"f2{c[0].upper()}{c[1:]}" for c in enteringStatCols]
  # Merge both fighters' entering performance-stat averages onto the fights table.
  fights = fights.merge(f1EnteringStats, on="fightIdx", how="left")
  fights = fights.merge(f2EnteringStats, on="fightIdx", how="left")

  # Derive career striking/takedown accuracy from the entering landed/attempted averages (this
  # is mathematically identical to cumulative landed / cumulative attempted, since both averages
  # share the same fight-count denominator). Do the same for the recent-form (last-5) versions.
  rollSuffix = f"Roll{ROLL_WINDOW_FIGHTS}"
  fights["f1SigStrAccEntering"] = fights["f1SigStrLandedEntering"] / fights["f1SigStrAttemptedEntering"]
  fights["f2SigStrAccEntering"] = fights["f2SigStrLandedEntering"] / fights["f2SigStrAttemptedEntering"]
  fights["f1TdAccEntering"] = fights["f1TdLandedEntering"] / fights["f1TdAttemptedEntering"]
  fights["f2TdAccEntering"] = fights["f2TdLandedEntering"] / fights["f2TdAttemptedEntering"]
  fights[f"f1SigStrAcc{rollSuffix}"] = fights[f"f1SigStrLanded{rollSuffix}"] / fights[f"f1SigStrAttempted{rollSuffix}"]
  fights[f"f2SigStrAcc{rollSuffix}"] = fights[f"f2SigStrLanded{rollSuffix}"] / fights[f"f2SigStrAttempted{rollSuffix}"]
  fights[f"f1TdAcc{rollSuffix}"] = fights[f"f1TdLanded{rollSuffix}"] / fights[f"f1TdAttempted{rollSuffix}"]
  fights[f"f2TdAcc{rollSuffix}"] = fights[f"f2TdLanded{rollSuffix}"] / fights[f"f2TdAttempted{rollSuffix}"]

  # Load the static fighter physical attributes table.
  attributes = LoadFighterAttributes(dataDir)
  # Build the fighter-1 attribute columns with an "f1" prefix.
  f1Attributes = attributes.add_prefix("f1").rename(columns={"f1FIGHTER": "fighter1"})
  # Build the fighter-2 attribute columns with an "f2" prefix.
  f2Attributes = attributes.add_prefix("f2").rename(columns={"f2FIGHTER": "fighter2"})
  # Merge the fighter-1 attributes onto the fights table.
  fights = fights.merge(f1Attributes, on="fighter1", how="left")
  # Merge the fighter-2 attributes onto the fights table.
  fights = fights.merge(f2Attributes, on="fighter2", how="left")
  # Defragment before the long run of individual column assignments below (age, diffs, etc.) -
  # otherwise pandas repeatedly re-fragments the frame from all the merges above and prior
  # per-column assignments, the same PerformanceWarning fixed the same way in BasketballFeatures.py.
  fights = fights.copy()

  # Compute fighter 1's age at the time of the fight.
  fights["f1AgeAtFight"] = (fights["eventDate"] - fights["f1dob"]).dt.days / 365.25
  # Compute fighter 2's age at the time of the fight.
  fights["f2AgeAtFight"] = (fights["eventDate"] - fights["f2dob"]).dt.days / 365.25

  # Compute the symmetric reach difference feature.
  fights["reachDiff"] = fights["f1reachIn"] - fights["f2reachIn"]
  # Compute the symmetric height difference feature.
  fights["heightDiff"] = fights["f1heightIn"] - fights["f2heightIn"]
  # Compute the symmetric age difference feature.
  fights["ageDiff"] = fights["f1AgeAtFight"] - fights["f2AgeAtFight"]
  # Compute the symmetric win percentage difference feature.
  fights["winPctDiff"] = fights["f1WinPctEntering"] - fights["f2WinPctEntering"]
  # Compute the symmetric experience (fight count) difference feature.
  fights["experienceDiff"] = fights["f1FightsEntering"] - fights["f2FightsEntering"]
  # Compute the symmetric streak difference feature.
  fights["streakDiff"] = fights["f1StreakEntering"] - fights["f2StreakEntering"]
  # Compute the symmetric finishing-tendency/layoff/activity difference features (entering each fight).
  fights["finishRateDiff"] = fights["f1FinishRateEntering"] - fights["f2FinishRateEntering"]
  fights["koWinRateDiff"] = fights["f1KoWinRateEntering"] - fights["f2KoWinRateEntering"]
  fights["subWinRateDiff"] = fights["f1SubWinRateEntering"] - fights["f2SubWinRateEntering"]
  fights["avgRoundDiff"] = fights["f1AvgRoundEntering"] - fights["f2AvgRoundEntering"]
  fights["layoffDiff"] = fights["f1DaysSinceLastFightEntering"] - fights["f2DaysSinceLastFightEntering"]
  fights["activityRateDiff"] = fights["f1ActivityRateEntering"] - fights["f2ActivityRateEntering"]

  # Compute the symmetric career-performance difference features (entering each fight), and the
  # matching recent-form (last-5-fights) difference features.
  fights["sigStrLandedDiff"] = fights["f1SigStrLandedEntering"] - fights["f2SigStrLandedEntering"]
  fights["sigStrAccDiff"] = fights["f1SigStrAccEntering"] - fights["f2SigStrAccEntering"]
  fights["sigStrAbsorbedDiff"] = fights["f1SigStrAbsorbedEntering"] - fights["f2SigStrAbsorbedEntering"]
  fights["tdLandedDiff"] = fights["f1TdLandedEntering"] - fights["f2TdLandedEntering"]
  fights["tdAccDiff"] = fights["f1TdAccEntering"] - fights["f2TdAccEntering"]
  fights["ctrlSecondsDiff"] = fights["f1CtrlSecondsEntering"] - fights["f2CtrlSecondsEntering"]
  fights["kdDiff"] = fights["f1KdEntering"] - fights["f2KdEntering"]
  fights["subAttDiff"] = fights["f1SubAttEntering"] - fights["f2SubAttEntering"]
  fights[f"sigStrLanded{rollSuffix}Diff"] = fights[f"f1SigStrLanded{rollSuffix}"] - fights[f"f2SigStrLanded{rollSuffix}"]
  fights[f"sigStrAcc{rollSuffix}Diff"] = fights[f"f1SigStrAcc{rollSuffix}"] - fights[f"f2SigStrAcc{rollSuffix}"]
  fights[f"sigStrAbsorbed{rollSuffix}Diff"] = fights[f"f1SigStrAbsorbed{rollSuffix}"] - fights[f"f2SigStrAbsorbed{rollSuffix}"]
  fights[f"tdLanded{rollSuffix}Diff"] = fights[f"f1TdLanded{rollSuffix}"] - fights[f"f2TdLanded{rollSuffix}"]
  fights[f"tdAcc{rollSuffix}Diff"] = fights[f"f1TdAcc{rollSuffix}"] - fights[f"f2TdAcc{rollSuffix}"]
  fights[f"ctrlSeconds{rollSuffix}Diff"] = fights[f"f1CtrlSeconds{rollSuffix}"] - fights[f"f2CtrlSeconds{rollSuffix}"]
  fights[f"kd{rollSuffix}Diff"] = fights[f"f1Kd{rollSuffix}"] - fights[f"f2Kd{rollSuffix}"]
  fights[f"subAtt{rollSuffix}Diff"] = fights[f"f1SubAtt{rollSuffix}"] - fights[f"f2SubAtt{rollSuffix}"]

  # Build a simple stance-matchup label (e.g. "Orthodox_Southpaw") - southpaw-vs-orthodox and
  # similar stance matchups are a known stylistic factor in how a fight plays out. Left as a
  # plain string column: the pipeline auto-detects and one-hot encodes object-dtype columns
  # exactly like weightClassClean below, so no extra registration is needed.
  fights["stanceMatchup"] = (
    fights["f1stance"].fillna("Unknown") + "_" + fights["f2stance"].fillna("Unknown")
  )

  # Collapse the raw weight class string down to a canonical division.
  fights["weightClassClean"] = fights["WEIGHTCLASS"].apply(CanonicalizeWeightClass)
  # Flag whether this bout was contested for a title.
  fights["isTitleFight"] = fights["WEIGHTCLASS"].str.contains("Title", na=False).astype(int)
  # Parse how many rounds this bout was scheduled for (3 or 5, almost always) - this is set
  # when the fight is booked, so it's known well before the fight and is not leakage. It matters
  # a lot for predicting roundClass specifically: a fight that goes to a Decision essentially
  # always ends in its LAST scheduled round (round 3 of a 3-round fight, round 5 of a 5-round
  # fight), so without this feature the model has to guess that from "isTitleFight" alone, which
  # only flags about half of all actual 5-round fights (many 5-round main events aren't title
  # fights) - scheduledRounds captures it directly and completely instead.
  fights["scheduledRounds"] = fights["TIME FORMAT"].apply(ParseScheduledRounds)

  # Define the final list of columns exposed to the modeling pipeline.
  featureCols = [
    "f1heightIn", "f1reachIn", "f1weightLbs", "f1AgeAtFight", "f1stance",
    "f2heightIn", "f2reachIn", "f2weightLbs", "f2AgeAtFight", "f2stance",
    "f1WinsEntering", "f1LossesEntering", "f1FightsEntering", "f1WinPctEntering", "f1StreakEntering",
    "f2WinsEntering", "f2LossesEntering", "f2FightsEntering", "f2WinPctEntering", "f2StreakEntering",
    # Career finishing tendency, layoff, and activity level entering each fight.
    "f1FinishRateEntering", "f1KoWinRateEntering", "f1SubWinRateEntering", "f1AvgRoundEntering",
    "f1DaysSinceLastFightEntering", "f1ActivityRateEntering",
    "f2FinishRateEntering", "f2KoWinRateEntering", "f2SubWinRateEntering", "f2AvgRoundEntering",
    "f2DaysSinceLastFightEntering", "f2ActivityRateEntering",
    # Career-to-date in-fight performance stats entering each fight (never that fight's own stats -
    # see the "In-fight performance stats" block above for why this isn't leakage).
    "f1SigStrLandedEntering", "f1SigStrAccEntering", "f1SigStrAbsorbedEntering",
    "f1TdLandedEntering", "f1TdAccEntering", "f1CtrlSecondsEntering", "f1KdEntering", "f1SubAttEntering",
    "f2SigStrLandedEntering", "f2SigStrAccEntering", "f2SigStrAbsorbedEntering",
    "f2TdLandedEntering", "f2TdAccEntering", "f2CtrlSecondsEntering", "f2KdEntering", "f2SubAttEntering",
    # Recent-form (last-5-fights) versions of the same in-fight performance stats - see the
    # "career-vs-recent-form" comment above for why both windows are kept as separate features.
    f"f1SigStrLandedRoll{ROLL_WINDOW_FIGHTS}", f"f1SigStrAccRoll{ROLL_WINDOW_FIGHTS}",
    f"f1SigStrAbsorbedRoll{ROLL_WINDOW_FIGHTS}", f"f1TdLandedRoll{ROLL_WINDOW_FIGHTS}", f"f1TdAccRoll{ROLL_WINDOW_FIGHTS}",
    f"f1CtrlSecondsRoll{ROLL_WINDOW_FIGHTS}", f"f1KdRoll{ROLL_WINDOW_FIGHTS}", f"f1SubAttRoll{ROLL_WINDOW_FIGHTS}",
    f"f2SigStrLandedRoll{ROLL_WINDOW_FIGHTS}", f"f2SigStrAccRoll{ROLL_WINDOW_FIGHTS}",
    f"f2SigStrAbsorbedRoll{ROLL_WINDOW_FIGHTS}", f"f2TdLandedRoll{ROLL_WINDOW_FIGHTS}", f"f2TdAccRoll{ROLL_WINDOW_FIGHTS}",
    f"f2CtrlSecondsRoll{ROLL_WINDOW_FIGHTS}", f"f2KdRoll{ROLL_WINDOW_FIGHTS}", f"f2SubAttRoll{ROLL_WINDOW_FIGHTS}",
    "reachDiff", "heightDiff", "ageDiff", "winPctDiff", "experienceDiff", "streakDiff",
    "finishRateDiff", "koWinRateDiff", "subWinRateDiff", "avgRoundDiff", "layoffDiff", "activityRateDiff",
    "sigStrLandedDiff", "sigStrAccDiff", "sigStrAbsorbedDiff", "tdLandedDiff", "tdAccDiff",
    "ctrlSecondsDiff", "kdDiff", "subAttDiff",
    f"sigStrLandedRoll{ROLL_WINDOW_FIGHTS}Diff", f"sigStrAccRoll{ROLL_WINDOW_FIGHTS}Diff",
    f"sigStrAbsorbedRoll{ROLL_WINDOW_FIGHTS}Diff", f"tdLandedRoll{ROLL_WINDOW_FIGHTS}Diff",
    f"tdAccRoll{ROLL_WINDOW_FIGHTS}Diff", f"ctrlSecondsRoll{ROLL_WINDOW_FIGHTS}Diff",
    f"kdRoll{ROLL_WINDOW_FIGHTS}Diff", f"subAttRoll{ROLL_WINDOW_FIGHTS}Diff",
    "weightClassClean", "isTitleFight", "scheduledRounds", "stanceMatchup",
    "roundClass", "methodClass",
  ]
  # Select and return only the final feature columns.
  return fights[featureCols].copy()


# ============================================================================================
# LIVE-MATCHUP SNAPSHOT (used by the predictor web app, never by training)
# ============================================================================================
# BuildUFCDataset() above always computes "entering" features that EXCLUDE the fight being
# predicted - correct for training, where that fight's real outcome is sitting right there in
# the same row and must never leak in. A live predictor UI has the opposite job: given two real
# fighters, estimate what their features would be if they fought again TODAY. That's exactly the
# fighter's full known history up through their most recent fight - no exclusion needed, since
# there's no "current row" to leak from. BuildCurrentFighterSnapshot() below reloads the same raw
# CSVs and applies the same parsing helpers as BuildUFCDataset(), but aggregates each fighter's
# ENTIRE history (or last ROLL_WINDOW_FIGHTS fights, for the recent-form columns) into one row per
# fighter, using the exact same column names BuildUFCDataset() prefixes with "f1"/"f2" - so the
# predictor app can build a matchup row by simply looking up two fighters and prefixing.
# Kept as a fully independent read-and-recompute of the raw CSVs (rather than refactoring
# BuildUFCDataset() to share internals) so a bug here can never affect the leakage-safe training
# path above, which is already tested and in production use.

# Define the function that computes a fighter's win/loss streak AFTER all of their recorded
# fights (as opposed to ComputeStreakEntering, which returns the streak BEFORE each fight).
def ComputeFinalStreak(subFrame: pandas.DataFrame) -> int:
  # Initialize the running streak counter.
  currentStreak = 0
  # Iterate through the fighter's results in chronological order.
  for result in subFrame["result"]:
    # Check if the fighter won this fight.
    if (result == "W"):
      # Extend a positive streak, or start a new one.
      currentStreak = currentStreak + 1 if (currentStreak >= 0) else 1
    elif (result == "L"):
      # Extend a negative streak, or start a new one.
      currentStreak = currentStreak - 1 if (currentStreak <= 0) else -1
    else:
      # Reset the streak on a draw/no-contest.
      currentStreak = 0
  # Return the final streak value after the last recorded fight.
  return currentStreak


# The values BuildUFCDataset() produces for a fighter's VERY FIRST recorded fight, where there is
# no prior history to average. Verified against the real dataset: of its 8694 rows, 923 have
# f1FightsEntering == 0 (510 have both fighters debuting), and on every one of them the counting
# columns are 0, win percentage falls back to an even 0.5 prior, days-since-last-fight falls back
# to 180, and EVERY rate/in-fight-performance column is NaN (the training pipeline then fills
# those NaNs with the training split's per-column median before scaling). A debut fighter added
# to the snapshot below is given exactly this profile so the model sees the same kind of row it
# was trained on, rather than a fabricated one.
DEBUT_WIN_PCT = 0.5
DEBUT_LAYOFF_DAYS = 180.0
# The snapshot columns that are known-zero for a debutant rather than unknown.
DEBUT_ZERO_COLS = ["WinsEntering", "LossesEntering", "FightsEntering", "StreakEntering"]
# The snapshot columns that carry no numeric feature meaning and are simply blank for a debutant.
SNAPSHOT_NON_FEATURE_COLS = ["LastWeightClass", "LastFightDate", "heightIn", "reachIn",
                             "weightLbs", "dob", "stance", "HasFightHistory"]


# Define the function that computes a median weighted by each fighter's number of fights.
# This matters for choosing a debutant's stand-in stats. Training imputed a debut row's NaNs with
# the median over training ROWS, and each row is one fighter-appearance - so a fighter with 20
# fights contributes 20 times as much to that median as a fighter with 1. Taking a plain median
# over the snapshot's one-row-per-fighter table instead over-weights the many one-and-done
# fighters and lands well below what training actually used (e.g. a plain median puts
# KoWinRateEntering at 0.000 and SigStrLandedEntering at 28.2, against training's 0.154 and 33.7).
# Weighting by fight count reproduces training's figures closely (mean error ~6% vs ~24%) without
# having to build the whole 26-second training dataset just to read its medians.
def WeightedMedian(values: pandas.Series, weights: pandas.Series) -> float:
  # Drop entries with no value to contribute.
  valueArr = numpy.asarray(values, dtype=float)
  weightArr = numpy.asarray(weights, dtype=float)
  validMask = ~numpy.isnan(valueArr) & ~numpy.isnan(weightArr) & (weightArr > 0)
  valueArr, weightArr = valueArr[validMask], weightArr[validMask]
  # Fall back to a plain median if weighting is impossible.
  if (len(valueArr) == 0):
    return float(numpy.nanmedian(numpy.asarray(values, dtype=float))) if (len(values) > 0) else 0.0
  # Sort by value and find where the cumulative weight crosses the halfway point.
  sortOrder = numpy.argsort(valueArr)
  valueArr, weightArr = valueArr[sortOrder], weightArr[sortOrder]
  cumulativeWeight = numpy.cumsum(weightArr)
  return float(valueArr[numpy.searchsorted(cumulativeWeight, cumulativeWeight[-1] / 2.0)])


# Define the function that maps each fighter's bout-string display name (the "First Last" form
# used in ufc_fight_results.csv's BOUT column, assembled from ufc_fighter_details.csv) to the
# FIGHTER key used in ufc_fighter_tott.csv, joining the two on their shared fighter-details URL.
# These two spellings disagree for 35 fighters - e.g. bouts say "Joe Kropschot" while the
# attributes table says "Joseph Kropschot", and "Kai Kamaka III" vs "Kai Kamaka" - so without
# this map their height/reach/weight/DOB/stance silently fall back to the league median.
def BuildFighterNameAliasMap(dataDir: str = UFC_DATA_DIR) -> Dict[str, str]:
  # Read the two fighter tables, which share a per-fighter URL.
  details = pandas.read_csv(os.path.join(dataDir, "ufc_fighter_details.csv"))
  tott = pandas.read_csv(os.path.join(dataDir, "ufc_fighter_tott.csv"))
  # Rebuild the "First Last" display name exactly the way bout strings spell it.
  details["displayName"] = (
    details["FIRST"].fillna("").astype(str).str.strip() + " " +
    details["LAST"].fillna("").astype(str).str.strip()
  ).str.strip()
  # Join the two spellings on the shared URL.
  merged = details[["URL", "displayName"]].merge(tott[["URL", "FIGHTER"]], on="URL", how="inner")
  merged = merged[(merged["displayName"] != "") & merged["FIGHTER"].notna()]
  # Return only the pairs where the two spellings actually differ.
  return {
    displayName: str(attrKey).strip()
    for displayName, attrKey in zip(merged["displayName"], merged["FIGHTER"])
    if (displayName != str(attrKey).strip())
  }


# Define the function that returns the full known UFC fighter roster - every fighter who has a
# row in ufc_fighter_tott.csv, whether or not they have ever actually fought - as a frame of
# (displayName, attrKey) pairs. displayName is the bout-string spelling the predictor UI shows
# and the user types; attrKey is the ufc_fighter_tott.csv key their physical attributes sit under.
def BuildFighterRoster(dataDir: str = UFC_DATA_DIR) -> pandas.DataFrame:
  # Read the two fighter tables, which share a per-fighter URL.
  details = pandas.read_csv(os.path.join(dataDir, "ufc_fighter_details.csv"))
  tott = pandas.read_csv(os.path.join(dataDir, "ufc_fighter_tott.csv"))
  details["displayName"] = (
    details["FIRST"].fillna("").astype(str).str.strip() + " " +
    details["LAST"].fillna("").astype(str).str.strip()
  ).str.strip()
  # Start from the attributes table (the widest roster) and attach the bout-string spelling.
  roster = tott[["URL", "FIGHTER"]].drop_duplicates(subset=["FIGHTER"], keep="first").copy()
  roster = roster.merge(details[["URL", "displayName"]].drop_duplicates(subset=["URL"]),
                        on="URL", how="left")
  roster["attrKey"] = roster["FIGHTER"].astype(str).str.strip()
  # Prefer the bout-string spelling; fall back to the attributes spelling when there is no
  # ufc_fighter_details.csv row for this fighter at all.
  roster["displayName"] = roster["displayName"].fillna("").replace({"": numpy.nan})
  roster["displayName"] = roster["displayName"].fillna(roster["attrKey"])
  roster = roster[roster["displayName"].astype(str).str.strip() != ""]
  # Return one row per fighter, deduplicated on the name the UI will show.
  return roster[["displayName", "attrKey"]].drop_duplicates(subset=["displayName"], keep="first")


# Define the function that builds one row per fighter with their up-to-date, as-of-today feature
# values - i.e. what BuildUFCDataset() would compute as their "entering" features for a
# hypothetical next fight. See the module-level comment above for why this is a separate,
# independent computation from BuildUFCDataset() rather than a shared code path.
def BuildCurrentFighterSnapshot(dataDir: str = UFC_DATA_DIR) -> pandas.DataFrame:
  # ---- Reload and clean the fight history, mirroring BuildUFCDataset()'s own cleaning rules ----
  events = pandas.read_csv(os.path.join(dataDir, "ufc_event_details.csv"))
  events["eventDate"] = pandas.to_datetime(events["DATE"], errors="coerce")
  results = pandas.read_csv(os.path.join(dataDir, "ufc_fight_results.csv"))
  fights = results.merge(events[["EVENT", "eventDate", "STATUS"]], on="EVENT", how="left")
  fights = fights[fights["STATUS"] != "UPCOMING"].copy()
  fights = fights.dropna(subset=["eventDate"]).copy()
  fights[["fighter1", "fighter2"]] = fights["BOUT"].apply(SplitBoutString)
  fights = fights.dropna(subset=["fighter1", "fighter2"])
  outcomeParts = fights["OUTCOME"].str.split("/", expand=True)
  fights["f1Result"] = outcomeParts[0]
  fights["f2Result"] = outcomeParts[1]
  fights = fights[~fights["OUTCOME"].isin(BAD_OUTCOMES)]
  fights = fights[~fights["METHOD"].isin(BAD_METHODS)]
  fights["methodClass"] = fights["METHOD"].map(METHOD_MAP)
  fights = fights.dropna(subset=["methodClass"])
  fights["roundClass"] = pandas.to_numeric(fights["ROUND"], errors="coerce")
  fights = fights.dropna(subset=["roundClass"])
  fights = fights.sort_values("eventDate").reset_index(drop=True)
  fights["fightIdx"] = fights.index
  # Also carry each fighter's most recent WEIGHTCLASS - used as the UI's default weight-class
  # suggestion for a hypothetical matchup (still overridable by the person using the app).
  fights["weightClassClean"] = fights["WEIGHTCLASS"].apply(CanonicalizeWeightClass)

  # ---- Build the long-format fighter-appearance table (one row per fighter per past fight) ----
  longRows = pandas.concat([
    fights[["fightIdx", "eventDate", "fighter1", "f1Result", "weightClassClean"]].rename(
      columns={"fighter1": "fighter", "f1Result": "result"}),
    fights[["fightIdx", "eventDate", "fighter2", "f2Result", "weightClassClean"]].rename(
      columns={"fighter2": "fighter", "f2Result": "result"}),
  ], ignore_index=True)
  longRows = longRows.merge(fights[["fightIdx", "methodClass", "roundClass"]], on="fightIdx", how="left")
  longRows = longRows.sort_values(["fighter", "eventDate", "fightIdx"]).reset_index(drop=True)
  longRows["winFlag"] = (longRows["result"] == "W").astype(int)
  longRows["lossFlag"] = (longRows["result"] == "L").astype(int)
  longRows["finishFlag"] = (longRows["methodClass"] != "Decision").astype(int)
  longRows["koWinFlag"] = ((longRows["result"] == "W") & (longRows["methodClass"] == "KO/TKO")).astype(int)
  longRows["subWinFlag"] = ((longRows["result"] == "W") & (longRows["methodClass"] == "Submission")).astype(int)

  fighterGroup = longRows.groupby("fighter")
  snapshot = pandas.DataFrame(index=fighterGroup.size().index)
  snapshot["WinsEntering"] = fighterGroup["winFlag"].sum()
  snapshot["LossesEntering"] = fighterGroup["lossFlag"].sum()
  snapshot["FightsEntering"] = fighterGroup.size()
  snapshot["WinPctEntering"] = snapshot["WinsEntering"] / snapshot["FightsEntering"]
  snapshot["StreakEntering"] = longRows.groupby("fighter", group_keys=False).apply(ComputeFinalStreak)
  snapshot["FinishRateEntering"] = fighterGroup["finishFlag"].mean()
  snapshot["KoWinRateEntering"] = fighterGroup["koWinFlag"].mean()
  snapshot["SubWinRateEntering"] = fighterGroup["subWinFlag"].mean()
  snapshot["AvgRoundEntering"] = fighterGroup["roundClass"].mean()
  # Most recently used weight class, and most recent fight date (for later "how stale is this
  # snapshot" display, and to compute age-at-hypothetical-fight relative to a fresh event date).
  lastFightRows = longRows.sort_values(["fighter", "eventDate", "fightIdx"]).groupby("fighter").tail(1)
  snapshot["LastWeightClass"] = lastFightRows.set_index("fighter")["weightClassClean"]
  snapshot["LastFightDate"] = lastFightRows.set_index("fighter")["eventDate"]
  # Typical days between this fighter's own fights (their historical pace) - used as a stand-in
  # for "days since last fight" in a hypothetical next fight, since the real future date is
  # unknown. A fighter with only one recorded fight has no gap to measure, so falls back to a
  # league-wide typical value (180 days) rather than an artificial 0.
  gaps = longRows.groupby("fighter")["eventDate"].diff().dt.days
  snapshot["DaysSinceLastFightEntering"] = gaps.groupby(longRows["fighter"]).mean().reindex(snapshot.index)
  snapshot["DaysSinceLastFightEntering"] = snapshot["DaysSinceLastFightEntering"].fillna(180).clip(lower=0, upper=2000)
  # Career activity rate (fights per year of active career to date).
  firstFightDate = fighterGroup["eventDate"].min()
  lastFightDate = fighterGroup["eventDate"].max()
  yearsSinceDebut = (lastFightDate - firstFightDate).dt.days / 365.25
  snapshot["ActivityRateEntering"] = snapshot["FightsEntering"] / yearsSinceDebut.replace(0, numpy.nan)

  # ---- In-fight performance stats: career average AND last-ROLL_WINDOW_FIGHTS average ----
  statCols = ["sigStrLanded", "sigStrAttempted", "tdLanded", "tdAttempted", "ctrlSeconds", "kd", "subAtt"]
  fightStats = LoadFightStats(dataDir)
  f1OwnStatCols = {c: f"f1Own{c[0].upper()}{c[1:]}" for c in statCols}
  fights = fights.merge(
    fightStats.rename(columns={"FIGHTER": "fighter1", **f1OwnStatCols}), on=["EVENT", "BOUT", "fighter1"], how="left")
  f2OwnStatCols = {c: f"f2Own{c[0].upper()}{c[1:]}" for c in statCols}
  fights = fights.merge(
    fightStats.rename(columns={"FIGHTER": "fighter2", **f2OwnStatCols}), on=["EVENT", "BOUT", "fighter2"], how="left")
  f1StatLong = fights[["fightIdx", "eventDate", "fighter1"] + list(f1OwnStatCols.values())].copy()
  f1StatLong.columns = ["fightIdx", "eventDate", "fighter"] + statCols
  f1StatLong["sigStrAbsorbed"] = fights["f2OwnSigStrLanded"]
  f2StatLong = fights[["fightIdx", "eventDate", "fighter2"] + list(f2OwnStatCols.values())].copy()
  f2StatLong.columns = ["fightIdx", "eventDate", "fighter"] + statCols
  f2StatLong["sigStrAbsorbed"] = fights["f1OwnSigStrLanded"]
  statLongRows = pandas.concat([f1StatLong, f2StatLong], ignore_index=True)
  statLongRows = statLongRows.sort_values(["fighter", "eventDate", "fightIdx"]).reset_index(drop=True)

  allStatCols = statCols + ["sigStrAbsorbed"]
  statGroup = statLongRows.groupby("fighter")
  for statCol in allStatCols:
    label = f"{statCol[0].upper()}{statCol[1:]}"
    snapshot[f"{label}Entering"] = statGroup[statCol].mean().reindex(snapshot.index)
    snapshot[f"{label}Roll{ROLL_WINDOW_FIGHTS}"] = statGroup[statCol].apply(
      lambda s: s.tail(ROLL_WINDOW_FIGHTS).mean()).reindex(snapshot.index)

  rollSuffix = f"Roll{ROLL_WINDOW_FIGHTS}"
  snapshot["SigStrAccEntering"] = snapshot["SigStrLandedEntering"] / snapshot["SigStrAttemptedEntering"]
  snapshot["TdAccEntering"] = snapshot["TdLandedEntering"] / snapshot["TdAttemptedEntering"]
  snapshot[f"SigStrAcc{rollSuffix}"] = snapshot[f"SigStrLanded{rollSuffix}"] / snapshot[f"SigStrAttempted{rollSuffix}"]
  snapshot[f"TdAcc{rollSuffix}"] = snapshot[f"TdLanded{rollSuffix}"] / snapshot[f"TdAttempted{rollSuffix}"]

  # Fighters can have a handful of fights missing a stat entirely (very old events) - fill any
  # remaining gaps in the in-fight performance columns with 0 rather than leaving them NaN.
  perfCols = [c for c in snapshot.columns if ("Entering" in c or rollSuffix in c) and c not in (
    "WinPctEntering", "DaysSinceLastFightEntering", "ActivityRateEntering")]
  snapshot[perfCols] = snapshot[perfCols].fillna(0)
  snapshot["WinPctEntering"] = snapshot["WinPctEntering"].fillna(0.5)
  snapshot["ActivityRateEntering"] = snapshot["ActivityRateEntering"].fillna(0)

  # Mark every fighter gathered so far as having real fight history, before debut rows are added
  # below. The predictor app surfaces this so a zero-history prediction can be labelled as such.
  snapshot["HasFightHistory"] = True

  # ---- Static physical attributes (height/reach/weight/DOB/stance) ----
  attributes = LoadFighterAttributes(dataDir).set_index("FIGHTER")
  # The snapshot is indexed by the bout-string spelling of each name, but the attributes table is
  # keyed by its own spelling, and the two disagree for some fighters - so translate through the
  # alias map before joining. Without this, 16 fighters who HAVE fought (e.g. "Kai Kamaka III",
  # "Ben Johnston") miss the join and silently receive median height/reach/weight.
  # The alias is used ONLY to fill a gap - if the fighter's own name already matches an
  # attributes row directly, that direct match wins. The two raw files disagree about identity
  # in a few places (e.g. URLs 1f8cd763 and 4e30f276 have "Jose Montanha" and "Henrique Da Silva
  # Lopes" swapped between them), so letting the alias override a name that already resolves
  # would trade 16 fixed fighters for a handful of newly-broken ones. Gap-filling only is
  # strictly safer: every fighter who resolved before still resolves to exactly the same row.
  aliasMap = BuildFighterNameAliasMap(dataDir)
  attrKeys = pandas.Index([
    name if (name in attributes.index) else aliasMap.get(name, name)
    for name in snapshot.index
  ])
  attributeRows = attributes.reindex(attrKeys)
  for col in attributes.columns:
    snapshot[col] = attributeRows[col].to_numpy()

  # ---- Fighters on the roster who have never had a completed fight (debutants) ----
  # Every feature above is derived from completed fights, so a fighter whose only appearance is
  # an UPCOMING bout produces no row at all - which is why the predictor used to reject them as
  # "Unknown fighter". They are still legitimately predictable: the training set contains 923
  # debut rows, so the model has learned what a zero-history fighter looks like. Give each of
  # them the exact debut profile documented at DEBUT_WIN_PCT above.
  roster = BuildFighterRoster(dataDir)
  knownAttrKeys = set(attrKeys)
  newcomers = roster[
    ~roster["displayName"].isin(snapshot.index) & ~roster["attrKey"].isin(knownAttrKeys)
  ].copy()

  if (len(newcomers) > 0):
    # The remaining feature columns are the ones BuildUFCDataset() leaves NaN on a debut row and
    # the training pipeline then median-imputes. Reproduce that here using the median over every
    # fighter who HAS fought, so a debutant reads to the model as "an unremarkable fighter with
    # no record" rather than as a fighter who lands zero strikes and wins nothing.
    medianCols = [
      c for c in snapshot.columns
      if (c not in DEBUT_ZERO_COLS and c not in SNAPSHOT_NON_FEATURE_COLS
          and c not in ("WinPctEntering", "DaysSinceLastFightEntering"))
    ]
    fightCounts = snapshot["FightsEntering"]
    debutValues = {col: WeightedMedian(snapshot[col], fightCounts) for col in medianCols}
    debutValues.update({col: 0 for col in DEBUT_ZERO_COLS})
    debutValues["WinPctEntering"] = DEBUT_WIN_PCT
    debutValues["DaysSinceLastFightEntering"] = DEBUT_LAYOFF_DAYS

    debutFrame = pandas.DataFrame(
      [debutValues] * len(newcomers), index=pandas.Index(newcomers["displayName"].tolist()))
    debutFrame["HasFightHistory"] = False
    debutFrame["LastWeightClass"] = numpy.nan
    debutFrame["LastFightDate"] = pandas.NaT
    # Attach each newcomer's physical attributes under their own attributes-table key.
    newcomerAttrs = attributes.reindex(pandas.Index(newcomers["attrKey"].tolist()))
    for col in attributes.columns:
      debutFrame[col] = newcomerAttrs[col].to_numpy()
    snapshot = pandas.concat([snapshot, debutFrame[snapshot.columns]])

  # A handful of fighters are missing physical measurements in the raw data - fall back to the
  # league-wide median rather than leaving a gap that would badly distort a live prediction.
  for col in ["heightIn", "reachIn", "weightLbs"]:
    snapshot[col] = snapshot[col].fillna(attributes[col].median())

  # Return one row per fighter, ready for the predictor app to look up by name.
  snapshot.index.name = "fighter"
  return snapshot


# Define the main execution block for standalone testing of this module.
if (__name__ == "__main__"):
  # Build the UFC dataset.
  ufcDataset = BuildUFCDataset()
  # Print the shape of the resulting dataset.
  fprint(f"UFC dataset shape: {ufcDataset.shape}")
  # Print the method class distribution.
  fprint(ufcDataset["methodClass"].value_counts())
  # Print the round class distribution.
  fprint(ufcDataset["roundClass"].value_counts())
