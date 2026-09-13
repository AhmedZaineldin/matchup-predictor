import os  # Import the os module to read the optional environment variable override.
import numpy  # Import the numpy module for numerical operations.
import pandas  # Import the pandas module for tabular data handling.

# Define the default directory containing the raw basketball CSV files, relative
# to wherever the pipeline is run from. Copy your team_box.csv and schedule.csv
# files into this folder (see README.md), or override it per-run via the
# BASKETBALL_DATA_DIR environment variable, or by passing dataDir=... into
# BuildBasketballDataset().
BASKETBALL_DATA_DIR = os.environ.get("BASKETBALL_DATA_DIR", os.path.join("Data", "Basketball"))

# Define the list of raw per-game box score columns used to build rolling pre-game features.
# Every column here goes through the exact same leakage-free treatment in BuildTeamGameHistory
# below (shift(1) before any rolling/expanding average, so a game's own numbers are never used
# to help predict itself) - adding a column here is enough to turn it into a pre-game feature.
# team_curated_rank / home_curated_rank / away_curated_rank are NOT included: this dataset has
# them 100% empty for every league, so there's nothing there to use. lead_changes and
# lead_percentage are excluded too - both are missing on ~84% of rows, too sparse to trust.
BOX_STAT_COLS = [
  "team_score", "opponent_team_score", "field_goal_pct",
  "three_point_field_goal_pct", "free_throw_pct", "total_rebounds",
  "offensive_rebounds", "defensive_rebounds", "assists", "steals",
  "blocks", "turnovers", "fouls", "points_in_paint", "fast_break_points",
  # Shot-volume/pace stats (how many shots a team typically takes/attempts per game).
  "field_goals_attempted", "three_point_field_goals_attempted", "free_throws_attempted",
  # Additional performance signals with good coverage across all 4 leagues.
  "turnover_points", "largest_lead", "technical_fouls",
]

# Define the rolling window length (in games) used for the recent-form features.
ROLL_WINDOW = 10


# Define the fprint helper so this module logs the same way as the rest of the pipeline.
def fprint(msg, *args, **kwargs):
  # Print the message immediately, flushing the output buffer.
  print(msg, flush=True, *args, **kwargs)


# Define the function that computes a team's chronological win/loss streak entering each game.
def ComputeStreakEntering(subFrame: pandas.DataFrame) -> pandas.Series:
  # Initialize the list that accumulates the streak value before each game.
  streakValues = []
  # Initialize the running streak counter.
  currentStreak = 0
  # Iterate through the team's results in chronological order.
  for winFlag in subFrame["winFlag"]:
    # Record the streak value entering this game, before updating it.
    streakValues.append(currentStreak)
    # Check if the team won this game.
    if (winFlag == 1):
      # Extend a positive streak, or start a new one.
      currentStreak = currentStreak + 1 if (currentStreak >= 0) else 1
    else:
      # Extend a negative streak, or start a new one.
      currentStreak = currentStreak - 1 if (currentStreak <= 0) else -1
  # Return the streak values aligned to the original index.
  return pandas.Series(streakValues, index=subFrame.index)


# Define the function that builds the long-format, leakage-free team-game history for one league.
def BuildTeamGameHistory(league: str, dataDir: str = BASKETBALL_DATA_DIR) -> pandas.DataFrame:
  # Read the schedule CSV file.
  schedule = pandas.read_csv(os.path.join(dataDir, "schedule.csv"), low_memory=False)
  # Filter the schedule down to the requested league.
  schedule = schedule[schedule["league"] == league].copy()
  # Parse the game date column into a proper datetime.
  schedule["game_date"] = pandas.to_datetime(schedule["game_date"], errors="coerce")

  # Read the team box score CSV file.
  teamBox = pandas.read_csv(os.path.join(dataDir, "team_box.csv"), low_memory=False)
  # Filter the team box scores down to the requested league.
  teamBox = teamBox[teamBox["league"] == league].copy()

  # Merge the team box scores with the game date and season metadata from the schedule.
  teamGames = teamBox.merge(
    schedule[["game_id", "game_date", "season", "season_type", "neutral_site", "conference_competition"]],
    on="game_id", how="inner",
  )
  # Drop games with a missing date.
  teamGames = teamGames.dropna(subset=["game_date"])
  # Convert the team_winner column into a clean 0/1 win flag.
  teamGames["winFlag"] = teamGames["team_winner"].astype(str).str.lower().map({"true": 1, "false": 0})
  # Drop games with an undetermined winner (postponed/canceled).
  teamGames = teamGames.dropna(subset=["winFlag"])
  # Cast the win flag to an integer type.
  teamGames["winFlag"] = teamGames["winFlag"].astype(int)

  # Sort the team-game history chronologically within each team.
  teamGames = teamGames.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
  # Group the team-game history by team for the rolling calculations.
  teamGroup = teamGames.groupby("team_id")

  # Iterate through each raw box-score statistic to build its pre-game rolling features.
  for col in BOX_STAT_COLS:
    # Compute the shifted last-N-game rolling average (never includes the current game).
    teamGames[f"{col}Roll10"] = teamGroup[col].transform(
      lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    # Compute the shifted season-to-date expanding average (never includes the current game).
    teamGames[f"{col}Season"] = teamGames.groupby(["team_id", "season"])[col].transform(
      lambda s: s.shift(1).expanding(min_periods=1).mean())

  # Compute the number of wins entering each game.
  teamGames["winsEntering"] = teamGroup["winFlag"].cumsum() - teamGames["winFlag"]
  # Compute the number of prior games entering each game.
  teamGames["gamesEntering"] = teamGroup.cumcount()
  # Compute the win percentage entering each game, defaulting to a neutral 0.5 prior with no history.
  teamGames["winPctEntering"] = (
    teamGames["winsEntering"] / teamGames["gamesEntering"].replace(0, numpy.nan)
  ).fillna(0.5)
  # Compute the win/loss streak entering each game.
  teamGames["streakEntering"] = teamGames.groupby("team_id", group_keys=False).apply(ComputeStreakEntering)
  # Compute the date of the team's previous game.
  teamGames["prevGameDate"] = teamGroup["game_date"].shift(1)
  # Compute the number of rest days since the team's previous game.
  teamGames["restDays"] = (teamGames["game_date"] - teamGames["prevGameDate"]).dt.days
  # Fill missing rest days (season opener) with a neutral value, and clip long off-seasons.
  teamGames["restDays"] = teamGames["restDays"].fillna(7).clip(upper=30)

  # Return the fully engineered long-format team-game history.
  return teamGames


# Define the function that builds the game-level (home vs away) modeling dataset for one league.
def BuildBasketballDataset(league: str, dataDir: str = BASKETBALL_DATA_DIR) -> pandas.DataFrame:
  # Build the long-format, leakage-free team-game history for this league.
  teamGames = BuildTeamGameHistory(league, dataDir)

  # Define the final list of pre-game feature columns (rolling + season-to-date + record + rest).
  featureCols = (
    [f"{c}Roll10" for c in BOX_STAT_COLS] +
    [f"{c}Season" for c in BOX_STAT_COLS] +
    ["winsEntering", "gamesEntering", "winPctEntering", "streakEntering", "restDays"]
  )

  # Extract the home-team perspective of every game.
  homeRows = teamGames[teamGames["team_home_away"] == "home"][
    ["game_id", "team_id", "opponent_team_id", "winFlag", "neutral_site",
     "conference_competition", "season_type"] + featureCols
  ].rename(columns={c: f"home{c[0].upper()}{c[1:]}" for c in featureCols})
  # Rename the home win flag column.
  homeRows = homeRows.rename(columns={"winFlag": "homeWinFlag"})

  # Extract the away-team perspective of every game.
  awayRows = teamGames[teamGames["team_home_away"] == "away"][["game_id"] + featureCols].rename(
    columns={c: f"away{c[0].upper()}{c[1:]}" for c in featureCols})

  # Merge the home and away perspectives into one row per game.
  games = homeRows.merge(awayRows, on="game_id", how="inner")

  # Compute the home-minus-away difference for every pre-game feature.
  diffColumns = {
    f"diff{c[0].upper()}{c[1:]}": games[f"home{c[0].upper()}{c[1:]}"] - games[f"away{c[0].upper()}{c[1:]}"]
    for c in featureCols
  }
  # Concatenate the diff columns onto the games table in one shot (avoids fragmentation warnings),
  # then defragment immediately - the individual column assignments just below would otherwise
  # re-trigger the same fragmentation warning against the freshly-concatenated frame.
  games = pandas.concat([games, pandas.DataFrame(diffColumns, index=games.index)], axis=1).copy()

  # Define the binary classification target: does the home team win.
  games["targetHomeWin"] = games["homeWinFlag"]
  # Cast the neutral site flag to an integer.
  games["neutral_site"] = games["neutral_site"].astype(int)
  # Cast the conference competition flag to an integer.
  games["conference_competition"] = games["conference_competition"].astype(int)

  # Drop the identifier columns that carry no generalizable signal.
  games = games.drop(columns=["game_id", "team_id", "opponent_team_id", "homeWinFlag"])
  # Return the finished game-level dataset.
  return games


# ============================================================================================
# LIVE-MATCHUP SNAPSHOT (used by the predictor web app, never by training)
# ============================================================================================
# BuildBasketballDataset() above always computes rolling features that EXCLUDE the game being
# predicted (shift(1) before every rolling/expanding average) - correct for training. A live
# predictor UI needs the opposite: given two real teams, what would their rolling features be if
# they played TODAY - i.e. each team's full history up through their most recently played game,
# with no exclusion needed. BuildCurrentTeamSnapshot() below reuses BuildTeamGameHistory() (the
# same tested per-game history used for training) but aggregates it into one row per team.

# Define the function that computes a team's win/loss streak AFTER all of their recorded games
# (as opposed to ComputeStreakEntering, which returns the streak BEFORE each game).
def ComputeFinalStreak(subFrame: pandas.DataFrame) -> int:
  # Initialize the running streak counter.
  currentStreak = 0
  # Iterate through the team's results in chronological order.
  for winFlag in subFrame["winFlag"]:
    # Check if the team won this game.
    if (winFlag == 1):
      # Extend a positive streak, or start a new one.
      currentStreak = currentStreak + 1 if (currentStreak >= 0) else 1
    else:
      # Extend a negative streak, or start a new one.
      currentStreak = currentStreak - 1 if (currentStreak <= 0) else -1
  # Return the final streak value after the last recorded game.
  return currentStreak


# Define the function that builds a team_id -> human-readable team name lookup for one league,
# parsed from the schedule's free-text "matchup" column (e.g. "Team A at Team B") since neither
# schedule.csv nor team_box.csv has a dedicated team-name column. A handful of team_ids have more
# than one associated string across the data (rebrands, minor formatting changes over the years),
# so the most common ("mode") name is used.
def BuildTeamNameLookup(league: str, dataDir: str = BASKETBALL_DATA_DIR) -> pandas.Series:
  # Read only the columns needed to build the lookup.
  schedule = pandas.read_csv(
    os.path.join(dataDir, "schedule.csv"), low_memory=False,
    usecols=["matchup", "home_team_id", "away_team_id", "league"])
  # Filter down to the requested league.
  schedule = schedule[schedule["league"] == league]
  # Split "Away Team Name at Home Team Name" into its two halves.
  parts = schedule["matchup"].str.split(" at ", n=1, expand=True)
  # Build a long list of (team_id, name) pairs from both the away and home side of every game.
  awayNames = pandas.DataFrame({"team_id": schedule["away_team_id"], "name": parts[0]})
  homeNames = pandas.DataFrame({"team_id": schedule["home_team_id"], "name": parts[1]})
  allNames = pandas.concat([awayNames, homeNames], ignore_index=True).dropna()
  # Return the most common name string per team_id.
  return allNames.groupby("team_id")["name"].agg(lambda s: s.mode().iloc[0] if (len(s.mode())) else s.iloc[0])


# Define the function that builds one row per team with their up-to-date, as-of-today feature
# values - i.e. what BuildBasketballDataset() would compute as their pre-game features for a
# hypothetical next game. See the module-level comment above for why this reuses
# BuildTeamGameHistory() rather than BuildBasketballDataset() itself (which only exposes the
# leakage-safe SHIFTED columns, not the raw per-game ones this needs).
def BuildCurrentTeamSnapshot(league: str, dataDir: str = BASKETBALL_DATA_DIR) -> pandas.DataFrame:
  # Reuse the same tested per-game team history used for training.
  teamGames = BuildTeamGameHistory(league, dataDir)
  # Build the team_id -> display name lookup.
  nameLookup = BuildTeamNameLookup(league, dataDir)

  # Group the per-game history by team for the aggregate ("current, as of today") calculations.
  teamGroup = teamGames.groupby("team_id")
  snapshot = pandas.DataFrame(index=teamGroup.size().index)
  snapshot.index.name = "team_id"
  fallbackNames = pandas.Series("Team " + snapshot.index.astype(str), index=snapshot.index)
  snapshot["teamName"] = nameLookup.reindex(snapshot.index).fillna(fallbackNames)
  snapshot["winsEntering"] = teamGroup["winFlag"].sum()
  snapshot["gamesEntering"] = teamGroup.size()
  snapshot["winPctEntering"] = snapshot["winsEntering"] / snapshot["gamesEntering"]
  snapshot["streakEntering"] = teamGames.sort_values(["team_id", "game_date"]).groupby(
    "team_id", group_keys=False).apply(ComputeFinalStreak)

  # Rest days: for a live "as of today" snapshot, the real elapsed time since each team's last
  # played game IS known (unlike UFC's sporadic fight calendar), so this uses the actual gap to
  # today rather than a historical average.
  lastGameDate = teamGroup["game_date"].max()
  today = pandas.Timestamp.now().normalize()
  snapshot["restDays"] = (today - lastGameDate).dt.days.fillna(7).clip(lower=0, upper=30)

  # Roll10 (last-10-games) and Season (current-season-to-date) averages of every box stat,
  # computed over each team's FULL history up through their most recent game (no shift/exclusion -
  # see the module comment above for why that's correct for a live snapshot, unlike training).
  currentSeason = teamGroup["season"].max()
  seasonMeans = teamGames.groupby(["team_id", "season"])[BOX_STAT_COLS].mean()
  seasonIdx = pandas.MultiIndex.from_arrays([currentSeason.index, currentSeason.values])
  currentSeasonStats = seasonMeans.reindex(seasonIdx)
  currentSeasonStats.index = currentSeason.index
  for col in BOX_STAT_COLS:
    snapshot[f"{col}Roll10"] = teamGroup[col].apply(lambda s: s.tail(ROLL_WINDOW).mean())
    snapshot[f"{col}Season"] = currentSeasonStats[col]

  # Return one row per team, ready for the predictor app to look up by team_id.
  return snapshot


# Define the main execution block for standalone testing of this module.
if (__name__ == "__main__"):
  # Iterate through each supported league.
  for league in ["nba", "wnba", "ncaa_mbb", "ncaa_wbb"]:
    # Build the dataset for the current league.
    leagueDataset = BuildBasketballDataset(league)
    # Print the shape and home-win rate for the current league.
    fprint(f"{league} shape={leagueDataset.shape} homeWinRate={leagueDataset['targetHomeWin'].mean():.3f}")
