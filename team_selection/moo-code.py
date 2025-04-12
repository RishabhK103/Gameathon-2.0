import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_and_preprocess_data(ground_file, squad_file, form_file, ground_index, team1, team2):
    """Load and preprocess data from CSV files."""
    try:
        ground_df = pd.read_csv(ground_file)
        squad_df = pd.read_csv(squad_file)
        form_df = pd.read_csv(form_file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    # Log columns
    logger.info(f"Squad DataFrame columns: {squad_df.columns.tolist()}")
    logger.info(f"Form DataFrame columns: {form_df.columns.tolist()}")

    # Clean form_df
    form_df = form_df.drop(["Fielding Form", "Credits", "Player Type", "Team"], axis=1, errors='ignore')

    # Filter for playing players
    #playing_df = squad_df[squad_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]
    playing_df = squad_df[squad_df["IsPlaying"].isin(["PLAYING"])]
    # Merge data
    selection_df = pd.merge(playing_df, form_df, left_on="Player Name", right_on="Player", how="left")
    logger.info(f"Merged DataFrame columns: {selection_df.columns.tolist()}")

    # Verify required columns
    required_columns = ["Player Name", "Team", "Player Type", "Batting Form", "Bowling Form"]
    missing_columns = [col for col in required_columns if col not in selection_df.columns]
    if missing_columns:
        logger.error(f"Missing columns in merged DataFrame: {missing_columns}")
        raise KeyError(f"Missing columns: {missing_columns}")

    # Select relevant columns
    selection_df = selection_df[required_columns]
    selection_df.columns = ["player_name", "team", "player_type", "batting_form", "bowling_form"]

    # Fill missing values
    selection_df["batting_form"] = selection_df["batting_form"].fillna(0)
    selection_df["bowling_form"] = selection_df["bowling_form"].fillna(0)

    # Filter for selected teams
    selection_df = selection_df[selection_df["team"].isin([team1, team2])]

    # Filter out players with very low form unless critical (e.g., WK)
    wk_players = selection_df[selection_df["player_type"] == "WK"]
    non_wk_low_form = selection_df[
        (selection_df["player_type"] != "WK") &
        (selection_df["batting_form"] < 50.0) &
        (selection_df["bowling_form"] < 50.0)
    ]
    if not non_wk_low_form.empty:
        logger.warning(f"Excluding non-WK low-form players:\n{non_wk_low_form[['player_name', 'team', 'player_type', 'batting_form', 'bowling_form']].to_string()}")
        selection_df = selection_df[~selection_df.index.isin(non_wk_low_form.index)]

    # Log players with low form
    low_form_players = selection_df[(selection_df["batting_form"] < 50.0) & (selection_df["bowling_form"] < 50.0)]
    if not low_form_players.empty:
        logger.warning(f"Remaining players with low form (<50.0):\n{low_form_players[['player_name', 'team', 'player_type', 'batting_form', 'bowling_form']].to_string()}")

    # Get ground weights
    try:
        ground_data = ground_df.iloc[ground_index]
        batter_weight = float(ground_data['Batting'])
        keeper_weight = float(ground_data['Batting'])
        bowler_weight = float(ground_data['Bowling'])
        allrounder_weight = (batter_weight + bowler_weight) / 2
        if allrounder_weight < 1:
            allrounder_weight = 1.0
    except IndexError:
        logger.error(f"Invalid ground_index: {ground_index}")
        raise

    # Calculate scores with extreme emphasis on recent form
    form_scale_factor = 100.0  # Greatly prioritize form
    low_form_threshold = 50.0
    def calculate_score(row):
        role = row["player_type"].strip().upper()
        batting = row["batting_form"] * form_scale_factor
        bowling = row["bowling_form"] * form_scale_factor
        # Severe penalty for low form
        penalty = 0.01 if (row["batting_form"] < low_form_threshold and row["bowling_form"] < low_form_threshold) else 1.0
        base_score = 1.0  # Ensure non-zero score
        if role == "BAT":
            return (batter_weight * batting + base_score) * penalty
        if role == "WK":
            return (keeper_weight * batting + base_score) * penalty
        if role == "BOWL":
            return (bowler_weight * bowling + base_score) * penalty
        if role == "ALL":
            return (allrounder_weight * max(batting, bowling) + base_score) * penalty
        return base_score * penalty

    selection_df["score"] = selection_df.apply(calculate_score, axis=1)

    # Apply home/away weights
    home_weight = 1.05
    away_weight = 1.0
    selection_df["adjusted_score"] = selection_df.apply(
        lambda row: row["score"] * (home_weight if row["team"] == team1 else away_weight), axis=1)

    # Add credits
    selection_df["credits"] = 9.0

    # Log top players
    logger.info("Top 10 players by adjusted_score before optimization:")
    top_players = selection_df.sort_values("adjusted_score", ascending=False).head(10)
    logger.info(top_players[["player_name", "team", "player_type", "batting_form", "bowling_form", "adjusted_score"]].to_string())

    # Log full player pool stats
    logger.info(f"Player pool stats: {len(selection_df)} players")
    logger.info(f"Wicketkeepers: {selection_df['player_type'].eq('WK').sum()}")
    logger.info(f"Bowlers: {selection_df['player_type'].eq('BOWL').sum()}")
    logger.info(f"All-rounders: {selection_df['player_type'].eq('ALL').sum()}")
    logger.info(f"Team {team1}: {selection_df['team'].eq(team1).sum()}, Team {team2}: {selection_df['team'].eq(team2).sum()}")

    # Reset index to ensure consistent indexing
    selection_df = selection_df.reset_index(drop=True)

    return selection_df

class Dream11Problem(ElementwiseProblem):
    def __init__(self, players, n_players, team1, team2):
        super().__init__(
            n_var=n_players,
            n_obj=1,
            n_constr=9,
            xl=0,
            xu=1,
            vtype=bool
        )
        self.players = players
        self.team1 = team1
        self.team2 = team2

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            team_df = self.players.iloc[x.astype(bool)]
        except Exception as e:
            logger.error(f"Error indexing players: {e}")
            team_df = pd.DataFrame()

        total_credits = team_df["credits"].sum() if not team_df.empty else 0.0
        n_players = len(team_df)
        n_wk = team_df["player_type"].eq("WK").sum()
        n_bat = team_df["player_type"].eq("BAT").sum()
        n_bowl = team_df["player_type"].eq("BOWL").sum()
        n_all = team_df["player_type"].eq("ALL").sum()
        n_team1 = team_df["team"].eq(self.team1).sum()
        n_team2 = team_df["team"].eq(self.team2).sum()
        n_bowling_options = n_bowl + n_all

        total_score = team_df["adjusted_score"].sum() if not team_df.empty else 0.0

        # Log for debugging
        if total_score > 0:
            logger.debug(f"Evaluated team: {n_players} players, score={total_score:.2f}")

        out["F"] = [-total_score if total_score > 0 else -1e-10]  # Small negative to avoid zero

        # Constraints
        constraints = [
            total_credits - 100.0,  # Max 100 credits
            11 - n_players,         # Exactly 11 players
            n_wk - 4,               # Max 4 WK
            1 - n_wk,               # Min 1 WK
            n_bat - 6,              # Max 6 BAT
            n_bowl - 6,             # Max 6 BOWL
            n_all - 6,              # Max 6 ALL
            max(n_team1, n_team2) - 7,  # Max 7 from one team
            3 - n_bowling_options   # Min 3 bowling options
        ]
        out["G"] = constraints

        # Log constraint violations
        if any(c > 0 for c in constraints):
            logger.debug(f"Constraint violations: credits={constraints[0]:.2f}, players={constraints[1]:.2f}, "
                         f"wk={constraints[2]:.2f}/{constraints[3]:.2f}, bat={constraints[4]:.2f}, "
                         f"bowl={constraints[5]:.2f}, all={constraints[6]:.2f}, team={constraints[7]:.2f}, "
                         f"bowling={constraints[8]:.2f}")

def optimize_team(ground_file, squad_file, form_file, output_file, team1, team2, ground_index):
    """Optimize Dream11 team using NSGA-II."""
    # Load data
    players = load_and_preprocess_data(ground_file, squad_file, form_file, ground_index, team1, team2)

    # Validate player pool
    if len(players) < 11:
        logger.error("Not enough players to form a team (need at least 11).")
        return None

    # Define problem
    problem = Dream11Problem(players, n_players=len(players), team1=team1, team2=team2)

    # Define algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=0.01),
        eliminate_duplicates=True
    )

    # Run optimization
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 300),
        seed=1,
        verbose=True
    )

    # Log objective values
    logger.info(f"Optimization result: res.F shape={res.F.shape}, min F={np.min(res.F) if res.F.size > 0 else 'N/A'}")

    # Check for valid solutions
    if res.X is None or len(res.X) == 0:
        logger.error("No solutions found by optimizer.")
        # Fallback to top 11 players
        selected_players = players.sort_values("adjusted_score", ascending=False).head(11).copy()
    else:
        # Filter non-zero objectives
        valid_indices = np.where(res.F.flatten() < -1e-5)[0]  # Negative due to maximization
        if len(valid_indices) == 0:
            logger.warning("All solutions have near-zero objective values.")
            # Fallback to top 11 players
            selected_players = players.sort_values("adjusted_score", ascending=False).head(11).copy()
        else:
            best_idx = valid_indices[np.argmin(res.F[valid_indices])]
            pareto_solutions = res.X[best_idx]
            try:
                selected_players = players.iloc[pareto_solutions.astype(bool)].copy()
            except Exception as e:
                logger.error(f"Error selecting solution: {e}")
                # Fallback
                selected_players = players.sort_values("adjusted_score", ascending=False).head(11).copy()

    # Validate team
    if len(selected_players) != 11:
        logger.warning(f"Selected team has {len(selected_players)} players instead of 11.")
        # Adjust team
        if len(selected_players) < 11:
            remaining = 11 - len(selected_players)
            additional = players[~players.index.isin(selected_players.index)].sort_values("adjusted_score", ascending=False).head(remaining)
            selected_players = pd.concat([selected_players, additional])
        elif len(selected_players) > 11:
            selected_players = selected_players.sort_values("adjusted_score", ascending=False).head(11)

    # Assign captain and vice-captain
    selected_players.sort_values("adjusted_score", ascending=False, inplace=True)
    selected_players["Role_In_Team"] = "Player"
    high_form_players = selected_players[(selected_players["batting_form"] > 80.0) | (selected_players["bowling_form"] > 80.0)]
    if len(high_form_players) >= 1:
        selected_players.loc[high_form_players.index[0], "Role_In_Team"] = "Captain"
    if len(high_form_players) >= 2:
        selected_players.loc[high_form_players.index[1], "Role_In_Team"] = "Vice Captain"

    # Log team details
    n_wk = selected_players["player_type"].eq("WK").sum()
    n_bat = selected_players["player_type"].eq("BAT").sum()
    n_bowl = selected_players["player_type"].eq("BOWL").sum()
    n_all = selected_players["player_type"].eq("ALL").sum()
    total_credits = selected_players["credits"].sum()
    n_team1 = selected_players["team"].eq(team1).sum()
    n_team2 = selected_players["team"].eq(team2).sum()
    logger.info("Selected Team Constraints: Players=%d, WK=%d, BAT=%d, BOWL=%d, ALL=%d, Credits=%.1f, Team1=%d, Team2=%d",
                len(selected_players), n_wk, n_bat, n_bowl, n_all, total_credits, n_team1, n_team2)
    logger.info("Selected Dream11 Team:")
    logger.info(selected_players[["player_name", "team", "player_type", "batting_form", "bowling_form", "adjusted_score", "Role_In_Team"]].to_string())

    # Save to CSV
    selected_players.to_csv(output_file, index=False, columns=[
        "player_name", "team", "player_type", "credits", "adjusted_score", "Role_In_Team", "batting_form", "bowling_form"
    ])

    return selected_players

if __name__ == "__main__":
    ground_file = "data/ground.csv"
    squad_file = "data/SquadPlayerNames.csv"
    form_file = "data/merged_output.csv"
    output_file = "selected_team.csv"
    team1 = "SRH"
    team2 = "PBKS"

    try:
        ground_df = pd.read_csv(ground_file)
    except FileNotFoundError:
        logger.error("Ground file not found.")
        exit(1)

    print("Grounds:")
    for i, r in ground_df.iterrows():
        print(f"{i + 1}. {r['Ground']} ({r['City']})")

    try:
        ground_number = int(input("\nEnter the ground number (1-13) for the match: "))
        if ground_number < 1 or ground_number > len(ground_df):
            raise ValueError
    except ValueError:
        print(f"Invalid input! Please enter a number between 1 and {len(ground_df)}.")
        exit(1)

    ground_index = ground_number - 1
    selected_ground = ground_df.iloc[ground_index]["Ground"]
    logger.info(f"Selected ground: {selected_ground}")

    selected_team = optimize_team(
        ground_file=ground_file,
        squad_file=squad_file,
        form_file=form_file,
        output_file=output_file,
        team1=team1,
        team2=team2,
        ground_index=ground_index
    )
    if selected_team is not None:
        logger.info("Optimization completed successfully.")
    else:
        logger.error("Optimization failed.")