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
    ground_df = pd.read_csv(ground_file)
    squad_df = pd.read_csv(squad_file)
    form_df = pd.read_csv(form_file)  # Loading form_file ("data/merged_output.csv")

    # Clean dataframes
    form_df = form_df.drop(["Fielding Form", "Credits", "Player Type", "Team"], axis=1, errors='ignore')
    squad_df = squad_df.drop("Credits", axis=1, errors='ignore')

    # Filter for playing players
    playing_df = squad_df[squad_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]

    # Merge squad and form data
    selection_df = pd.merge(playing_df, form_df, left_on="Player Name", right_on="Player", how="left")
    selection_df = selection_df[["Player Name", "Team", "Player Type", "Batting Form", "Bowling Form"]]
    selection_df.columns = ["player_name", "team", "player_type", "batting_form", "bowling_form"]

    # Fill missing values with 0
    selection_df["batting_form"] = selection_df["batting_form"].fillna(0)
    selection_df["bowling_form"] = selection_df["bowling_form"].fillna(0)

    # Filter for selected teams
    selection_df = selection_df[selection_df["team"].isin([team1, team2])]

    # Get ground weights
    ground_data = ground_df.iloc[ground_index]
    batter_weight = float(ground_data['Batting'])
    keeper_weight = float(ground_data['Batting'])
    bowler_weight = float(ground_data['Bowling'])
    allrounder_weight = (batter_weight + bowler_weight) / 2
    if allrounder_weight < 1:
        allrounder_weight = 1.0

    # Calculate scores with strong emphasis on recent form
    form_scale_factor = 2.5  # Increased to 2.5 for stronger form weighting
    def calculate_score(row):
        role = row["player_type"].strip().upper()
        batting = row['batting_form'] * form_scale_factor
        bowling = row['bowling_form'] * form_scale_factor
        if role == "BAT":
            return batter_weight * batting
        if role == "WK":
            return keeper_weight * batting
        if role == "BOWL":
            return bowler_weight * bowling
        if role == "ALL":
            return allrounder_weight * max(batting, bowling)
        return 0

    selection_df["score"] = selection_df.apply(calculate_score, axis=1)

    # Apply home/away weights
    home_weight = 1.05
    away_weight = 1.0
    selection_df["adjusted_score"] = selection_df.apply(
        lambda row: row["score"] * (home_weight if row["team"] == team1 else away_weight), axis=1)

    # Add credits (default 9.0 if not available)
    selection_df["credits"] = 9.0  # Replace with actual credits if available

    # Log top players by adjusted_score
    top_players = selection_df.sort_values("adjusted_score", ascending=False).head(10)
    logger.info("Top 10 players by adjusted_score before optimization:")
    logger.info(top_players[["player_name", "team", "player_type", "adjusted_score"]].to_string())

    return selection_df

class Dream11Problem(ElementwiseProblem):
    def __init__(self, players, n_players, team1, team2):
        super().__init__(
            n_var=n_players,
            n_obj=3,
            n_constr=9,
            xl=0,
            xu=1,
            vtype=bool
        )
        self.players = players
        self.team1 = team1
        self.team2 = team2

    def _evaluate(self, x, out, *args, **kwargs):
        team_df = self.players[x.astype(bool)]
        total_credits = team_df["credits"].sum()
        n_players = len(team_df)
        n_wk = team_df["player_type"].eq("WK").sum()
        n_bat = team_df["player_type"].eq("BAT").sum()
        n_bowl = team_df["player_type"].eq("BOWL").sum()
        n_all = team_df["player_type"].eq("ALL").sum()
        n_team1 = team_df["team"].eq(self.team1).sum()
        n_team2 = team_df["team"].eq(self.team2).sum()
        n_bowling_options = n_bowl + n_all

        # Objectives: Maximize adjusted score, batting form, bowling form
        total_score = team_df["adjusted_score"].sum()
        total_batting = team_df["batting_form"].sum()
        total_bowling = team_df["bowling_form"].sum()

        # Negate objectives for minimization
        out["F"] = [-total_score, -total_batting, -total_bowling]

        # Constraints
        out["G"] = [
            total_credits - 100,  # Credits <= 100
            11 - n_players,       # Exactly 11 players
            n_wk - 4,             # Max 4 WK
            1 - n_wk,             # Min 1 WK
            n_bat - 6,            # Max 6 BAT
            n_bowl - 6,           # Max 6 BOWL
            n_all - 6,            # Max 6 ALL
            max(n_team1, n_team2) - 7,  # Max 7 from one team
            5 - n_bowling_options  # Min 5 bowling options
        ]

def optimize_team(ground_file, squad_file, form_file, output_file, team1, team2, ground_index):
    """Optimize Dream11 team using NSGA-II."""
    # Load data
    players = load_and_preprocess_data(ground_file, squad_file, form_file, ground_index, team1, team2)

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
    res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=True)

    # Extract best solution
    pareto_solutions = res.X[res.F.argmin(axis=0)[0]]  # Max total score
    selected_players = players[pareto_solutions.astype(bool)].copy()

    # Assign captain and vice-captain based on adjusted_score (reflects form)
    selected_players.sort_values("adjusted_score", ascending=False, inplace=True)
    selected_players["Role_In_Team"] = "Player"
    if len(selected_players) > 0:
        selected_players.iloc[0, selected_players.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_players) > 1:
        selected_players.iloc[1, selected_players.columns.get_loc("Role_In_Team")] = "Vice Captain"

    # Log and save team
    logger.info("Selected Dream11 Team:")
    logger.info(selected_players[["player_name", "team", "player_type", "credits", "adjusted_score", "Role_In_Team"]].to_string())
    selected_players.to_csv(output_file, index=False)

    return pareto_solutions

if __name__ == "__main__":
    ground_file = "data/ground.csv"
    squad_file = "data/SquadPlayerNames.csv"
    form_file = "data/merged_output.csv"  # Using your specified form file
    output_file = "selected_team.csv"
    team1 = "SRH"
    team2 = "PBKS"

    # Ground selection
    ground_df = pd.read_csv(ground_file)
    print("Grounds:")
    for i, r in ground_df.iterrows():
        print(f"{i + 1}. {r['Ground']} ({r['City']})")

    ground_number = int(input("\nEnter the ground number (1-13) for the match: "))
    ground_index = ground_number - 1

    # Run optimization
    optimize_team(ground_file, squad_file, form_file, output_file, team1, team2, ground_index)
    logger.info("Optimization completed.")