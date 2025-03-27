import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

class Dream11Problem(ElementwiseProblem):
    def __init__(self, match_df):
        self.match_df = match_df.reset_index(drop=True)  # Ensure index is 0 to n-1
        self.n_players = len(match_df)
        
        # Define teams
        self.teams = match_df["team"].unique()
        if len(self.teams) != 2:
            raise ValueError("Expected exactly two teams in the player pool")
        self.team1, self.team2 = self.teams
        
        # Define indices for roles and teams
        self.wk_indices = match_df[match_df["player_type"] == "WK"].index.tolist()
        self.bat_indices = match_df[match_df["player_type"] == "BAT"].index.tolist()
        self.bowl_indices = match_df[match_df["player_type"] == "BOWL"].index.tolist()
        self.all_indices = match_df[match_df["player_type"] == "ALL"].index.tolist()
        self.team1_indices = match_df[match_df["team"] == self.team1].index.tolist()
        self.team2_indices = match_df[match_df["team"] == self.team2].index.tolist()
        
        # Number of variables: one binary variable per player (selected or not) + captain + vice-captain
        n_vars = self.n_players * 3  # x (selected), c (captain), vc (vice-captain)
        
        # Define the problem
        super().__init__(
            n_var=n_vars,
            n_obj=3,  # Three objectives: runs, wickets, dismissals
            n_eq_constr=3,  # Equality constraints: exactly 11 players, 1 captain, 1 vice-captain
            n_ieq_constr=16,  # Inequality constraints: role limits (8), team limits (4), credits (1), captain/vice-captain rules (3)
            xl=0,  # Lower bound for binary variables
            xu=1,  # Upper bound for binary variables
            vtype=bool  # Binary variables
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Split the decision variables: x (selected), c (captain), vc (vice-captain)
        x = X[:self.n_players]  # First n_players variables are for selection (0 or 1)
        c = X[self.n_players:2*self.n_players]  # Next n_players variables are for captain
        vc = X[2*self.n_players:]  # Last n_players variables are for vice-captain
        
        # Convert boolean arrays to integers for arithmetic operations
        x = x.astype(int)
        c = c.astype(int)
        vc = vc.astype(int)
        
        # Objectives: Maximize runs, wickets, dismissals (with captain/vice-captain multipliers)
        runs = 0
        wkts = 0
        dis = 0
        for i in range(self.n_players):
            multiplier = x[i] + c[i] + 0.5 * vc[i]  # 1 for selected, +1 for captain, +0.5 for vice-captain
            runs += self.match_df.loc[i, "runs_per_mat"] * multiplier
            wkts += self.match_df.loc[i, "wkts_per_mat"] * multiplier
            dis += self.match_df.loc[i, "dis_per_mat"] * multiplier
        
        # In pymoo, objectives are minimized by default, so we negate them to maximize
        out["F"] = [-runs, -wkts, -dis]
        
        # Equality constraints
        # 1. Exactly 11 players
        total_players = np.sum(x)
        eq1 = total_players - 11  # Should be 0
        
        # 2. Exactly 1 captain
        total_captains = np.sum(c)
        eq2 = total_captains - 1  # Should be 0
        
        # 3. Exactly 1 vice-captain
        total_vice_captains = np.sum(vc)
        eq3 = total_vice_captains - 1  # Should be 0
        
        out["H"] = [eq1, eq2, eq3]
        
        # Inequality constraints (should be <= 0)
        # Role constraints
        wk_count = np.sum(x[self.wk_indices])
        bat_count = np.sum(x[self.bat_indices])
        bowl_count = np.sum(x[self.bowl_indices])
        all_count = np.sum(x[self.all_indices])
        
        # 1-2 Wicketkeepers
        wk_min = 1 - wk_count  # wk_count >= 1
        wk_max = wk_count - 2  # wk_count <= 2
        
        # 3-5 Batsmen
        bat_min = 3 - bat_count  # bat_count >= 3
        bat_max = bat_count - 5  # bat_count <= 5
        
        # 3-5 Bowlers
        bowl_min = 3 - bowl_count  # bowl_count >= 3
        bowl_max = bowl_count - 5  # bowl_count <= 5
        
        # 1-3 All-rounders
        all_min = 1 - all_count  # all_count >= 1
        all_max = all_count - 3  # all_count <= 3
        
        # Team constraints: 5-6 players per team
        team1_count = np.sum(x[self.team1_indices])
        team2_count = np.sum(x[self.team2_indices])
        
        team1_min = 5 - team1_count  # team1_count >= 5
        team1_max = team1_count - 6  # team1_count <= 6
        team2_min = 5 - team2_count  # team2_count >= 5
        team2_max = team2_count - 6  # team2_count <= 6
        
        # Credits constraint
        total_credits = np.sum(x * self.match_df["credits"].values)
        credits_constraint = total_credits - 100  # total_credits <= 100
        
        # Captain and vice-captain constraints
        # Captain must be a selected player: c[i] <= x[i] for all i
        # Vice-captain must be a selected player: vc[i] <= x[i] for all i
        # A player cannot be both captain and vice-captain: c[i] + vc[i] <= 1 for all i
        captain_selected = 0
        vc_selected = 0
        captain_vc_exclusive = 0
        for i in range(self.n_players):
            # c[i] <= x[i] means c[i] - x[i] <= 0, so violation is max(0, c[i] - x[i])
            captain_selected += max(0, c[i] - x[i])
            # vc[i] <= x[i] means vc[i] - x[i] <= 0, so violation is max(0, vc[i] - x[i])
            vc_selected += max(0, vc[i] - x[i])
            # c[i] + vc[i] <= 1 means c[i] + vc[i] - 1 <= 0, so violation is max(0, c[i] + vc[i] - 1)
            captain_vc_exclusive += max(0, c[i] + vc[i] - 1)
        
        out["G"] = [
            wk_min, wk_max,
            bat_min, bat_max,
            bowl_min, bowl_max,
            all_min, all_max,
            team1_min, team1_max,
            team2_min, team2_max,
            credits_constraint,
            captain_selected,
            vc_selected,
            captain_vc_exclusive
        ]

def optimize_team(match_df, output_file):
    """Optimize a Dream11 team using multi-objective optimization with pymoo."""
    print("Optimizing Dream11 team with multi-objective optimization using pymoo...")
    
    # Feasibility check
    print("Checking feasibility of player pool...")
    total_players = len(match_df)
    wk_players = len(match_df[match_df["player_type"] == "WK"])
    bat_players = len(match_df[match_df["player_type"] == "BAT"])
    bowl_players = len(match_df[match_df["player_type"] == "BOWL"])
    all_players = len(match_df[match_df["player_type"] == "ALL"])
    
    teams = match_df["team"].unique()
    if len(teams) != 2:
        raise ValueError("Expected exactly two teams in the player pool")
    team1, team2 = teams
    team1_players = len(match_df[match_df["team"] == team1])
    team2_players = len(match_df[match_df["team"] == team2])
    
    print(f"Total players: {total_players} (need at least 11)")
    print(f"Wicketkeepers: {wk_players} (need at least 1)")
    print(f"Batsmen: {bat_players} (need at least 3)")
    print(f"Bowlers: {bowl_players} (need at least 3)")
    print(f"All-rounders: {all_players} (need at least 1)")
    print(f"{team1} players: {team1_players} (need at least 5)")
    print(f"{team2} players: {team2_players} (need at least 5)")
    
    if total_players < 11:
        raise ValueError("Not enough players (need at least 11)")
    if wk_players < 1:
        raise ValueError("Not enough wicketkeepers (need at least 1)")
    if bat_players < 3:
        raise ValueError("Not enough batsmen (need at least 3)")
    if bowl_players < 3:
        raise ValueError("Not enough bowlers (need at least 3)")
    if all_players < 1:
        raise ValueError("Not enough all-rounders (need at least 1)")
    if team1_players < 5 or team2_players < 5:
        raise ValueError("Not enough players from each team (need at least 5 per team)")

    # Define the problem
    problem = Dream11Problem(match_df)
    
    # Define the algorithm (NSGA-II)
    algorithm = NSGA2(
        pop_size=100,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=BitflipMutation(prob=0.01),
        eliminate_duplicates=True
    )
    
    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 200),  # Run for 200 generations
        seed=1,
        verbose=True
    )
    
    # Extract the Pareto front solutions
    pareto_solutions = []
    for i in range(len(res.X)):
        X = res.X[i]
        x = X[:problem.n_players]  # Selected players
        c = X[problem.n_players:2*problem.n_players]  # Captain
        vc = X[2*problem.n_players:]  # Vice-captain
        
        # Convert to integers
        x = x.astype(int)
        c = c.astype(int)
        vc = vc.astype(int)
        
        # Extract the team
        selected_players = []
        total_runs = 0
        total_wkts = 0
        total_dis = 0
        for j in range(problem.n_players):
            if x[j] == 1:
                role = "Player"
                multiplier = 1
                if c[j] == 1:
                    role = "Captain"
                    multiplier = 2
                elif vc[j] == 1:
                    role = "Vice-Captain"
                    multiplier = 1.5
                total_runs += match_df.iloc[j]["runs_per_mat"] * multiplier
                total_wkts += match_df.iloc[j]["wkts_per_mat"] * multiplier
                total_dis += match_df.iloc[j]["dis_per_mat"] * multiplier
                selected_players.append({
                    "player_name": match_df.iloc[j]["player_name"],
                    "team": match_df.iloc[j]["team"],
                    "player_type": match_df.iloc[j]["player_type"],
                    "runs_per_mat": match_df.iloc[j]["runs_per_mat"],
                    "wkts_per_mat": match_df.iloc[j]["wkts_per_mat"],
                    "dis_per_mat": match_df.iloc[j]["dis_per_mat"],
                    "credits": match_df.iloc[j]["credits"],
                    "role": role
                })
        
        solution_df = pd.DataFrame(selected_players)
        pareto_solutions.append({
            "team": solution_df,
            "total_runs": total_runs,
            "total_wkts": total_wkts,
            "total_dis": total_dis
        })
    
    # Print all Pareto-optimal solutions
    print("\nPareto-Optimal Solutions:")
    for idx, sol in enumerate(pareto_solutions):
        print(f"\nSolution {idx + 1}:")
        print(f"Total runs_per_mat: {sol['total_runs']:.2f}")
        print(f"Total wkts_per_mat: {sol['total_wkts']:.2f}")
        print(f"Total dis_per_mat: {sol['total_dis']:.2f}")
        print(sol["team"])
    
    # Select a balanced solution (e.g., the one with the highest sum of normalized objectives)
    max_score = -float('inf')
    selected_solution = None
    for sol in pareto_solutions:
        # Normalize objectives (assuming max values for scaling)
        norm_runs = sol["total_runs"] / 400  # Rough max runs
        norm_wkts = sol["total_wkts"] / 20   # Rough max wkts
        norm_dis = sol["total_dis"] / 10     # Rough max dis
        score = norm_runs + norm_wkts + norm_dis
        if score > max_score:
            max_score = score
            selected_solution = sol
    
    # Save the selected solution
    team_df = selected_solution["team"]
    team_df.to_csv(output_file, index=False)
    print(f"\nSelected Dream11 Team for {team1} vs {team2} (Balanced Solution):")
    print(team_df)
    
    return pareto_solutions