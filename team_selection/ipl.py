import pandas as pd
import pulp

# ============================================================
# Adjustable Parameters
# ============================================================
batter_weight = 1.0
bowler_weight = 1.0
allrounder_weight = 1.0  # Boost for all-rounders
keeper_weight = 1.0       # Weight for keepers

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv('data/player_form_last3.csv')

# ============================================================
# 2. Corrected Score Calculation (Case-Insensitive Role Check)
# ============================================================
def calculate_score(row):
    role = row["Player Type"].strip().upper()  # Ensure uppercase matching
    batting = row["Batting Form"]
    bowling = row["Bowling Form"]
    
    if role == "BAT":
        return batter_weight * batting
    elif role == "BOWL":
        return bowler_weight * bowling
    elif role == "ALL":
        return allrounder_weight * ((batting + bowling) / 2)
    elif role == "WK":
        return keeper_weight * batting
    else:
        print(f"Warning: Unknown role '{role}' for player {row['Player']}")
        return 0  # Fallback if role is invalid

df["Score"] = df.apply(calculate_score, axis=1)

# ============================================================
# 3. Optimization (No Credit Constraint)
# ============================================================
def optimize_team(team1, team2, total_players=20):
    team_df = df[df["Team"].isin([team1, team2])].copy()
    
    # Check if enough players exist for constraints
    batters = team_df[team_df["Player Type"].str.strip().str.upper() == "BAT"]
    bowlers = team_df[team_df["Player Type"].str.strip().str.upper() == "BOWL"]
    allrounders = team_df[team_df["Player Type"].str.strip().str.upper() == "ALL"]
    keepers = team_df[team_df["Player Type"].str.strip().str.upper() == "WK"]
    
    if len(batters) < 4 or len(bowlers) < 3 or len(keepers) < 1:
        print("Not enough players for constraints. Relaxing requirements.")
        return None
    
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat="Binary")
    
    # Objective: Maximize total score
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players])
    
    # Constraints
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Role constraints
    prob += pulp.lpSum([x[i] for i in batters.index]) >= 4, "Min_Batters"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= 3, "Min_Bowlers"
    prob += pulp.lpSum([x[i] for i in keepers.index]) >= 1, "Min_Keepers"
    
    # Team constraints
    prob += pulp.lpSum([x[i] for i in team_df[team_df["Team"] == team1].index]) >= 1, f"Min_from_{team1}"
    prob += pulp.lpSum([x[i] for i in team_df[team_df["Team"] == team2].index]) >= 1, f"Min_from_{team2}"
    
    # Solve
    prob.solve()
    
    if pulp.LpStatus[prob.status] != "Optimal":
        print("No solution found! Try relaxing constraints.")
        return None
    
    # Extract selected players
    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_df = team_df.loc[selected].copy()
    selected_df.sort_values("Score", ascending=False, inplace=True)
    
    # Assign roles
    selected_df["Role_In_Team"] = "Player"
    if len(selected_df) > 0:
        selected_df.iloc[0, selected_df.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_df) > 1:
        selected_df.iloc[1, selected_df.columns.get_loc("Role_In_Team")] = "Vice Captain"
    
    return selected_df[["Player", "Team", "Player Type", "Score", "Role_In_Team"]]

# ============================================================
# 4. Test Run
# ============================================================
if __name__ == "__main__":
    team1 = "KKR"
    team2 = "SRH"

    best_team = optimize_team(team1, team2)
    
    if best_team is not None:
        print("Optimal Fantasy Team:")
        print(best_team)
    else:
        print("No valid team could be formed. Check player roles and constraints.")