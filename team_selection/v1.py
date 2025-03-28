import pandas as pd
import pulp

# ============================================================
# Adjustable Parameters (Defaults, will be adjusted by ground)
# ============================================================
batter_weight = 1.0
bowler_weight = 1.0
allrounder_weight = 1.0  # Boost for all-rounders
keeper_weight = 1.0  # Weight for keepers

# ============================================================
# Ground Data (CSV String)
# ============================================================
ground_df = pd.read_csv("data/ground.csv")

# ============================================================
# 1. Load Player Data from Provided CSV
# ============================================================
squad_df = pd.read_csv("data/SquadPlayerNames_7.csv")

# Filter for playing players (PLAYING or X_FACTOR_SUBSTITUTE)
playing_players = squad_df[squad_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]

# Load form scores and merge with playing players
form_df = pd.read_csv('data/player_form_scores(3).csv')
df = pd.merge(playing_players, form_df, left_on="Player Name", right_on="Player", how="left")

# Select and rename relevant columns
df = df[["Player Name", "Team_x", "Player Type_x", "Batting Form", "Bowling Form"]]
df.columns = ["Player", "Team", "Player Type", "Batting Form", "Bowling Form"]

# Handle potential missing form scores by filling with 0 (or adjust as needed)
df["Batting Form"] = df["Batting Form"].fillna(0)
df["Bowling Form"] = df["Bowling Form"].fillna(0)

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
        return allrounder_weight * max(batting, bowling)  # Use max instead of average
    elif role == "WK":
        return keeper_weight * batting
    else:
        print(f"Warning: Unknown role '{role}' for player {row['Player']}")
        return 0  # Fallback if role is invalid

# ============================================================
# 3. Optimization (No Credit Constraint, Adjustable Team Weights)
# ============================================================
def optimize_team(team1, team2, total_players=11, team1_weight=1.0, team2_weight=1.0):
    team_df = df[df["Team"].isin([team1, team2])].copy()
    
    # Apply team-specific weights to scores
    team_df["Adjusted_Score"] = team_df.apply(
        lambda row: row["Score"] * (team1_weight if row["Team"] == team1 else team2_weight), axis=1
    )
    
    # Check if enough players exist for constraints
    batters = team_df[team_df["Player Type"].str.strip().str.upper() == "BAT"]
    bowlers = team_df[team_df["Player Type"].str.strip().str.upper() == "BOWL"]
    allrounders = team_df[team_df["Player Type"].str.strip().str.upper() == "ALL"]
    keepers = team_df[team_df["Player Type"].str.strip().str.upper() == "WK"]
    
    # Players capable of bowling (bowlers + all-rounders)
    bowling_options = pd.concat([bowlers, allrounders])
    
    if len(batters) < 2 or len(bowlers) < 3 or len(bowling_options) < 5 or len(keepers) < 1:
        print("Not enough players for constraints. Relaxing requirements.")
        print(f"Available batters: {len(batters)}, bowlers: {len(bowlers)}, bowling options: {len(bowling_options)}, keepers: {len(keepers)}")
        return None
    
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat="Binary")
    
    # Objective: Maximize total adjusted score
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Adjusted_Score"] for i in players])
    
    # Constraints
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Role constraints
    prob += pulp.lpSum([x[i] for i in batters.index]) >= 2, "Min_Batters"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= 2, "Min_Bowlers"
    prob += pulp.lpSum([x[i] for i in bowling_options.index]) >= 5, "Min_Bowling_Options"
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
    selected_df.sort_values("Adjusted_Score", ascending=False, inplace=True)
    
    # Assign roles
    selected_df["Role_In_Team"] = "Player"
    if len(selected_df) > 0:
        selected_df.iloc[0, selected_df.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_df) > 1:
        selected_df.iloc[1, selected_df.columns.get_loc("Role_In_Team")] = "Vice Captain"
    
    return selected_df[["Player", "Team", "Player Type", "Adjusted_Score", "Role_In_Team"]]

# ============================================================
# 4. Test Run
# ============================================================
if __name__ == "__main__":
    # Display available grounds with numbers
    print("Available Grounds:")
    for i, row in ground_df.iterrows():
        print(f"{i + 1}. {row['Ground']} ({row['City']})")
    
    # Prompt user for ground number input
    try:
        ground_number = int(input("\nEnter the ground number (1-13) for the match: "))
        if ground_number < 1 or ground_number > 13:
            raise ValueError
    except ValueError:
        print("Invalid input! Please enter a number between 1 and 13.")
        exit(1)
    
    # Map ground number to index (1-based to 0-based)
    ground_index = ground_number - 1
    selected_ground = ground_df.iloc[ground_index]["Ground"]
    
    # Get ground-specific weights
    ground_data = ground_df.iloc[ground_index]
    batter_weight = float(ground_data["Batting"])
    keeper_weight = float(ground_data["Batting"])
    bowler_weight = float(ground_data["Bowling"])
    
    print(f"\nSelected Ground: {selected_ground}")
    print(f"Batter Weight: {batter_weight}")
    print(f"Bowler Weight: {bowler_weight}")
    
    # Recalculate scores with ground-adjusted weights
    df["Score"] = df.apply(calculate_score, axis=1)
    
    # Define teams and team weights (updated to match CSV)
    home_team1 = "SRH"
    away_team2 = "LSG"
    team1_weight = 1.05  # Slight home advantage
    team2_weight = 1.0  
    
    # Optimize team
    best_team = optimize_team(
        team1=home_team1,
        team2=away_team2,
        team1_weight=team1_weight,
        team2_weight=team2_weight
    )
    
    if best_team is not None:
        print("\nOptimal Fantasy Team:")
        print(best_team)
    else:
        print("No valid team could be formed. Check player roles and constraints.")