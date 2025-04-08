import pandas as pd
import pulp

# Load data files
ground_df = pd.read_csv("data/ground.csv")
squad_df = pd.read_csv("data/SquadPlayerNames.csv")
form_df = pd.read_csv("data/merged_output_updated.csv")

# Clean dataframes
squad_df = squad_df[["Player Name", "Team", "Player Type", "Bowler Type", "IsPlaying", "lineupOrder"]]
form_df = form_df[["Player", "Batting Form", "Bowling Form"]]

# Take players which are in playing 11 or impact player
playing_df = squad_df[squad_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]

# Merge playing squad with form data
selection_df = pd.merge(playing_df, form_df, left_on="Player Name", right_on="Player", how="left")
selection_df = selection_df.drop(columns=["Player"])

# Fill missing values
selection_df["Batting Form"] = selection_df["Batting Form"].fillna(0)
selection_df["Bowling Form"] = selection_df["Bowling Form"].fillna(0)
selection_df["Bowler Type"] = selection_df["Bowler Type"].fillna('-')

def calculate_score(row, batter_weight, bowler_weight, ground_type):
    """Calculate player score based on their role, form, and ground type"""
    role = row["Player Type"].strip().upper()
    batting = row['Batting Form']
    bowling = row['Bowling Form']
    bowler_type = row['Bowler Type'].strip().upper()

    if role == "BAT":
        return batter_weight * batting
    if role == "WK":
        return batter_weight * batting
    if role == "BOWL":
        # Apply ground-specific multiplier for bowlers
        if ground_type == "SPIN" and bowler_type == "SPIN":
            return bowler_weight * bowling * 1.2  # 20% boost for spinners on spin-friendly grounds
        elif ground_type == "PACE" and bowler_type == "PACE":
            return bowler_weight * bowling * 1.2  # 20% boost for pacers on pace-friendly grounds
        return bowler_weight * bowling
    if role == "ALL":
        return 1 * max(batting, bowling)  # Fixed 1.15 weight for all-rounders
    return 0

def optimize(team1, team2, batter_weight, bowler_weight, ground_type):
    """Optimize fantasy team selection with ground-specific constraints"""
    total_players = 11
    home_weight = 1.05
    away_weight = 1.0

    team_df = selection_df[selection_df["Team"].isin([team1, team2])].copy()
    
    # Calculate scores with ground-specific adjustments
    team_df["Score"] = team_df.apply(
        lambda row: calculate_score(row, batter_weight, bowler_weight, ground_type), 
        axis=1
    )
    
    # Apply home/away weight adjustment
    team_df["Adjusted_Score"] = team_df.apply(
        lambda row: row["Score"] * (home_weight if row["Team"] == team1 else away_weight), 
        axis=1
    )
    
    # Filter players by role
    batters = team_df[team_df["Player Type"].str.strip().str.upper() == "BAT"]
    bowlers = team_df[team_df["Player Type"].str.strip().str.upper() == "BOWL"]
    allrounders = team_df[team_df["Player Type"].str.strip().str.upper() == "ALL"]
    keepers = team_df[team_df["Player Type"].str.strip().str.upper() == "WK"]

    # Categorize bowlers by type
    pace_bowlers = team_df[(team_df["Player Type"].str.strip().str.upper() == "BOWL") & 
                          (team_df["Bowler Type"].str.strip().str.upper() == "PACE")]
    spin_bowlers = team_df[(team_df["Player Type"].str.strip().str.upper() == "BOWL") & 
                          (team_df["Bowler Type"].str.strip().str.upper() == "SPIN")]

    # Players capable of bowling (bowlers + all-rounders)
    bowling_options = pd.concat([bowlers, allrounders])

    # Create optimization problem
    prob = pulp.LpProblem("Fantasy Team Optimization", pulp.LpMaximize)
    players = team_df.index.tolist()
    
    # Decision variables
    x = pulp.LpVariable.dicts("player", players, cat="Binary")

    # Objective: Maximize total adjusted score
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Adjusted_Score"] for i in players])
    
    # Basic constraints
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    prob += pulp.lpSum([x[i] for i in batters.index]) >= 1, "Min_Batters"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= 1, "Min_Bowlers"
    prob += pulp.lpSum([x[i] for i in bowling_options.index]) >= 5, "Min_Bowling_Options"
    prob += pulp.lpSum([x[i] for i in keepers.index]) >= 1, "Min_Keepers"
    
    # Team constraints
    prob += pulp.lpSum([x[i] for i in team_df[team_df["Team"] == team1].index]) >= 1, f"Min_from_{team1}"
    prob += pulp.lpSum([x[i] for i in team_df[team_df["Team"] == team2].index]) >= 1, f"Min_from_{team2}"
    
    # Ground-specific bowler constraints
    if ground_type == "SPIN":
        prob += pulp.lpSum([x[i] for i in spin_bowlers.index]) >= 2, "Min_Spinners"
    elif ground_type == "PACE":
        prob += pulp.lpSum([x[i] for i in pace_bowlers.index]) >= 3, "Min_Pacers"
    
    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        print("No solution found! Try relaxing constraints.")
        return None

    # Get selected players
    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_11 = team_df.loc[selected].copy()
    selected_11.sort_values("Adjusted_Score", ascending=False, inplace=True)
    
    # Assign roles
    selected_11["Role_In_Team"] = "Player"
    if len(selected_11) > 0:
        selected_11.iloc[0, selected_11.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_11) > 1:
        selected_11.iloc[1, selected_11.columns.get_loc("Role_In_Team")] = "Vice Captain"
    
    return selected_11[["Player Name", "Team", "Player Type", "Bowler Type", "Adjusted_Score", "Role_In_Team"]]

if __name__ == "__main__":
    print("Grounds:")
    for i, r in ground_df.iterrows():
        print(f"{i + 1}. {r['Ground']} ({r['City']}) - Pitch: {r.get('Pitch Type', 'Unknown')}")

    try:
        ground_number = int(input("\nEnter the ground number (1-13) for the match: "))
        if ground_number < 1 or ground_number > 13:
            raise ValueError
    except ValueError:
        print("Invalid input! Please enter a number between 1 and 13.")
        exit(1)

    ground_index = ground_number - 1
    selected_ground = ground_df.iloc[ground_index]["Ground"]
    ground_data = ground_df.iloc[ground_index]
    
    # Determine ground type (add this column to your ground.csv if not present)
    ground_type = "SPIN" if "spin" in str(ground_data.get("Pitch Type", "")).lower() else "PACE"
    
    batter_weight = float(ground_data['Batting'])
    bowler_weight = float(ground_data['Bowling'])

    print(f"\nSelected Ground: {selected_ground} ({ground_type}-friendly)")
    print(f"Batting Weight: {batter_weight:.2f}")
    print(f"Bowling Weight: {bowler_weight:.2f}")
    print(f"All-rounders will receive fixed 1.15x multiplier")

    # Get teams for the match (modify to take user input if needed)
    home_team = "KKR"
    away_team = "SRH"
    print(f"Teams: {home_team} vs {away_team}")

    # Get optimized team
    best_team = optimize(
        team1=home_team,
        team2=away_team,
        batter_weight=batter_weight,
        bowler_weight=bowler_weight,
        ground_type=ground_type
    )
    
    if best_team is not None:
        print("\nOptimal Fantasy Team:")
        print(best_team)
        
        # Print bowler type distribution
        bowlers = best_team[best_team["Player Type"].str.upper() == "BOWL"]
        if not bowlers.empty:
            print("\nBowler Types:")
            print(bowlers[["Player Name", "Bowler Type"]])
    else:
        print("No valid team could be formed. Check player roles and constraints.")