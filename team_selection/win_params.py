import pandas as pd
import pulp

# Load dataframes
ground_df = pd.read_csv("data/ground.csv")
squad_df = pd.read_csv("data/SquadPlayerNames.csv")
form_df = pd.read_csv("data/merged_output.csv")

# Cleaning dataframes (removing unimportant values)
form_df.drop(["Fielding Form", "Credits", "Player Type", "Team"], axis=1, inplace=True)
squad_df.drop("Credits", axis=1, inplace=True)

# Taking players which are in playing 11 or impact player
playing_df = squad_df[squad_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]

# Merging dataframes
selection_df = pd.merge(playing_df, form_df, left_on="Player Name", right_on="Player", how="left")

# Selecting and renaming relevant columns
selection_df = selection_df[["Player Name", "Team", "Player Type", "Batting Form", "Bowling Form", "lineupOrder"]]
selection_df.columns = ["Player", "Team", "Player Type", "Batting Form", "Bowling Form", "lineupOrder"]

# Handling missing form scores
selection_df["Batting Form"] = selection_df["Batting Form"].fillna(0)
selection_df["Bowling Form"] = selection_df["Bowling Form"].fillna(0)

# Adjustable weights (will be set by ground)
batter_weight = 1.0
bowler_weight = 1.0
allrounder_weight = 1.0
keeper_weight = 1.0

# Calculating scores according to role
def calculate_score(row):
    role = row["Player Type"]
    batting = row['Batting Form']
    bowling = row['Bowling Form']

    if role == "BAT":
        return batter_weight * batting
    if role == "WK":
        return keeper_weight * batting
    if role == "BOWL":
        return bowler_weight * bowling
    if role == "ALL":
        return allrounder_weight * max(batting, bowling)

# Optimization function without team constraints
def optimize(total_players=11):
    team_df = selection_df.copy()  # Use the entire dataset without team filtering
    
    # Check if enough players exist for constraints
    batters = team_df[team_df["Player Type"].str.strip().str.upper() == "BAT"]
    bowlers = team_df[team_df["Player Type"].str.strip().str.upper() == "BOWL"]
    allrounders = team_df[team_df["Player Type"].str.strip().str.upper() == "ALL"]
    keepers = team_df[team_df["Player Type"].str.strip().str.upper() == "WK"]

    # Players capable of bowling (bowlers + all-rounders)
    bowling_options = pd.concat([bowlers, allrounders])

    if len(batters) < 1 or len(bowlers) < 1 or len(bowling_options) < 5 or len(keepers) < 1 or len(allrounders) < 1:
        print("Not enough players for constraints.")
        print(f"Available batters: {len(batters)}, bowlers: {len(bowlers)}, bowling options: {len(bowling_options)}, keepers: {len(keepers)}, allrounders: {len(allrounders)}")
        return None

    prob = pulp.LpProblem("Fantasy Team", pulp.LpMaximize)
    players = team_df.index.tolist()
    
    x = pulp.LpVariable.dicts("player", players, cat="Binary")

    # Objective: Maximize total score (no team-specific weights)
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players])
    
    # Constraints
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Role constraints
    prob += pulp.lpSum([x[i] for i in batters.index]) >= 1, "Min_Batters"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= 1, "Min_Bowlers"
    prob += pulp.lpSum([x[i] for i in bowling_options.index]) >= 5, "Min_Bowling_Options"
    prob += pulp.lpSum([x[i] for i in keepers.index]) >= 1, "Min_Keepers"
    prob += pulp.lpSum([x[i] for i in allrounders.index]) >= 1, "Min_Allrounders"  # New constraint
    
    # Solve
    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        print("No solution found! Try relaxing constraints.")
        return None

    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_11 = team_df.loc[selected].copy()
    selected_11.sort_values("Score", ascending=False, inplace=True)
    
    # Assign roles
    selected_11["Role_In_Team"] = "Player"
    if len(selected_11) > 0:
        selected_11.iloc[0, selected_11.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_11) > 1:
        selected_11.iloc[1, selected_11.columns.get_loc("Role_In_Team")] = "Vice Captain"
    
    return selected_11[["Player", "Team", "Player Type", "Score", "Role_In_Team"]]

if __name__ == "__main__":
    print("Grounds : ")
    for i, r in ground_df.iterrows():
        print(f"{i + 1}. {r['Ground']} ({r['City']})")

    try:
        ground_number = int(input("\nEnter the ground number (1-13) for the match: "))
        if ground_number < 1 or ground_number > 13:
            raise ValueError
    except ValueError:
        print("Invalid input! Please enter a number between 1 and 13.")
        exit(1)

    ground_index = ground_number - 1
    selected_ground = ground_df.iloc[ground_index]["Ground"]

    # Set ground-specific weights
    ground_data = ground_df.iloc[ground_index]
    batter_weight = float(ground_data['Batting'])
    keeper_weight = float(ground_data['Batting'])
    bowler_weight = float(ground_data['Bowling'])
    allrounder_weight = ((batter_weight + bowler_weight) / 2)
    if allrounder_weight < 1:
        allrounder_weight = 1.0

    # Calculate scores
    selection_df["Score"] = selection_df.apply(calculate_score, axis=1)
    
    # Optimize team (no team parameters)
    best_team = optimize() 
    
    if best_team is not None:
        print("\nOptimal Fantasy Team:")
        print(best_team)
    else:
        print("No valid team could be formed. Check player roles and constraints.")