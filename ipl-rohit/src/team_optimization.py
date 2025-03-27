import pandas as pd
import pulp

def optimize_team(match_df, output_file):
    """Optimize a Dream11 team using linear programming."""
    print("Optimizing Dream11 team...")
    
    # Feasibility check
    print("Checking feasibility of player pool...")
    total_players = len(match_df)
    wk_players = len(match_df[match_df["player_type"] == "WK"])
    bat_players = len(match_df[match_df["player_type"] == "BAT"])
    bowl_players = len(match_df[match_df["player_type"] == "BOWL"])
    all_players = len(match_df[match_df["player_type"] == "ALL"])
    
    # Get team names (assuming there are exactly two teams)
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
    
    # Validate constraints
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
    prob = pulp.LpProblem("Dream11_Team_Selection", pulp.LpMaximize)
    
    # Variables
    players = match_df.index
    x = pulp.LpVariable.dicts("x", players, cat="Binary")  # 1 if player is selected, 0 otherwise
    c = pulp.LpVariable.dicts("c", players, cat="Binary")  # 1 if player is captain, 0 otherwise
    vc = pulp.LpVariable.dicts("vc", players, cat="Binary")  # 1 if player is vice-captain, 0 otherwise
    
    # Objective function: Maximize a weighted combination of runs, wickets, and dismissals
    # Weights are chosen to balance contributions (e.g., a wicket is worth more than a run)
    prob += pulp.lpSum(
        (match_df.loc[i, "runs_per_mat"] +
         match_df.loc[i, "wkts_per_mat"] * 25 +  # Wickets are more valuable
         match_df.loc[i, "dis_per_mat"] * 10) *  # Dismissals are valuable for wicketkeepers
        (x[i] + c[i] + 0.5 * vc[i]) for i in players
    )
    
    # Constraints
    # 1. Exactly 11 players
    prob += pulp.lpSum(x[i] for i in players) == 11
    
    # 2. Role constraints
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "WK") >= 1
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "WK") <= 2
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "BAT") >= 3
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "BAT") <= 5
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "BOWL") >= 3
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "BOWL") <= 5
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "ALL") >= 1
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "player_type"] == "ALL") <= 3
    
    # 3. Team constraints
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "team"] == team1) >= 5
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "team"] == team1) <= 6
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "team"] == team2) >= 5
    prob += pulp.lpSum(x[i] for i in players if match_df.loc[i, "team"] == team2) <= 6
    
    # 4. Credits constraint
    prob += pulp.lpSum(x[i] * match_df.loc[i, "credits"] for i in players) <= 100
    
    # 5. Captain and vice-captain constraints
    prob += pulp.lpSum(c[i] for i in players) == 1  # Exactly 1 captain
    prob += pulp.lpSum(vc[i] for i in players) == 1  # Exactly 1 vice-captain
    for i in players:
        prob += c[i] <= x[i]  # Captain must be a selected player
        prob += vc[i] <= x[i]  # Vice-captain must be a selected player
        prob += c[i] + vc[i] <= 1  # A player cannot be both captain and vice-captain
    
    # Solve the problem
    prob.solve()
    
    # Extract the solution
    selected_players = []
    for i in players:
        if x[i].value() == 1:
            role = "Player"
            if c[i].value() == 1:
                role = "Captain"
            elif vc[i].value() == 1:
                role = "Vice-Captain"
            selected_players.append({
                "player_name": match_df.loc[i, "player_name"],
                "team": match_df.loc[i, "team"],
                "player_type": match_df.loc[i, "player_type"],
                "runs_per_mat": match_df.loc[i, "runs_per_mat"],
                "wkts_per_mat": match_df.loc[i, "wkts_per_mat"],
                "dis_per_mat": match_df.loc[i, "dis_per_mat"],
                "credits": match_df.loc[i, "credits"],
                "role": role
            })
    
    # Convert to DataFrame and save
    team_df = pd.DataFrame(selected_players)
    team_df.to_csv(output_file, index=False)
    print("\nYour Dream11 Team for {} vs {}:".format(team1, team2))
    print(team_df)
    return team_df