from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import pandas as pd

def optimize_dream11_team(input_file, output_file, team1, team2):
    df = pd.read_csv(input_file)
    
    print("Checking feasibility of player pool...")
    print(f"Total players: {len(df)} (need at least 11)")
    print(f"Wicketkeepers: {len(df[df['player_type'] == 'WK'])} (need at least 1)")
    print(f"Batsmen: {len(df[df['player_type'] == 'BAT'])} (need at least 3)")
    print(f"Bowlers: {len(df[df['player_type'] == 'BOWL'])} (need at least 3)")
    print(f"All-rounders: {len(df[df['player_type'] == 'ALL'])} (need at least 1)")
    print(f"{team1} players: {len(df[df['team'] == team1])} (need at least 5)")
    print(f"{team2} players: {len(df[df['team'] == team2])} (need at least 5)")

    if len(df) < 11:
        raise ValueError("Not enough players to form a team (need at least 11). Please update squad.csv to mark more players as PLAYING.")
    if len(df[df["player_type"] == "WK"]) < 1:
        raise ValueError("Not enough wicketkeepers (need at least 1). Please update squad.csv to include more WK players marked as PLAYING.")
    if len(df[df["player_type"] == "BAT"]) < 3:
        raise ValueError("Not enough batsmen (need at least 3). Please update squad.csv to include more BAT players marked as PLAYING.")
    if len(df[df["player_type"] == "BOWL"]) < 3:
        raise ValueError("Not enough bowlers (need at least 3). Please update squad.csv to include more BOWL players marked as PLAYING.")
    if len(df[df["player_type"] == "ALL"]) < 1:
        raise ValueError("Not enough all-rounders (need at least 1). Please update squad.csv to include more ALL players marked as PLAYING.")
    if len(df[df["team"] == team1]) < 5:
        raise ValueError(f"Not enough {team1} players (need at least 5). Please update squad.csv to include more {team1} players marked as PLAYING.")
    if len(df[df["team"] == team2]) < 5:
        raise ValueError(f"Not enough {team2} players (need at least 5). Please update squad.csv to include more {team2} players marked as PLAYING.")

    prob = LpProblem("Dream11_Team", LpMaximize)
    players = df["player_name"].tolist()
    x = {p: LpVariable(f"x_{p}", cat="Binary") for p in players}

    batting_strength = lpSum(df.loc[i, "runs_per_mat"] * x[p] if df.loc[i, "player_type"] in ["BAT", "ALL"] else 0
                             for i, p in enumerate(players))
    bowling_strength = lpSum(df.loc[i, "wkts_per_mat"] * x[p] if df.loc[i, "player_type"] in ["BOWL", "ALL"] else 0
                             for i, p in enumerate(players))
    fielding_strength = lpSum(df.loc[i, "dis_per_mat"] * x[p] for i, p in enumerate(players))

    # Define the objective function without the imbalance term
    prob += 0.45 * batting_strength + 0.45 * bowling_strength + 0.1 * fielding_strength

    # Basic constraints
    prob += lpSum(x[p] for p in players) == 11
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "WK") >= 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "WK") <= 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BAT") >= 3
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BAT") <= 5
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BOWL") >= 3
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BOWL") <= 5
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "ALL") >= 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "ALL") <= 3
    prob += lpSum(df.loc[i, "credits"] * x[p] for i, p in enumerate(players)) <= 100

    # Team balance constraints
    team1_players = lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "team"] == team1)
    team2_players = lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "team"] == team2)
    team1_available = len(df[df["team"] == team1])
    team2_available = len(df[df["team"] == team2])
    prob += team1_players >= min(5, team1_available)  # At least 5 players from team1
    prob += team1_players <= min(6, team1_available)  # At most 6 players from team1
    prob += team2_players >= min(5, team2_available)  # At least 5 players from team2
    prob += team2_players <= min(6, team2_available)  # At most 6 players from team2

    status = prob.solve()
    if status != 1:
        raise ValueError("Optimization failed. Please check if the constraints can be satisfied with the available players.")

    selected_players = [p for p in players if x[p].value() == 1]
    team_df = df[df["player_name"].isin(selected_players)][["player_name", "team", "player_type", "runs_per_mat", "wkts_per_mat", "dis_per_mat", "credits"]]
    
    team_df["expected_points"] = (0.45 * team_df["runs_per_mat"] + 
                                 0.45 * team_df["wkts_per_mat"] + 
                                 0.1 * team_df["dis_per_mat"])
    team_df = team_df.sort_values(by="expected_points", ascending=False)
    team_df["role"] = ""
    team_df.iloc[0, team_df.columns.get_loc("role")] = "Captain"
    team_df.iloc[1, team_df.columns.get_loc("role")] = "Vice-Captain"
    
    team_df.to_csv(output_file, index=False)
    return team_df