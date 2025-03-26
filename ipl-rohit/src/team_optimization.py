from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import pandas as pd

def optimize_dream11_team(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Feasibility check
    print("Checking feasibility of player pool...")
    print(f"Total players: {len(df)}")
    print("Wicketkeepers:", len(df[df["player_type"] == "WK"]))
    print("Batsmen:", len(df[df["player_type"] == "BAT"]))
    print("Bowlers:", len(df[df["player_type"] == "BOWL"]))
    print("All-rounders:", len(df[df["player_type"] == "ALL"]))
    print("DC players:", len(df[df["team"] == "DC"]))
    print("LSG players:", len(df[df["team"] == "LSG"]))

    # Check minimum requirements
    if len(df) < 11:
        raise ValueError("Not enough players to form a team (need at least 11)")
    if len(df[df["player_type"] == "WK"]) < 1:
        raise ValueError("Not enough wicketkeepers (need at least 1)")
    if len(df[df["player_type"] == "BAT"]) < 3:
        raise ValueError("Not enough batsmen (need at least 3)")
    if len(df[df["player_type"] == "BOWL"]) < 3:
        raise ValueError("Not enough bowlers (need at least 3)")
    if len(df[df["player_type"] == "ALL"]) < 1:  # Relaxed from 2 to 1
        raise ValueError("Not enough all-rounders (need at least 1)")
    if len(df[df["team"] == "DC"]) < 4:
        raise ValueError("Not enough DC players (need at least 4)")
    if len(df[df["team"] == "LSG"]) < 4:
        raise ValueError("Not enough LSG players (need at least 4)")

    prob = LpProblem("Dream11_Team", LpMaximize)
    players = df["player_name"].tolist()
    x = {p: LpVariable(f"x_{p}", cat="Binary") for p in players}

    batting_strength = lpSum(df.loc[i, "runs_per_mat"] * x[p] if df.loc[i, "player_type"] in ["BAT", "ALL"] else 0
                             for i, p in enumerate(players))
    bowling_strength = lpSum(df.loc[i, "wkts_per_mat"] * x[p] if df.loc[i, "player_type"] in ["BOWL", "ALL"] else 0
                             for i, p in enumerate(players))
    fielding_strength = lpSum(df.loc[i, "dis_per_mat"] * x[p] for i, p in enumerate(players))

    prob += 0.45 * batting_strength + 0.45 * bowling_strength + 0.1 * fielding_strength

    prob += lpSum(x[p] for p in players) == 11
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "WK") >= 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "WK") <= 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BAT") >= 3
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BAT") <= 5
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BOWL") >= 3
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "BOWL") <= 5
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "ALL") >= 1  # Relaxed from 2 to 1
    prob += lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "player_type"] == "ALL") <= 3
    prob += lpSum(df.loc[i, "credits"] * x[p] for i, p in enumerate(players)) <= 100

    team1_players = lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "team"] == "DC")
    team2_players = lpSum(x[p] for i, p in enumerate(players) if df.loc[i, "team"] == "LSG")
    prob += team1_players >= 4
    prob += team1_players <= 7
    prob += team2_players >= 4
    prob += team2_players <= 7

    status = prob.solve()
    if status != 1:
        raise ValueError("Optimization failed")

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

if __name__ == "__main__":
    optimize_dream11_team("../data/match_players.csv", "../data/dream11_team.csv")