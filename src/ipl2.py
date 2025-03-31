import json
import os
from collections import defaultdict
import pandas as pd

def get_extras(delivery):
    """Calculate total extras from a delivery."""
    extras = delivery.get("extras", {})
    return sum(extras.values()) if isinstance(extras, dict) else extras

def process_match(json_file, player_roles=None):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    match_id = str(data["info"]["dates"][0]) if "dates" in data["info"] and data["info"]["dates"] else os.path.splitext(os.path.basename(json_file))[0]
    venue = data["info"].get("venue", "Unknown")
    teams = data["info"]["teams"]
    season = data["info"].get("season", "Unknown")
    
    dataset = []
    points = defaultdict(float)
    player_stats = defaultdict(lambda: {
        "runs": 0, "balls_faced": 0, "fours": 0, "sixes": 0,
        "wickets": 0, "runs_conceded": 0, "balls_bowled": 0, "dot_balls": 0,
        "catches": 0, "stumpings": 0, "run_outs": 0, "points": 0
    })

    # --- BATTING ---
    batting_data = defaultdict(list)
    for innings in data.get("innings", []):
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                batter = delivery.get("batter")
                if batter:
                    runs = delivery["runs"].get("batter", 0)
                    extras_dict = delivery.get("extras", {})
                    faced = 0 if "wides" in extras_dict or "noballs" in extras_dict else 1
                    
                    batting_data[batter].append({
                        "runs": runs,
                        "faced": faced,
                        "delivery": delivery
                    })
                    player_stats[batter]["runs"] += runs
                    player_stats[batter]["balls_faced"] += faced
                    if runs == 4:
                        player_stats[batter]["fours"] += 1
                    elif runs == 6:
                        player_stats[batter]["sixes"] += 1

    for player, deliveries in batting_data.items():
        p_points = 0
        total_runs = sum(d["runs"] for d in deliveries)
        balls_faced = sum(d["faced"] for d in deliveries)
        
        p_points += total_runs  # 1 point per run (includes fours/sixes runs)
        p_points += player_stats[player]["fours"] * 4  # 1 bonus point per four
        p_points += player_stats[player]["sixes"] * 6  # 2 bonus points per six
        
        if total_runs >= 100:
            p_points += 16
        elif total_runs >= 75:
            p_points += 12
        elif total_runs >= 50:
            p_points += 8
        elif total_runs >= 25:
            p_points += 4
        
        if total_runs == 0:
            for d in deliveries:
                if "wickets" in d["delivery"]:
                    for w in d["delivery"]["wickets"]:
                        if w.get("player_out") == player:
                            p_points -= 2
                            break
                    break
        
        if balls_faced >= 10:
            sr = (total_runs / balls_faced) * 100
            if sr > 170:
                p_points += 6
            elif sr > 150:
                p_points += 4
            elif sr >= 130:
                p_points += 2
            elif 60 <= sr <= 70:
                p_points -= 2
            elif 50 <= sr < 60:
                p_points -= 4
            elif sr < 50:
                p_points -= 6
        
        points[player] += p_points
        player_stats[player]["points"] += p_points

    # --- BOWLING ---
    bowling_data = defaultdict(list)
    for innings in data.get("innings", []):
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                bowler = delivery.get("bowler")
                if bowler:
                    runs_conceded = delivery["runs"].get("total", 0)
                    bowling_data[bowler].append({
                        "runs_conceded": runs_conceded,
                        "delivery": delivery,
                        "over": over.get("over")
                    })
                    player_stats[bowler]["runs_conceded"] += runs_conceded
                    player_stats[bowler]["balls_bowled"] += 1 if "extras" not in delivery or "wides" not in delivery["extras"] else 0
                    if runs_conceded == 0:
                        player_stats[bowler]["dot_balls"] += 1

    for bowler, balls in bowling_data.items():
        b_points = 0
        dot_balls = sum(1 for b in balls if b["runs_conceded"] == 0)
        b_points += dot_balls * 1
        
        wicket_count = 0
        lbw_bowled_wickets = 0
        for b in balls:
            delivery = b["delivery"]
            if "wickets" in delivery:
                for w in delivery["wickets"]:
                    kind = w.get("kind")
                    if kind != "run out":
                        wicket_count += 1
                        player_stats[bowler]["wickets"] += 1
                        if kind in ["lbw", "bowled"]:
                            lbw_bowled_wickets += 1
        
        b_points += wicket_count * 25
        b_points += lbw_bowled_wickets * 8
        
        if wicket_count >= 3:
            if wicket_count == 3:
                b_points += 4
            elif wicket_count == 4:
                b_points += 8
            elif wicket_count >= 5:
                b_points += 12
        
        overs = {}
        for b in balls:
            over_no = b["over"]
            overs.setdefault(over_no, 0)
            overs[over_no] += b["runs_conceded"]
        maiden_overs = sum(1 for total in overs.values() if total == 0)
        b_points += maiden_overs * 12
        
        balls_bowled = len(balls)
        if balls_bowled >= 12:
            overs_bowled = balls_bowled / 6
            total_runs_conceded = sum(b["runs_conceded"] for b in balls)
            economy = total_runs_conceded / overs_bowled
            if economy < 5:
                b_points += 6
            elif 5 <= economy <= 5.99:
                b_points += 4
            elif 6 <= economy <= 7:
                b_points += 2
            elif 10 <= economy <= 11:
                b_points -= 2
            elif 11.01 <= economy <= 12:
                b_points -= 4
            elif economy > 12:
                b_points -= 6
        
        points[bowler] += b_points
        player_stats[bowler]["points"] += b_points

    # --- FIELDING ---
    catch_counts = defaultdict(int)
    for innings in data.get("innings", []):
        for over in innings.get("overs", []):
            for delivery in over.get("deliveries", []):
                if "wickets" in delivery:
                    for wicket in delivery["wickets"]:
                        if "fielders" in wicket:
                            for fielder in wicket["fielders"]:
                                name = fielder.get("name")
                                if name:
                                    kind = wicket.get("kind")
                                    if kind == "caught":
                                        points[name] += 8
                                        player_stats[name]["catches"] += 1
                                        catch_counts[name] += 1
                                    elif kind == "stumped":
                                        points[name] += 12
                                        player_stats[name]["stumpings"] += 1
                                    elif kind == "run out":
                                        points[name] += 12 if len(wicket["fielders"]) == 1 else 6
                                        player_stats[name]["run_outs"] += 1

    for fielder, count in catch_counts.items():
        if count >= 3:
            points[fielder] += 4
            player_stats[fielder]["points"] += 4

    # --- LINEUP BONUS ---
    for team in data["info"]["players"]:
        for player in data["info"]["players"][team]:
            points[player] += 4
            player_stats[player]["points"] += 4

    # Debug high scores
    for player, p in points.items():
        if p > 200:
            print(f"{player}: {p} points - {player_stats[player]}")

    # Build dataset
    for player in set(list(points.keys()) + list(data["info"]["players"][teams[0]]) + list(data["info"]["players"][teams[1]])):
        stats = player_stats[player]
        role = player_roles.get(player, "Unknown") if player_roles else "Unknown"
        opposition = teams[1] if player in data["info"]["players"][teams[0]] else teams[0]
        dataset.append({
            "match_id": match_id,
            "player": player,
            "team": teams[0] if player in data["info"]["players"][teams[0]] else teams[1],
            "opposition": opposition,
            "venue": venue,
            "season": season,
            "role": role,
            "runs": stats["runs"],
            "balls_faced": stats["balls_faced"],
            "fours": stats["fours"],
            "sixes": stats["sixes"],
            "wickets": stats["wickets"],
            "runs_conceded": stats["runs_conceded"],
            "balls_bowled": stats["balls_bowled"],
            "dot_balls": stats["dot_balls"],
            "catches": stats["catches"],
            "stumpings": stats["stumpings"],
            "run_outs": stats["run_outs"],
            "points": stats["points"]
        })

    return dataset

def create_dataset_with_rolling(json_folder, player_roles=None, min_season=2021, window=5):
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"Folder {json_folder} does not exist")
    
    all_data = []
    for root, _, files in os.walk(json_folder):
        for file in sorted(files):
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    season = data["info"].get("season", "Unknown")
                    season_year = int(str(season).split("/")[1]) + 2000 if "/" in str(season) else int(season) if str(season).isdigit() else 0
                    if season_year >= min_season:
                        match_data = process_match(file_path, player_roles)
                        all_data.extend(match_data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {file_path}: {e}")
    
    df = pd.DataFrame(all_data)
    df["match_id"] = df["match_id"].astype(str)
    df = df.sort_values("match_id")
    
    for player in df["player"].unique():
        player_df = df[df["player"] == player].sort_values("match_id")
        df.loc[player_df.index, "avg_points_last_5"] = player_df["points"].rolling(window=5, min_periods=1).mean().shift(1)
        df.loc[player_df.index, "avg_runs_last_5"] = player_df["runs"].rolling(window=5, min_periods=1).mean().shift(1)
        df.loc[player_df.index, "avg_wickets_last_5"] = player_df["wickets"].rolling(window=5, min_periods=1).mean().shift(1)
    
    df.fillna(0, inplace=True)
    df.to_csv("dream11_dataset_with_rolling.csv", index=False)
    return df

# Example usage
json_folder = "/home/parth/iplgameathon/fantasy-sports-machine-learning/data/ipl/ipl_json_ball_by_ball"
player_roles = {"Virat Kohli": "Batsman", "Jasprit Bumrah": "Bowler"}
dataset = create_dataset_with_rolling(json_folder, player_roles)
print(dataset.head())