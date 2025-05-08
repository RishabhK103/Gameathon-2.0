import pandas as pd
import pulp
import os # Import os to check file existence

def optimize_fantasy_team():
    # --- File Paths ---
    ground_file = "data/ground.csv"
    squad_file = "data/SquadPlayerNames.csv" # Assuming this contains IsPlaying, lineupOrder etc.
    # Use the output file from the previous script
    form_file = "data/recent_averages/merged_output.csv"

    # --- Check if input files exist ---
    required_files = [ground_file, squad_file, form_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required input file not found: {f}")
            print("Please ensure all necessary data files are present.")
            return None

    # --- Load Data ---
    try:
        ground_df = pd.read_csv(ground_file)
        squad_df = pd.read_csv(squad_file)
        form_df = pd.read_csv(form_file)
        print("Successfully loaded ground, squad, and form data.")
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None


    # --- Ground Selection ---
    print("\nGrounds : ")
    for i, r in ground_df.iterrows():
        print(f"{i + 1}. {r['Ground']} ({r['City']})")

    try:
        ground_number = int(input(f"\nEnter the ground number (1-{len(ground_df)}) for the match: "))
        if ground_number < 1 or ground_number > len(ground_df):
            raise ValueError("Ground number out of range.")
        ground_index = ground_number - 1 # Adjust to 0-based index
        selected_ground = ground_df.iloc[ground_index]["Ground"]
        print(f"Selected Ground: {selected_ground}")
    except ValueError as e:
        print(f"Invalid input! {e}. Please enter a number between 1 and {len(ground_df)}.")
        return None
    except Exception as e:
        print(f"An error occurred during ground selection: {e}")
        return None


    # --- Data Cleaning and Preparation ---

    # Clean form_df: Drop columns not needed for optimization.
    # Conditionally drop 'Fielding Form' if it exists.
    print(f"Form data columns before cleaning: {form_df.columns.tolist()}")
    cols_to_drop_form = ["Credits", "Player Type", "Team"] # Base columns to drop
    if "Fielding Form" in form_df.columns:
        cols_to_drop_form.append("Fielding Form")
        print("Found 'Fielding Form' column, will be dropped.")
    else:
        print("'Fielding Form' column not found in form data (as expected).")

    # Check if columns actually exist before dropping
    existing_cols_to_drop_form = [col for col in cols_to_drop_form if col in form_df.columns]
    if existing_cols_to_drop_form:
         form_df.drop(columns=existing_cols_to_drop_form, inplace=True)
         print(f"Dropped columns from form data: {existing_cols_to_drop_form}")
    print(f"Form data columns after cleaning: {form_df.columns.tolist()}")


    # Clean squad_df: Keep only necessary columns
    # Ensure required columns exist
    required_squad_cols = ["Player Name", "Team", "Player Type", "IsPlaying", "lineupOrder"]
    if not all(col in squad_df.columns for col in required_squad_cols):
        print(f"Error: Squad file '{squad_file}' is missing required columns.")
        print(f"Required: {required_squad_cols}")
        print(f"Found: {squad_df.columns.tolist()}")
        return None
    # Keep only necessary columns from squad_df
    squad_df = squad_df[required_squad_cols]


    # Filter squad_df for playing players
    playing_df = squad_df[
        squad_df["IsPlaying"].str.upper().isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"Number of players marked as playing/substitute: {len(playing_df)}")
    if len(playing_df) < 11:
        print("Warning: Fewer than 11 players marked as playing/substitute.")
        # Decide whether to proceed or exit
        # return None


    # Merge playing players with their form scores
    # Use left merge to keep all playing players, even if form score is missing
    selection_df = pd.merge(
        playing_df,
        form_df,
        left_on="Player Name", # Assumes player name matches between squad and form files
        right_on="Player",     # Assumes 'Player' is the name column in form_df
        how="left"
    )
    print(f"Shape after merging playing squad with form data: {selection_df.shape}")

    # Handle potential duplicate 'Player' column if 'Player' was in form_df
    if 'Player' in selection_df.columns and 'Player Name' in selection_df.columns:
         # Keep 'Player Name' as the definitive name, drop the one from form_df
         selection_df.drop(columns=['Player'], inplace=True)
         selection_df.rename(columns={'Player Name': 'Player'}, inplace=True) # Rename for consistency


    # Select and rename final columns for optimization
    final_cols = [
        "Player", "Team", "Player Type", "Batting Form", "Bowling Form", "lineupOrder"
    ]
    # Check if all expected columns are present after merge
    missing_cols = [col for col in final_cols if col not in selection_df.columns]
    if missing_cols:
        print(f"Error: Missing expected columns after merge: {missing_cols}")
        print(f"Available columns: {selection_df.columns.tolist()}")
        return None

    selection_df = selection_df[final_cols]


    # Handle missing form scores by filling with a default (e.g., 0 or average)
    # Using 0 might undervalue players with missing data. Consider using average/median if appropriate.
    initial_bat_nan = selection_df["Batting Form"].isna().sum()
    initial_bowl_nan = selection_df["Bowling Form"].isna().sum()
    selection_df["Batting Form"] = selection_df["Batting Form"].fillna(0)
    selection_df["Bowling Form"] = selection_df["Bowling Form"].fillna(0)
    if initial_bat_nan > 0 or initial_bowl_nan > 0:
        print(f"Filled {initial_bat_nan} missing Batting Form scores and {initial_bowl_nan} missing Bowling Form scores with 0.")


    # --- Calculate Ground-Adjusted Scores ---
    ground_data = ground_df.iloc[ground_index]
    # Ensure weights are floats
    try:
        batter_weight = float(ground_data["Batting"])
        keeper_weight = float(ground_data["Batting"]) # Keepers often valued for batting
        bowler_weight = float(ground_data["Bowling"])
    except KeyError as e:
        print(f"Error: Missing 'Batting' or 'Bowling' column in ground file '{ground_file}': {e}")
        return None
    except ValueError as e:
        print(f"Error: Non-numeric value found in 'Batting' or 'Bowling' column in ground file: {e}")
        return None

    # Calculate all-rounder weight (ensure it's at least 1)
    allrounder_weight = max(1.0, (batter_weight + bowler_weight) / 2)

    print(f"Ground Weights - Batter: {batter_weight:.2f}, Keeper: {keeper_weight:.2f}, Bowler: {bowler_weight:.2f}, All-Rounder: {allrounder_weight:.2f}")

    def calculate_score(row):
        # Normalize Player Type for reliable comparison
        role = str(row["Player Type"]).strip().upper()
        batting = row["Batting Form"]
        bowling = row["Bowling Form"]

        if role == "BAT":
            return batter_weight * batting
        elif role == "WK":
            return keeper_weight * batting # Primarily batting score for WK
        elif role == "BOWL":
            return bowler_weight * bowling
        elif role == "ALL":
            # Consider a weighted average or max based on strategy
            # Using max emphasizes their primary strength in form
            return allrounder_weight * max(batting, bowling)
            # Alternative: return allrounder_weight * (0.6 * batting + 0.4 * bowling) # Example weighted avg
        else:
            print(f"Warning: Unknown Player Type '{row['Player Type']}' for player {row['Player']}. Assigning score 0.")
            return 0 # Default score for unknown types

    selection_df["Score"] = selection_df.apply(calculate_score, axis=1)
    print("Calculated ground-adjusted scores for players.")


    # --- Optimization Setup ---
    team_df = selection_df.copy()
    team_df.reset_index(drop=True, inplace=True) # Ensure index is sequential for PuLP

    # Define player groups based on normalized Player Type
    team_df['Player Type Norm'] = team_df['Player Type'].str.strip().str.upper()
    batters = team_df[team_df["Player Type Norm"] == "BAT"]
    bowlers = team_df[team_df["Player Type Norm"] == "BOWL"]
    allrounders = team_df[team_df["Player Type Norm"] == "ALL"]
    keepers = team_df[team_df["Player Type Norm"] == "WK"]
    # Bowling options include Bowlers and All-rounders
    bowling_options = team_df[team_df["Player Type Norm"].isin(["BOWL", "ALL"])]

    # --- Check Constraints Feasibility ---
    print("\nChecking player availability for constraints:")
    print(f"  Available Batters (BAT): {len(batters)}")
    print(f"  Available Keepers (WK): {len(keepers)}")
    print(f"  Available All-Rounders (ALL): {len(allrounders)}")
    print(f"  Available Bowlers (BOWL): {len(bowlers)}")
    print(f"  Available Bowling Options (BOWL+ALL): {len(bowling_options)}")

    min_bat, min_bowl, min_wk, min_all, min_bowl_opt = 1, 1, 1, 1, 5 # Define minimums

    constraints_met = True
    if len(batters) < min_bat: print(f"  Not enough Batters (need {min_bat})"); constraints_met = False
    if len(keepers) < min_wk: print(f"  Not enough Keepers (need {min_wk})"); constraints_met = False
    if len(allrounders) < min_all: print(f"  Not enough All-Rounders (need {min_all})"); constraints_met = False
    if len(bowlers) < min_bowl: print(f"  Not enough Bowlers (need {min_bowl})"); constraints_met = False
    if len(bowling_options) < min_bowl_opt: print(f"  Not enough Bowling Options (need {min_bowl_opt})"); constraints_met = False
    if len(team_df) < 11: print(f"  Not enough total players available (need 11, have {len(team_df)})"); constraints_met = False

    if not constraints_met:
        print("Error: Cannot satisfy optimization constraints with the available playing XI.")
        return None

    # --- PuLP Optimization ---
    prob = pulp.LpProblem("Fantasy_Team_Optimization", pulp.LpMaximize)
    players_indices = team_df.index.tolist()

    # Decision variable: 1 if player i is selected, 0 otherwise
    x = pulp.LpVariable.dicts("player", players_indices, cat="Binary")

    # Objective Function: Maximize total score
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players_indices]), "Total_Score"

    # Constraints
    prob += pulp.lpSum([x[i] for i in players_indices]) == 11, "Total_11_Players"
    prob += pulp.lpSum([x[i] for i in batters.index]) >= min_bat, f"Min_{min_bat}_Batters"
    prob += pulp.lpSum([x[i] for i in keepers.index]) >= min_wk, f"Min_{min_wk}_Keepers"
    prob += pulp.lpSum([x[i] for i in allrounders.index]) >= min_all, f"Min_{min_all}_Allrounders"
    prob += pulp.lpSum([x[i] for i in bowlers.index]) >= min_bowl, f"Min_{min_bowl}_Bowlers"
    prob += pulp.lpSum([x[i] for i in bowling_options.index]) >= min_bowl_opt, f"Min_{min_bowl_opt}_Bowling_Options"

    # Add team constraints (e.g., max 7 players per team) - Optional
    # teams = team_df['Team'].unique()
    # for team in teams:
    #     team_indices = team_df[team_df['Team'] == team].index
    #     prob += pulp.lpSum([x[i] for i in team_indices]) <= 7, f"Max_7_from_{team}"


    # Solve the problem
    print("\nSolving optimization problem...")
    prob.solve()

    # Check solution status
    status = pulp.LpStatus[prob.status]
    print(f"Optimization Status: {status}")

    if status != "Optimal":
        print("Optimization did not find an optimal solution. Check constraints or player data.")
        return None

    # --- Extract and Format Results ---
    selected_indices = [i for i in players_indices if pulp.value(x[i]) == 1]
    selected_11 = team_df.loc[selected_indices].copy()

    # Sort by score to help with C/VC selection
    selected_11.sort_values("Score", ascending=False, inplace=True)

    # Assign Captain (C) and Vice-Captain (VC)
    selected_11["Role_In_Team"] = "Player" # Default role

    # Prioritize players higher up the order (lower lineupOrder) for C/VC
    # Define a threshold for preferred C/VC lineup order
    preferred_order_threshold = 5
    captain_candidates = selected_11[selected_11["lineupOrder"] < preferred_order_threshold].sort_values("Score", ascending=False)

    if len(captain_candidates) >= 1:
        captain_idx = captain_candidates.index[0]
        selected_11.loc[captain_idx, "Role_In_Team"] = "Captain"
        print(f"Assigned Captain: {selected_11.loc[captain_idx, 'Player']}")

        # Find VC among remaining preferred candidates or highest remaining scorer
        vc_candidates = captain_candidates.drop(captain_idx)
        if len(vc_candidates) >= 1:
            vice_captain_idx = vc_candidates.index[0]
            selected_11.loc[vice_captain_idx, "Role_In_Team"] = "Vice Captain"
            print(f"Assigned Vice Captain (preferred order): {selected_11.loc[vice_captain_idx, 'Player']}")
        else:
            # If no other preferred candidate, pick highest scorer among remaining players
            remaining_players = selected_11[selected_11["Role_In_Team"] == "Player"].sort_values("Score", ascending=False)
            if len(remaining_players) > 0:
                vice_captain_idx = remaining_players.index[0]
                selected_11.loc[vice_captain_idx, "Role_In_Team"] = "Vice Captain"
                print(f"Assigned Vice Captain (next best score): {selected_11.loc[vice_captain_idx, 'Player']}")
            else:
                 print("Warning: Could not assign Vice Captain (only Captain selected).")

    elif len(selected_11) > 0: # Fallback if no one meets lineupOrder threshold
        print("Warning: No players found with lineupOrder < {preferred_order_threshold}. Assigning C/VC based purely on highest score.")
        captain_idx = selected_11.index[0] # Highest scorer overall
        selected_11.loc[captain_idx, "Role_In_Team"] = "Captain"
        print(f"Assigned Captain (highest score): {selected_11.loc[captain_idx, 'Player']}")
        if len(selected_11) > 1:
            vice_captain_idx = selected_11.index[1] # Second highest scorer overall
            selected_11.loc[vice_captain_idx, "Role_In_Team"] = "Vice Captain"
            print(f"Assigned Vice Captain (second highest score): {selected_11.loc[vice_captain_idx, 'Player']}")
    else:
        print("Error: No players selected in the final team.")
        return None


    # Return the final selected team details
    return selected_11[["Player", "Team", "Player Type", "Score", "Role_In_Team"]]

# --- Main Execution ---
def main():
    print("Starting Fantasy Team Optimization...")
    best_team = optimize_fantasy_team()

    if best_team is not None:
        print("\n" + "="*30)
        print("   Optimal Fantasy Team")
        print("="*30)
        # Format output for better readability
        print(best_team.to_string(index=False))
        print("="*30)
        total_score = best_team['Score'].sum()
        print(f"Predicted Total Score: {total_score:.2f}")
        print("="*30)
    else:
        print("\nCould not generate an optimal fantasy team based on the provided data and constraints.")

if __name__ == "__main__":
    main()
