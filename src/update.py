# update.py
# ==============================================================================
# Update Player Data Script
# This script updates IPL cricket data from ESPNcricinfo by scraping data for a specified
# number of recent months using the Scrapper class from ipl20scrapper.py, cleaning it,
# and merging it with existing data files.
#
# Instructions:
# - Run with an integer argument for months back, e.g.:
#       python3 update.py 3
#
# Dependencies:
# - Requires ipl20scrapper.py in the same directory or appropriate path.
#
# ==============================================================================

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from ipl20scrapper import Scrapper

from playerform import UpdatePlayerForm  

# Get the current system date dynamically
CURRENT_DATE = datetime.now()

def get_date_range(months_back=3):
    """
    Calculate the start and end dates for the past 'months_back' months.
    Returns formatted strings for the Scrapper class.
    """
    end_date = CURRENT_DATE
    start_date = end_date - timedelta(days=months_back * 30)  # Approximate months as 30 days
    return start_date.strftime("%d+%b+%Y"), end_date.strftime("%d+%b+%Y")

def update_existing_data_files(data_dir="../data/ipl"):
    """
    Update existing CSV files with new data, removing overlapping spans.
    """
    data_types = ["batting", "bowling", "fielding"]
    for data_type in data_types:
        old_file = os.path.join(data_dir, f"{data_type}_recent_averages.csv")
        new_file = os.path.join(data_dir, f"{data_type}_recent_averages_temp.csv")
        
        if not os.path.exists(new_file):
            print(f"New file {new_file} not found. Skipping {data_type} update.")
            continue
        
        new_df = pd.read_csv(new_file)
        if not os.path.exists(old_file):
            new_df.to_csv(old_file, index=False)
            print(f"Created new file {old_file} with updated {data_type} data.")
            os.remove(new_file)
            continue
        
        old_df = pd.read_csv(old_file)
        new_span = new_df["Span"].iloc[0] if "Span" in new_df.columns else None
        if new_span and "Span" in old_df.columns:
            updated_old_df = old_df[old_df["Span"] != new_span]
        else:
            updated_old_df = old_df  # If no Span column, append without filtering
        updated_df = pd.concat([updated_old_df, new_df], ignore_index=True)
        updated_df.to_csv(old_file, index=False)
        os.remove(new_file)
        print(f"Updated {data_type} data in {old_file}.")

def update_player_data(months_back=3):
    """
    Scrapes IPL data for the past 'months_back' months and updates existing files.
    """
    if months_back < 1:
        raise ValueError("Months back must be a positive integer.")
    
    # Calculate date range
    spanmin1, spanmax1 = get_date_range(months_back)
    print(f"Scraping IPL data from {spanmin1.replace('+', ' ')} to {spanmax1.replace('+', ' ')}...")

    # Use the Scrapper class directly with the calculated date range
    scrapper = Scrapper(spanmin1, spanmax1)
    
    # Override output files to temporary locations
    scrapper.output_files = {
        "batting": "../data/ipl/batting_recent_averages_temp.csv",
        "bowling": "../data/ipl/bowling_recent_averages_temp.csv",
        "fielding": "../data/ipl/fielding_recent_averages_temp.csv"
    }
    for file_path in scrapper.output_files.values():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Run the scraper
    scrapper.scrape_and_clean()

    # Update existing files with new data
    update_existing_data_files()


    UpdatePlayerForm()

    print("IPL data update completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 update.py <months_back>")
        sys.exit(1)
    try:
        months_back = int(sys.argv[1])
        update_player_data(months_back)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)