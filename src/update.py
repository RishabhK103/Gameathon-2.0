import sys
from datetime import datetime, timedelta
import os
import pandas as pd
# Assuming both scripts are in the same directory or adjust paths accordingly
from ipl20scrapper import Scrapper  # Replace with your actual scraper module name
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
    return start_date.strftime("%d %b %Y"), end_date.strftime("%d %b %Y")

def update_ipl_data():
    """
    Scrapes IPL data for the past 3 months and updates player form scores.
    """
    try:
        # Step 1: Set up date range (past 3 months to current date)
        spanmin1, spanmax1 = get_date_range(months_back=3)
        print(f"Scraping IPL data from {spanmin1} to {spanmax1}...")

        # Step 2: Automated Scraper with pre-set date range and CSV file checks
        class AutomatedScrapper(Scrapper):
            def __init__(self):
                self.ipl_teams_codes = {
                    "KKR": "4341", "CSK": "4343", "MI": "4346", "RCB": "4340", "SRH": "5143",
                    "RR": "4345", "PBKS": "4342", "DC": "4344", "GT": "6904", "LSG": "6903",
                }
                # Use the date range calculated earlier
                self.spanmin1 = spanmin1
                self.spanmax1 = spanmax1
                self.url_template = (
                    "https://stats.espncricinfo.com/ci/engine/stats/index.html?"
                    "class=6;page={page};spanmax1={spanmax1};spanmin1={spanmin1};spanval1=span;"
                    "team={team};template=results;type={type}"
                )
                self.base_urls = {
                    "bowling": self.url_template.format(
                        page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="bowling"
                    ),
                    "batting": self.url_template.format(
                        page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="batting"
                    ),
                    "fielding": self.url_template.format(
                        page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="fielding"
                    )
                }
                self.output_files = {
                    "batting": "../data/ipl/batting_recent_averages.csv",
                    "bowling": "../data/ipl/bowling_recent_averages.csv",
                    "fielding": "../data/ipl/fielding_recent_averages.csv"
                }

                # Ensure CSV files exist before scraping
                self.ensure_csv_files()

            def ensure_csv_files(self):
                """
                Creates CSV files if they don't exist, adding appropriate headers.
                """
                headers = {
                    "batting": ["Player", "Matches", "Runs", "Average", "Strike Rate"],
                    "bowling": ["Player", "Matches", "Wickets", "Average", "Economy"],
                    "fielding": ["Player", "Matches", "Catches", "Run Outs"]
                }

                for category, file_path in self.output_files.items():
                    # Create the directory if it doesn't exist
                    dir_name = os.path.dirname(file_path)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    # Create the file with headers if it doesn't exist
                    if not os.path.exists(file_path):
                        print(f"Creating new file: {file_path}")
                        df = pd.DataFrame(columns=headers[category])
                        df.to_csv(file_path, index=False)

        # Step 3: Initialize and run the scrapper
        scrapper = AutomatedScrapper()
        scrapper.scrape_and_clean()

        # Step 4: Update player form with the newly scraped data
        print("\nUpdating player form scores...")
        UpdatePlayerForm()

        print("IPL data update completed successfully.")

    except Exception as e:
        print(f"An error occurred during the update process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        update_ipl_data()
    except KeyboardInterrupt:
        print("\nUpdate process interrupted by user.")
        sys.exit(1)
