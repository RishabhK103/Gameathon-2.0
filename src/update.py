import os
from datetime import datetime
from src.scrapper.scrapper import Scrapper

from src.scrapper.player_form import UpdatePlayerForm
from src.utils import get_date_range, clean_files, merge

CURRENT_DATE = datetime.now()


def update_player_data(months_back=3):
    """
    Scrapes IPL data for the past 'months_back' months and updates existing files.
    """
    if months_back < 1:
        raise ValueError("Months back must be a positive integer.")

    spanmin1, spanmax1 = get_date_range(months_back)
    print(
        f"Scraping IPL data from {spanmin1.replace('+', ' ')} to {spanmax1.replace('+', ' ')}..."
    )

    # Use the Scrapper class directly with the calculated date range
    scrapper = Scrapper(spanmin1, spanmax1)

    # Override output files to temporary locations
    scrapper.output_files = {
        "batting": "data/recent_averages/batting_recent_averages_temp.csv",
        "bowling": "data/recent_averages/bowling_recent_averages_temp.csv",
        "fielding": "data/recent_averages/fielding_recent_averages_temp.csv",
    }
    for file_path in scrapper.output_files.values():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Run the scraper
    scrapper.scrape_and_clean()

    clean_files()
    UpdatePlayerForm()
    merge()

    print("IPL data update completed successfully.")
