import os
import pandas as pd
from src.utils import get_date_range
from src.scrapper import Scrapper


def update_player_data(months_back=3):
    spanmin1, spanmax1 = get_date_range(months_back)
    print(
        f"Scraping IPL data from {spanmin1.replace('+', ' ')} to {spanmax1.replace('+', ' ')}..."
    )

    scrapper = Scrapper(spanmin1, spanmax1)

    for file_path in scrapper.output_files.values():
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    scrapper.scrape_and_clean()

    csv_path = "data/recent_averages/player_form_scores.csv"
    output_path = "data/recent_averages/player_form_scores_final.csv"

    df = pd.read_csv(csv_path)

    form_columns = ["Batting Form", "Bowling Form", "Fielding Form"]

    role_means = df.groupby("Player Type")[form_columns].mean()

    for col in form_columns:
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(), "Player Type"].map(
            role_means[col]
        )

    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to {output_path}")
