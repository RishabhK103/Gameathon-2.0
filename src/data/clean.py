import os
import pandas as pd


def clean_files():
    """
    Update existing CSV files with new data, removing overlapping spans.
    """

    data_dir = "data/recent_averages"

    data_types = ["batting", "bowling", "fielding"]
    for data_type in data_types:
        old_file = os.path.join(data_dir, f"{data_type}.csv")
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
