def merge(csv1, csv2):
    weight_prev = 0.3
    weight_recent = 0.7
    merged = csv1.merge(
        csv2,
        on=["Player", "Player Type", "Team", "Credits"],
        suffixes=("_csv1", "_csv2"),
    )

    merged["Batting Form"] = (
        weight_prev * merged["Batting Form_csv1"]
        + weight_recent * merged["Batting Form_csv2"]
    )
    merged["Bowling Form"] = (
        weight_prev * merged["Bowling Form_csv1"]
        + weight_recent * merged["Bowling Form_csv2"]
    )
    merged["Fielding Form"] = (
        weight_prev * merged["Fielding Form_csv1"]
        + weight_recent * merged["Fielding Form_csv2"]
    )

    final_df = merged[
        [
            "Player",
            "Batting Form",
            "Bowling Form",
            "Fielding Form",
            "Credits",
            "Player Type",
            "Team",
        ]
    ]

    print("saving to the merger_ouput.csv file ........")
    final_df.to_csv("data/recent_averages/merged_output.csv", index=False)
