import pandas as pd
import time
from datetime import datetime
from src.scrape import Scrapper
from src.calc import PlayerForm
from src.merge import clean_files,merge
from src.optimize import optimize_fantasy_team

if __name__ == "__main__":
    end_date = datetime.now()

    # Calculate start date as 3 months before the end date
    # Using pandas DateOffset for easy month subtraction
    start_date = end_date - pd.DateOffset(months=3)

    # Format dates into the required string format (DD+Mon+YYYY)
    spanmax1_str = end_date.strftime("%d+%b+%Y")
    spanmin1_str = start_date.strftime("%d+%b+%Y")

    print(f"Scraping data from {spanmin1_str} to {spanmax1_str} (Last 3 months)") # Updated print statement

    scraper = Scrapper(spanmin1=spanmin1_str, spanmax1=spanmax1_str)
    scraper.scrape_and_clean()
    
    preprocessor=PlayerForm()
    preprocessor.run()
 
    clean_files()
    merge()
   

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

