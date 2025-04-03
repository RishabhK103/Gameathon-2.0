import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def ensure_directory(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

class Scraper:
    def __init__(self):
        self.ipl_teams_codes = {
            "KKR": "4341", "CHE": "4343", "MI": "4346", "RCB": "4340", "SRH": "5143",
            "RR": "4345", "PBKS": "4342", "DC": "4344", "GT": "6904", "LSG": "6903",
        }
        self.ground_ids = {
            "Eden Gardens": "292", "M.Chinnaswamy Stadium": "683", "MA Chidambaram Stadium": "291",
            "Wankhede Stadium": "713", "Rajiv Gandhi International Stadium": "1981"
        }
        self.url_template = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=6;page={page};spanmax1={spanmax1};spanmin1={spanmin1};spanval1=span;team={team};ground={ground};opposition={opposition};template=results;type={type}"
        
        # Enhanced Chrome options
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.set_page_load_timeout(300)
        self.driver.set_script_timeout(300)
        self.driver.implicitly_wait(10)

    def clean_data(self, df, data_type):
        if df is None or df.empty:
            print(f"No data to clean for {data_type}")
            return df
        
        df.replace(["-", "", " ", "NA"], np.nan, inplace=True)
        
        if "Player" in df.columns:
            df["Player"] = df["Player"].str.strip()
        
        # More robust data type conversion
        if data_type == "batting":
            int_cols = ["Mat", "Inns", "NO", "Runs", "BF", "100", "50", "0", "4s", "6s"]
            float_cols = ["Ave", "SR"]
            
            if "HS" in df.columns:
                df["HS"] = df["HS"].str.replace("*", "", regex=False)
        elif data_type == "bowling":
            int_cols = ["Mat", "Inns", "Mdns", "Runs", "Wkts", "4", "5"]
            float_cols = ["Ave", "Econ", "SR"]
        else:
            return df
        
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        
        return df

    def scrape_data(self, data_type, spanmin1, spanmax1, team=None, ground=None, opposition=None, output_file=None):
        # Skip if file already exists with non-zero size
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Skipping {data_type} scrape for {output_file} as it already exists")
            return

        headers = {
            "batting": ["Team", "Player", "Span", "Mat", "Inns", "NO", "Runs", "HS", "Ave", "BF", "SR", "100", "50", "0", "4s", "6s"],
            "bowling": ["Team", "Player", "Span", "Mat", "Inns", "Overs", "Mdns", "Runs", "Wkts", "BBI", "Ave", "Econ", "SR", "4", "5"]
        }[data_type]

        all_data = []
        teams = [team] if team else self.ipl_teams_codes.keys()

        for team_name in teams:
            team_code = self.ipl_teams_codes[team_name]
            retries = 5  # Increased retry attempts
            
            for attempt in range(retries):
                try:
                    # Construct initial URL
                    url = self.url_template.format(
                        page=1, spanmax1=spanmax1, spanmin1=spanmin1, team=team_code,
                        ground=ground if ground else "", 
                        opposition=opposition if opposition else "", 
                        type=data_type
                    )
                    print(f"Scraping {data_type} for {team_name}, attempt {attempt + 1}/{retries}: {url}")
                    
                    # Navigate to page
                    self.driver.get(url)
                    
                    # Wait for table to load
                    table = WebDriverWait(self.driver, 30).until(
                        EC.presence_of_element_located((
                            By.XPATH, 
                            "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]"
                        ))
                    )
                    
                    # Determine total pages
                    total_pages = 1
                    try:
                        last_page_link = self.driver.find_element(By.XPATH, "//a[contains(@class, 'PaginationLink') and contains(text(), 'Last')]")
                        total_pages = int(last_page_link.get_attribute("href").split("page=")[1].split(";")[0])
                    except (NoSuchElementException, IndexError):
                        print(f"No pagination found for {team_name}, using single page")
                    
                    print(f"Processing {total_pages} pages for {team_name}")
                    
                    # Scrape data from all pages
                    for page in range(1, total_pages + 1):
                        url = self.url_template.format(
                            page=page, spanmax1=spanmax1, spanmin1=spanmin1, team=team_code,
                            ground=ground if ground else "", 
                            opposition=opposition if opposition else "", 
                            type=data_type
                        )
                        
                        self.driver.get(url)
                        time.sleep(2)  # Small delay to ensure page loads
                        
                        # Find table rows
                        rows = self.driver.find_elements(By.XPATH, "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]/tbody/tr")
                        
                        for row in rows[1:]:  # Skip header row
                            cells = [cell.text.strip() for cell in row.find_elements(By.TAG_NAME, "td") if cell.text.strip()]
                            
                            if cells and len(cells) == len(headers) - 1:
                                all_data.append([team_name] + cells)
                    
                    # If data was successfully scraped, break retry loop
                    if all_data:
                        break
                
                except (TimeoutException, Exception) as e:
                    print(f"Error scraping {data_type} for {team_name} on attempt {attempt + 1}/{retries}: {e}")
                    time.sleep(5)  # Wait before retrying
        
        # Create DataFrame or empty DataFrame if no data
        df = pd.DataFrame(all_data, columns=headers) if all_data else pd.DataFrame(columns=headers)
        
        # Clean and save data
        df = self.clean_data(df, data_type)
        
        # Ensure output directory exists
        ensure_directory(output_file)
        
        # Save data, even if empty
        df.to_csv(output_file, index=False)
        print(f"Saved {data_type} data to {output_file} (rows: {len(df)})")

    def scrape_all(self, venue, team1, team2):
        # Historical data with wider date range
        self.scrape_data("batting", "01+Jan+2020", "01+Jun+2024", output_file="data/historical/historical_batting.csv")
        self.scrape_data("bowling", "01+Jan+2020", "01+Jun+2024", output_file="data/historical/historical_bowling.csv")
        
        # Recent data with narrow date range
        self.scrape_data("batting", "01+Jan+2025", "31+Dec+2025", output_file="data/recent/recent_batting.csv")
        self.scrape_data("bowling", "01+Jan+2025", "31+Dec+2025", output_file="data/recent/recent_bowling.csv")
        
        # Get ground ID
        ground_id = self.ground_ids.get(venue, "")
        
        # Venue and opposition-specific data
        for team in [team1, team2]:
            # Venue data
            self.scrape_data("batting", "01+Jan+2020", "01+Jun+2024", team=team, ground=ground_id,
                             output_file=f"data/venue/venue_batting_{team}_{ground_id}.csv")
            self.scrape_data("bowling", "01+Jan+2020", "01+Jun+2024", team=team, ground=ground_id,
                             output_file=f"data/venue/venue_bowling_{team}_{ground_id}.csv")
            
            # Recent venue data
            self.scrape_data("batting", "01+Jan+2025", "31+Dec+2025", team=team, ground=ground_id,
                             output_file=f"data/venue/recent_venue_batting_{team}_{ground_id}.csv")
            self.scrape_data("bowling", "01+Jan+2025", "31+Dec+2025", team=team, ground=ground_id,
                             output_file=f"data/venue/recent_venue_bowling_{team}_{ground_id}.csv")

        # Opposition-specific data
        for team, opp in [(team1, team2), (team2, team1)]:
            opp_id = self.ipl_teams_codes[opp]
            self.scrape_data("batting", "01+Jan+2020", "01+Jun+2024", team=team, opposition=opp_id,
                             output_file=f"data/opposition/opposition_batting_{team}_vs_{opp}.csv")
            self.scrape_data("bowling", "01+Jan+2020", "01+Jun+2024", team=team, opposition=opp_id,
                             output_file=f"data/opposition/opposition_bowling_{team}_vs_{opp}.csv")

    def close(self):
        try:
            self.driver.quit()
        except Exception as e:
            print(f"Error closing webdriver: {e}")