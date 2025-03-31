import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from selenium.common.exceptions import TimeoutException
from datetime import datetime
import urllib3.exceptions

# Browser and driver setup
brave_path = "/usr/bin/brave"
chrome_driver_path = "/usr/bin/chromedriver"
options = Options()
options.binary_location = brave_path
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

driver.set_page_load_timeout(600)

# Function to parse a date string and return a datetime object
def parse_date(date_string):
    try:
        return datetime.strptime(date_string, "%d %b %Y")
    except ValueError:
        print("Invalid date format. Please use DD MMM YYYY (e.g., 17 Feb 2023).")
        return None

# Function to get and validate user input for dates
def get_date_input():
    while True:
        start_date_str = input("Enter start date (e.g., 17 Feb 2023): ").strip()
        start_date = parse_date(start_date_str)
        if start_date is None:
            continue

        end_date_str = input("Enter end date (e.g., 22 Oct 2024): ").strip()
        end_date = parse_date(end_date_str)
        if end_date is None:
            continue

        if end_date <= start_date:
            print("End date must be after start date. Please try again.")
            continue

        spanmin1 = start_date.strftime("%d+%b+%Y")
        spanmax1 = end_date.strftime("%d+%b+%Y")
        return spanmin1, spanmax1

class Scrapper:
    def __init__(self):
        self.ipl_teams_codes = {
            "KKR": "4341", "CSK": "4343", "MI": "4346", "RCB": "4340", "SRH": "5143",
            "RR": "4345", "PBKS": "4342", "DC": "4344", "GT": "6904", "LSG": "6903",
        }
        self.spanmin1, self.spanmax1 = get_date_input()
        # Updated URL template with page parameter
        self.url_template = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=6;page={page};spanmax1={spanmax1};spanmin1={spanmin1};spanval1=span;team={team};template=results;type={type}"
        self.base_urls = {
            "bowling": self.url_template.format(page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="bowling"),
            "batting": self.url_template.format(page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="batting"),
            "fielding": self.url_template.format(page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team="4340", type="fielding")
        }
        self.output_files = {
            "batting": "../data/ipl/batting_averages_overall.csv",
            "bowling": "../data/ipl/bowling_averages_overall.csv",
            "fielding": "../data/ipl/fielding_averages_overall.csv"
        }

    def clean_data(self, df, data_type):
        if df is None or df.empty:
            print(f"No data to clean for {data_type}")
            return df
        try:
            # Replace empty or placeholder values with NaN
            df.replace(["-", "", " "], np.nan, inplace=True)
            if "Player" in df.columns:
                df["Player"] = df["Player"].str.strip()
            if data_type == "batting":
                # Remove asterisk from HS (Highest Score)
                if "HS" in df.columns:
                    df["HS"] = df["HS"].str.replace("*", "", regex=False)
                int_cols = ["Mat", "Inns", "NO", "Runs", "BF", "100", "50", "0", "4s", "6s"]
                float_cols = ["Ave", "SR"]
            elif data_type == "bowling":
                int_cols = ["Mat", "Inns", "Mdns", "Runs", "Wkts", "4", "5"]
                float_cols = ["Ave", "Econ", "SR"]
            elif data_type == "fielding":
                int_cols = ["Mat", "Inns", "Dis", "Ct", "St", "Ct Wk", "Ct Fi", "MD"]
                float_cols = ["D/I"]
            else:
                int_cols = []
                float_cols = []
            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        except Exception as e:
            print(f"Error cleaning data for {data_type}: {e}")
        return df

    def scrape_and_clean(self):
        for file_path in self.output_files.values():
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Test phase (unchanged)
        test_data_type = "batting"
        test_url = self.base_urls[test_data_type]
        test_headers = ["Team", "Player", "Span", "Mat", "Inns", "NO", "Runs", "HS", "Ave", "BF", "SR", "100", "50", "0", "4s", "6s"]

        print(f"\nTesting hardcoded URL for {test_data_type}: {test_url}")
        test_data = []
        retries = 3
        for attempt in range(retries):
            try:
                driver.get(test_url)
                table = WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((
                        By.XPATH,
                        "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]"
                    ))
                )
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if not cells:
                        continue
                    cell_data = [cell.text for cell in cells if cell.text.strip() != ""]
                    if not cell_data:
                        continue
                    row_data = ["RCB"] + cell_data
                    if len(row_data) != len(test_headers):
                        print(f"Column mismatch in test data: expected {len(test_headers)} columns, got {len(row_data)}")
                        print(f"Row data: {row_data}")
                        continue
                    test_data.append(row_data)
                print(f"Successfully scraped {len(test_data)} rows for {test_data_type} from RCB:")
                if test_data:
                    print(f"Sample row: {test_data[0]}")
                    df_test = pd.DataFrame(test_data, columns=test_headers)
                    print(df_test.head())
                else:
                    print("No data scraped in test phase.")
                break
            except (TimeoutException, urllib3.exceptions.ReadTimeoutError) as e:
                print(f"Timeout while testing {test_url} on attempt {attempt + 1}/{retries}: {e}")
                if attempt == retries - 1:
                    print("Failed to load test URL after all retries.")
                    with open("timeout_page.html", "w", encoding="utf-8") as f:
                        f.write(driver.page_source)
                    print("Page source saved to 'timeout_page.html' for debugging.")
                    driver.quit()
                    return
                time.sleep(5)
            except Exception as e:
                print(f"Error testing {test_url}: {e}")
                driver.quit()
                return

        proceed = input("\nDoes the test data look correct? (yes/no): ").strip().lower()
        if proceed != "yes":
            print("Aborting full scrape.")
            driver.quit()
            return

        print("\nProceeding with full scrape...")
        for data_type, base_url in self.base_urls.items():
            output_file = self.output_files[data_type]
            if data_type == "batting":
                headers = ["Team", "Player", "Span", "Mat", "Inns", "NO", "Runs", "HS", "Ave", "BF", "SR", "100", "50", "0", "4s", "6s"]
            elif data_type == "bowling":
                headers = ["Team", "Player", "Span", "Mat", "Inns", "Overs", "Mdns", "Runs", "Wkts", "BBI", "Ave", "Econ", "SR", "4", "5"]
            else:  # Fielding
                headers = ["Team", "Player", "Span", "Mat", "Inns", "Dis", "Ct", "St", "Ct Wk", "Ct Fi", "MD", "D/I"]
            all_data = []
            for team, code in self.ipl_teams_codes.items():
                retries = 3
                for attempt in range(retries):
                    try:
                        # Load the first page to determine the total number of pages
                        url = self.url_template.format(page=1, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team=code, type=data_type)
                        print(f"Scraping {data_type} data for {team}, page 1 to check pagination...")
                        driver.get(url)
                        # Wait for the table to ensure the page is loaded
                        WebDriverWait(driver, 30).until(
                            EC.presence_of_element_located((
                                By.XPATH,
                                "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]"
                            ))
                        )

                        # Wait for pagination links to load (if they exist)
                        total_pages = 1  # Default to 1 page
                        try:
                            # Wait for any pagination link to appear
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    "//a[contains(@class, 'PaginationLink')]"
                                ))
                            )
                            # Find the "Last" link
                            last_page_link = driver.find_element(
                                By.XPATH,
                                "//a[contains(@class, 'PaginationLink') and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'last')]"
                            )
                            last_page_url = last_page_link.get_attribute("href")
                            total_pages = int(last_page_url.split("page=")[1].split(";")[0])
                            print(f"Found pagination for {team} ({data_type}): {total_pages} pages")
                        except Exception as e:
                            print(f"No pagination found for {team} ({data_type}), assuming 1 page. ")
                            # Save page source for debugging
                            # with open(f"pagination_debug_{team}_{data_type}.html", "w", encoding="utf-8") as f:
                            #     f.write(driver.page_source)
                            # print(f"Page source saved to 'pagination_debug_{team}_{data_type}.html' for debugging.")

                        # Loop through all pages dynamically
                        for page in range(1, total_pages + 1):
                            url = self.url_template.format(page=page, spanmax1=self.spanmax1, spanmin1=self.spanmin1, team=code, type=data_type)
                            print(f"Scraping {data_type} data for {team}, page {page}...")
                            driver.get(url)
                            table = WebDriverWait(driver, 30).until(
                                EC.presence_of_element_located((
                                    By.XPATH,
                                    "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]"
                                ))
                            )
                            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                            for row in rows:
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if not cells:
                                    continue
                                cell_data = [cell.text for cell in cells if cell.text.strip() != ""]
                                if not cell_data:
                                    continue
                                row_data = [team] + cell_data
                                if len(row_data) != len(headers):
                                    print(f"Column mismatch for {team} ({data_type}, page {page}): expected {len(headers)} columns, got {len(row_data)}")
                                    print(f"Row data: {row_data}")
                                    continue
                                all_data.append(row_data)
                        print(f"Scraped {data_type} averages for {team}: {total_pages} pages")
                        break
                    except (TimeoutException, urllib3.exceptions.ReadTimeoutError) as e:
                        print(f"Timeout for {team} ({data_type}) on attempt {attempt + 1}/{retries}: {e}")
                        if attempt == retries - 1:
                            print(f"Failed to scrape {data_type} data for {team} after {retries} attempts.")
                        time.sleep(5)
                    except Exception as e:
                        print(f"Error scraping {team} ({data_type}): {e}")
                        break
                    time.sleep(1)

            if not all_data:
                print(f"No data scraped for {data_type}")
                continue

            df = pd.DataFrame(all_data, columns=headers)
            df = self.clean_data(df, data_type)
            df.to_csv(output_file, index=False)
            print(f"Saved cleaned {data_type} data to {output_file}")
scrapper = Scrapper()
scrapper.scrape_and_clean()

driver.quit()