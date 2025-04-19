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
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

driver.set_page_load_timeout(600)


class Scrapper:
    def __init__(self, spanmin1, spanmax1):
        self.ipl_teams_codes = {
            "KKR": "4341",
            "CSK": "4343",
            "MI": "4346",
            "RCB": "4340",
            "SRH": "5143",
            "RR": "4345",
            "PBKS": "4342",
            "DC": "4344",
            "GT": "6904",
            "LSG": "6903",
        }
        self.spanmin1 = spanmin1  # e.g., "01+Jan+2025"
        self.spanmax1 = spanmax1  # e.g., "31+Mar+2025"
        self.url_template = (
            "https://stats.espncricinfo.com/ci/engine/stats/index.html?"
            "class=6;page={page};spanmax1={spanmax1};spanmin1={spanmin1};"
            "spanval1=span;team={team};template=results;type={type}"
        )
        self.base_urls = {
            "bowling": self.url_template.format(
                page=1,
                spanmax1=self.spanmax1,
                spanmin1=self.spanmin1,
                team="4340",
                type="bowling",
            ),
            "batting": self.url_template.format(
                page=1,
                spanmax1=self.spanmax1,
                spanmin1=self.spanmin1,
                team="4340",
                type="batting",
            ),
            "fielding": self.url_template.format(
                page=1,
                spanmax1=self.spanmax1,
                spanmin1=self.spanmin1,
                team="4340",
                type="fielding",
            ),
        }
        self.output_files = {
            "batting": "data/recent_averages/batting.csv",
            "bowling": "data/recent_averages/bowling.csv",
            "fielding": "data/recent_averages/fielding.csv",
        }

    def clean_data(self, df, data_type):
        if df is None or df.empty:
            print(f"No data to clean for {data_type}")
            return df
        try:
            df.replace(["-", "", " "], np.nan, inplace=True)
            if "Player" in df.columns:
                df["Player"] = df["Player"].str.strip()
            if data_type == "batting":
                if "HS" in df.columns:
                    df["HS"] = df["HS"].str.replace("*", "", regex=False)
                int_cols = [
                    "Mat",
                    "Inns",
                    "NO",
                    "Runs",
                    "BF",
                    "100",
                    "50",
                    "0",
                    "4s",
                    "6s",
                ]
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

        print("\nProceeding with full scrape...")
        for data_type, _ in self.base_urls.items():
            output_file = self.output_files[data_type]
            if data_type == "batting":
                headers = [
                    "Team",
                    "Player",
                    "Mat",
                    "Inns",
                    "NO",
                    "Runs",
                    "HS",
                    "Ave",
                    "BF",
                    "SR",
                    "100",
                    "50",
                    "0",
                    "4s",
                    "6s",
                ]
            elif data_type == "bowling":
                headers = [
                    "Team",
                    "Player",
                    "Mat",
                    "Inns",
                    "Overs",
                    "Mdns",
                    "Runs",
                    "Wkts",
                    "BBI",
                    "Ave",
                    "Econ",
                    "SR",
                    "4",
                    "5",
                ]
            else:  # Fielding
                headers = [
                    "Team",
                    "Player",
                    "Mat",
                    "Inns",
                    "Dis",
                    "Ct",
                    "St",
                    "Ct Wk",
                    "Ct Fi",
                    "MD",
                    "D/I",
                ]
            all_data = []
            for team, code in self.ipl_teams_codes.items():
                retries = 3
                for attempt in range(retries):
                    try:
                        url = self.url_template.format(
                            page=1,
                            spanmax1=self.spanmax1,
                            spanmin1=self.spanmin1,
                            team=code,
                            type=data_type,
                        )
                        print(
                            f"Scraping {data_type} data for {team}, page 1 to check pagination..."
                        )
                        driver.get(url)
                        WebDriverWait(driver, 30).until(
                            EC.presence_of_element_located(
                                (
                                    By.XPATH,
                                    "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]",
                                )
                            )
                        )

                        total_pages = 1
                        try:
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located(
                                    (
                                        By.XPATH,
                                        "//a[contains(@class, 'PaginationLink')]",
                                    )
                                )
                            )
                            last_page_link = driver.find_element(
                                By.XPATH,
                                "//a[contains(@class, 'PaginationLink') and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'last')]",
                            )
                            last_page_url = last_page_link.get_attribute("href")
                            total_pages = int(
                                last_page_url.split("page=")[1].split(";")[0]
                            )
                            print(
                                f"Found pagination for {team} ({data_type}): {total_pages} pages"
                            )
                        except Exception:
                            print(
                                f"No pagination found for {team} ({data_type}), assuming 1 page."
                            )

                        for page in range(1, total_pages + 1):
                            url = self.url_template.format(
                                page=page,
                                spanmax1=self.spanmax1,
                                spanmin1=self.spanmin1,
                                team=code,
                                type=data_type,
                            )
                            print(
                                f"Scraping {data_type} data for {team}, page {page}..."
                            )
                            driver.get(url)
                            table = WebDriverWait(driver, 30).until(
                                EC.presence_of_element_located(
                                    (
                                        By.XPATH,
                                        "//table[@class='engineTable'][.//caption[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'overall figures')]]",
                                    )
                                )
                            )
                            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                            for row in rows:
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if not cells:
                                    continue
                                cell_data = [
                                    cell.text
                                    for cell in cells
                                    if cell.text.strip() != ""
                                ]
                                if not cell_data:
                                    continue
                                row_data = [team] + cell_data
                                if len(row_data) != len(headers):
                                    print(
                                        f"Column mismatch for {team} ({data_type}, page {page}): expected {len(headers)} columns, got {len(row_data)}"
                                    )
                                    print(f"Row data: {row_data}")
                                    continue
                                all_data.append(row_data)
                        print(
                            f"Scraped {data_type} averages for {team}: {total_pages} pages"
                        )
                        break
                    except (TimeoutException, urllib3.exceptions.ReadTimeoutError) as e:
                        print(
                            f"Timeout for {team} ({data_type}) on attempt {attempt + 1}/{retries}: {e}"
                        )
                        if attempt == retries - 1:
                            print(
                                f"Failed to scrape {data_type} data for {team} after {retries} attempts."
                            )
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

            # Add "Span" column using the provided dates
            start_dt = datetime.strptime(self.spanmin1.replace("+", " "), "%d %b %Y")
            end_dt = datetime.strptime(self.spanmax1.replace("+", " "), "%d %b %Y")
            span_value = f"{start_dt.year}-{end_dt.year}"
            df["Span"] = span_value

            df.to_csv(output_file, index=False)
            print(f"Saved cleaned {data_type} data to {output_file}")

        driver.quit()
