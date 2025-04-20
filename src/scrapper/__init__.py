import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime
from tqdm import tqdm


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
        self.spanmin1 = spanmin1
        self.spanmax1 = spanmax1
        self.url_template = (
            "https://stats.espncricinfo.com/ci/engine/stats/index.html?"
            "class=6;page={page};spanmax1={spanmax1};spanmin1={spanmin1};"
            "spanval1=span;team={team};template=results;type={type}"
        )
        self.data_types_to_scrape = ["batting", "bowling"]
        self.output_files = {
            "batting": "data/recent_averages/batting_data.csv",
            "bowling": "data/recent_averages/bowling_data.csv",
        }
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        self.request_timeout = 60

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
            else:
                print(f"Warning: Unknown data_type '{data_type}' in clean_data")
                return df

            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        except Exception as e:
            print(f"Error cleaning data for {data_type}: {e}")
        return df

    def find_data_table(self, soup):
        """Finds the correct stats table (the one with 'Overall Figures' caption)."""
        tables = soup.find_all("table", class_="engineTable")
        for table in tables:
            caption = table.find("caption")  # Find any caption tag within the table
            # Now check if the found caption exists and has the correct text
            if caption and "overall figures" in caption.get_text(strip=True).lower():
                return table  # Return the table if the caption text matches
        return None  # Return None if no matching table/caption is found

    def scrape_and_clean(self):
        # Ensure base directory exists *before* the loop
        try:
            # Get dir from the first defined output file path
            base_dir = os.path.dirname(next(iter(self.output_files.values())))
            # Create directory if it doesn't exist
            os.makedirs(base_dir, exist_ok=True)
            print(f"Ensured output directory exists: {os.path.abspath(base_dir)}")
        except Exception as e:
            print(
                f"CRITICAL ERROR: Could not create output directory '{base_dir}'. Error: {e}"
            )
            print("Please check permissions and path. Exiting.")
            return  # Stop execution if directory can't be created

        print(
            "\nProceeding with scrape for Batting and Bowling using Requests and BeautifulSoup..."
        )

        # Outer tqdm for data types
        for data_type in tqdm(self.data_types_to_scrape, desc="Data Types"):
            output_file = self.output_files[data_type]
            absolute_output_path = os.path.abspath(
                output_file
            )  # Get absolute path for clarity

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
            else:
                print(f"Skipping unknown data type: {data_type}")
                continue

            all_data = []
            print(f"\n--- Starting scrape for {data_type.capitalize()} ---")

            # Team progress bar
            team_pbar = tqdm(self.ipl_teams_codes.items(), desc=f"Teams ({data_type})")

            # --- Scraping loop for teams and pages ---
            for team, code in team_pbar:
                team_pbar.set_description(f"Team: {team}")
                retries = 3
                current_page = 1
                total_pages = 1
                first_page_processed = False

                # Initialize page progress bar (will update total later)
                page_pbar = tqdm(total=1, desc=f"Pages for {team}", leave=False)

                while current_page <= total_pages:
                    attempt = 0
                    success = False
                    page_pbar.set_description(f"Page {current_page}/{total_pages}")

                    while attempt < retries and not success:
                        try:
                            url = self.url_template.format(
                                page=current_page,
                                spanmax1=self.spanmax1,
                                spanmin1=self.spanmin1,
                                team=code,
                                type=data_type,
                            )

                            response = self.session.get(
                                url, timeout=self.request_timeout
                            )
                            response.raise_for_status()
                            page_content = response.text

                            soup = BeautifulSoup(page_content, "html.parser")

                            if "No results available" in page_content.lower():
                                if current_page == 1:
                                    total_pages = 0
                                else:
                                    total_pages = current_page - 1
                                success = True
                                break

                            table = self.find_data_table(soup)
                            if not table:
                                raise ValueError(
                                    f"Data table structure not found and 'No results available' message absent."
                                )

                            if not first_page_processed:
                                try:
                                    pagination_div = soup.find(
                                        "div", class_="pagination"
                                    )
                                    if pagination_div:
                                        last_page_link = pagination_div.find(
                                            "a",
                                            string=lambda t: t and "last" in t.lower(),
                                        )
                                        if (
                                            last_page_link
                                            and "href" in last_page_link.attrs
                                        ):
                                            last_page_url = last_page_link["href"]
                                            total_pages = int(
                                                last_page_url.split("page=")[1].split(
                                                    ";"
                                                )[0]
                                            )
                                        else:
                                            total_pages = current_page
                                    else:
                                        total_pages = current_page
                                except (
                                    AttributeError,
                                    IndexError,
                                    ValueError,
                                    TypeError,
                                ) as e:
                                    total_pages = current_page
                                finally:
                                    total_pages = max(total_pages, current_page)
                                    first_page_processed = True
                                    # Update total in progress bar once we know it
                                    page_pbar.total = total_pages
                                    page_pbar.refresh()

                            rows = table.find_all("tr", class_="data1")
                            if not rows:
                                rows = table.find_all("tr")[1:]

                            rows_processed_this_page = 0
                            # Using tqdm for rows processing
                            for row in rows:
                                cells = row.find_all("td")
                                if not cells:
                                    continue

                                # Extract text, stripping whitespace
                                cell_data = [
                                    cell.get_text(strip=True) for cell in cells
                                ]

                                # Skip rows that might be empty or separators
                                if not any(cell_data):
                                    continue

                                # --- FIX: Handle potential extra column ---
                                expected_data_columns = (
                                    len(headers) - 1
                                )  # Number of columns expected from HTML table

                                if len(cell_data) > expected_data_columns:
                                    # Slice off extra column(s) assumed to be at the end
                                    cell_data = cell_data[:expected_data_columns]
                                # --- END FIX ---

                                # Now create the final row data including the team
                                row_data = [team] + cell_data

                                # Check length against headers *after* potential slicing
                                if len(row_data) != len(headers):
                                    continue  # Skip this row if mismatch still occurs

                                # Append the correctly formatted row
                                all_data.append(row_data)
                                rows_processed_this_page += 1

                            success = True

                        except requests.exceptions.Timeout as e:
                            attempt += 1
                            if attempt == retries:
                                total_pages = current_page - 1
                                # Update progress bar
                                page_pbar.total = total_pages
                                page_pbar.refresh()
                            else:
                                time.sleep(5)
                        except requests.exceptions.RequestException as e:
                            attempt += 1
                            if attempt == retries:
                                total_pages = current_page - 1
                                # Update progress bar
                                page_pbar.total = total_pages
                                page_pbar.refresh()
                            else:
                                time.sleep(5)
                        except Exception as e:
                            total_pages = current_page - 1
                            # Update progress bar
                            page_pbar.total = total_pages
                            page_pbar.refresh()
                            break  # Exit retry loop for this page

                    if not success:
                        break  # Exit the while loop for pages

                    current_page += 1
                    page_pbar.update(1)
                    if current_page <= total_pages:
                        time.sleep(1)  # Be polite between page requests

                # Close the page progress bar when done with this team
                page_pbar.close()
                time.sleep(1)  # Delay between teams
            # --- End of scraping loop ---

            # --- DataFrame creation and saving ---
            if not all_data:
                print(
                    f"No data collected for {data_type} across all teams. Skipping file save."
                )
                continue  # Skip to the next data_type

            print(f"Creating DataFrame for {data_type} with {len(all_data)} rows.")
            df = pd.DataFrame(all_data, columns=headers)

            # Show progress during data cleaning
            with tqdm(total=1, desc=f"Cleaning {data_type} data") as clean_pbar:
                df = self.clean_data(df, data_type)
                clean_pbar.update(1)

            # Check if DataFrame is empty *after* cleaning
            if df.empty:
                print(
                    f"DataFrame for {data_type} is empty after cleaning. Skipping file save."
                )
                continue  # Skip to the next data_type

            # Add Span column
            try:
                start_dt = datetime.strptime(
                    self.spanmin1.replace("+", " "), "%d %b %Y"
                )
                end_dt = datetime.strptime(self.spanmax1.replace("+", " "), "%d %b %Y")
                span_value = f"{start_dt.year}-{end_dt.year}"
                df["Span"] = span_value
            except ValueError as e:
                print(
                    f"Warning: Error parsing date strings '{self.spanmin1}', '{self.spanmax1}': {e}. Cannot add Span column."
                )

            try:
                with tqdm(total=1, desc=f"Saving {data_type} data") as save_pbar:
                    print(
                        f"Saving {data_type} data ({df.shape[0]} rows) to: {absolute_output_path}"
                    )
                    df.to_csv(output_file, index=False)
                    save_pbar.update(1)
                print(f"Successfully saved cleaned {data_type} data to {output_file}")
            except IOError as e:
                print(
                    f"!!! I/O ERROR saving {data_type} data to {absolute_output_path}: {e}"
                )
                print("   Please check file permissions and if the path is valid.")
            except Exception as e:
                print(
                    f"!!! UNEXPECTED ERROR saving {data_type} data to {absolute_output_path}: {e}"
                )
