from datetime import datetime, timedelta


def get_date_range(months_back=3):
    """
    Calculate the start and end dates for the past 'months_back' months,
    ending yesterday instead of today.
    Returns formatted strings for the Scrapper class.
    """
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=months_back * 30)
    return start_date.strftime("%d+%b+%Y"), end_date.strftime("%d+%b+%Y")
