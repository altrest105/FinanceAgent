import os
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time

from rss_funcs import collect_from_rss, summarize_text, translate_to_russian, fetch_full_text
from rss_pipeline import process_rss_sources, RSS_FILE, SOURCES_FILE

def clean_old_news(file_path=RSS_FILE):
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path, parse_dates=["published"], dayfirst=True)
    one_week_ago = datetime.now() - timedelta(days=7)
    rows_to_keep = []

    for idx, row in df.iterrows():
        if pd.isnull(row["published"]):
            continue

        if row["published"] >= one_week_ago:
            rows_to_keep.extend(df.loc[idx:].to_dict("records"))
            break

    if rows_to_keep:
        df_new = pd.DataFrame(rows_to_keep)
        df_new.to_csv(file_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=df.columns).to_csv(file_path, index=False, encoding="utf-8-sig")


def rss_job():
    process_rss_sources(SOURCES_FILE, RSS_FILE)

def cleanup_job():
    clean_old_news(RSS_FILE)


schedule.every(10).minutes.do(rss_job)
schedule.every(30).minutes.do(cleanup_job)

def start_aging_process():
    print("RSS-pipeline")
    while True:
        schedule.run_pending()
        time.sleep(5)