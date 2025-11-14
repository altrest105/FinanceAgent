import os
import pandas as pd
from tqdm import tqdm

from rss_funcs import collect_from_rss, summarize_text, translate_to_russian

SOURCES_FILE = "source_rss.csv"
RSS_FILE = "rss_base.csv"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_rss_sources(
        input_csv=SOURCES_FILE,
        output_csv=RSS_FILE
    ):

    input_csv = os.path.join(BASE_DIR, input_csv)
    output_csv = os.path.join(BASE_DIR, output_csv)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Файл '{input_csv}' не найден")

    df_sources = pd.read_csv(input_csv)

    if "rss" not in df_sources.columns:
        raise ValueError("source_rss.csv должен содержать колонку 'rss'")

    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_summaries = set(df_existing["summary"].astype(str).tolist())
    else:
        df_existing = pd.DataFrame()
        existing_summaries = set()

    all_news = []

    for url in tqdm(df_sources["rss"].tolist()):
        try:
            items = collect_from_rss(url)
        except Exception as e:
            print(f"Ошибка {url}: {e}")
            continue

        for item in items:
            title = item.get("title", "")
            content = item.get("content", "")
            link = item.get("link", "")
            published = item.get("published", "")

            title_ru = translate_to_russian(title)
            content_ru = translate_to_russian(content)

            summary = summarize_text(content_ru)
            if summary is None:
                summary = content_ru[:500]

            if summary in existing_summaries:
                continue

            existing_summaries.add(summary)

            all_news.append({
                "title": title_ru,
                "summary": summary,
                "full_content": content_ru,
                "link": link,
                "published": published,
                "source": url,
            })

    if len(all_news) > 0:
        df_new = pd.DataFrame(all_news)
        if df_existing.empty:
            df_out = df_new
        else:
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_existing

    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")