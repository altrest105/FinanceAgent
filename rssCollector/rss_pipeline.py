import os
import pandas as pd
from tqdm import tqdm

from rss_funcs import collect_from_rss, summarize_text, translate_to_russian


def process_rss_sources(input_csv="source_rss.csv", output_csv="rss_base.csv"):

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Файл '{input_csv}' не найден")

    df_sources = pd.read_csv(input_csv)

    if "rss" not in df_sources.columns:
        raise ValueError("source_rss.csv должен содержать колонку 'rss'")

    all_news = []

    for url in tqdm(df_sources["rss"].tolist()):
        try:
            items = collect_from_rss(url)
        except Exception as e:
            print(f"❌ Ошибка RSS {url}: {e}")
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

            all_news.append({
                "title": title_ru,
                "summary": summary,
                "full_content": content_ru,
                "link": link,
                "published": published,
                "source": url,
            })

    df_out = pd.DataFrame(all_news)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    process_rss_sources()