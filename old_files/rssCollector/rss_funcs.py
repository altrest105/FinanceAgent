import feedparser
import requests
from bs4 import BeautifulSoup

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from googletrans import Translator

MAX_NEWS = 5

translator = Translator()


def fetch_full_text(url, fallback=None):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")

        article = soup.find("article")
        if article:
            return article.get_text(separator="\n").strip()

        paragraphs = soup.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text() for p in paragraphs)
            return text.strip()

        return fallback

    except Exception:
        return fallback


def collect_from_rss(url):
    feed = feedparser.parse(url)
    news = []

    for entry in feed.entries[:MAX_NEWS]:
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")
        summary = entry.get("summary", "")

        full = fetch_full_text(link, fallback=summary)

        news.append({
            "title": title,
            "link": link,
            "published": published,
            "content": full
        })

    return news


def summarize_text(text, sentences_count=2):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("russian"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        content = " ".join(str(s) for s in summary)
        return content if content else None
    except Exception:
        return None


def translate_to_russian(text):
    if not text:
        return ""
    try:
        return translator.translate(text, src="auto", dest="ru").text
    except Exception:
        return text
