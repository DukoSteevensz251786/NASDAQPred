import requests
import pandas as pd
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta

load_dotenv()

API_KEY   = os.getenv("MARKETAUX_API_KEY")
BASE_URL  = "https://api.marketaux.com/v1/news/all"
TICKERS   = ["COIN"]

MAX_REQUESTS  = 2400  # today's safe limit — change to 2400 tomorrow
MAX_PAGES     = 25  # today's page cap — change to 25 tomorrow
SLEEP_BETWEEN = 2.0

REQUESTS_MADE = 0


def load_existing(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "date" in df.columns and len(df) > 0:
            done = set(zip(df["ticker"], df["date"].str[:7]))
            print(f"Resuming — {len(df)} articles already collected")
            print(f"Skipping {len(done)} ticker/month combinations already done\n")
            return df, done
    return pd.DataFrame(), set()


def fetch_month(ticker, start_str, end_str):
    global REQUESTS_MADE

    month_articles = []
    page = 1

    while True:
        if REQUESTS_MADE >= MAX_REQUESTS:
            print(f"\n  ⚠️  Daily limit reached ({REQUESTS_MADE} requests). Run again tomorrow.")
            return month_articles, True

        params = {
            "symbols":          ticker,
            "filter_entities":  "true",
            "language":         "en",
            "published_after":  start_str,
            "published_before": end_str,
            "page":             page,
            "api_token":        API_KEY,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            data     = response.json()
            REQUESTS_MADE += 1
        except Exception as e:
            print(f"    Request error: {e} — retrying in 10s...")
            time.sleep(10)
            continue

        if "error" in data:
            code = data["error"].get("code", "")
            msg  = data["error"].get("message", "")
            print(f"    API error [{code}]: {msg}")
            if code in ("usage_limit_reached", "rate_limit_reached"):
                return month_articles, True
            break

        articles = data.get("data", [])
        if not articles:
            break

        for a in articles:
            sentiment_score = 0
            for entity in a.get("entities", []):
                if entity.get("symbol") == ticker:
                    sentiment_score = entity.get("sentiment_score", 0)
                    break

            month_articles.append({
                "ticker":          ticker,
                "date":            a.get("published_at", "")[:10],
                "headline":        a.get("title", ""),
                "summary":         a.get("description", ""),
                "source":          a.get("source", ""),
                "url":             a.get("url", ""),
                "sentiment_score": sentiment_score,
            })

        meta        = data.get("meta", {})
        total_found = meta.get("found", 0)
        total_pages = -(-total_found // len(articles)) if articles else 1

        print(f"    {start_str[:10]} → {end_str[:10]} | "
              f"page {page}/{min(total_pages, MAX_PAGES)} | "
              f"+{len(month_articles)} articles | "
              f"requests: {REQUESTS_MADE}/{MAX_REQUESTS}")

        if page >= total_pages or page >= MAX_PAGES:
            break

        page += 1
        time.sleep(SLEEP_BETWEEN)

    return month_articles, False


def fetch_news_for_ticker(ticker, months, already_done_months):
    global REQUESTS_MADE

    print(f"  Fetching {ticker}...")
    all_articles = []
    end_date     = datetime.today()
    skipped      = 0

    for i in range(months):
        if REQUESTS_MADE >= MAX_REQUESTS:
            return all_articles, True

        month_end   = end_date - timedelta(days=i * 30)
        month_start = month_end - timedelta(days=30)

        start_str = month_start.strftime("%Y-%m-%dT%H:%M")
        end_str   = month_end.strftime("%Y-%m-%dT%H:%M")
        month_key = (ticker, end_str[:7])

        if month_key in already_done_months:
            skipped += 1
            continue

        articles, limit_hit = fetch_month(ticker, start_str, end_str)
        all_articles.extend(articles)

        time.sleep(SLEEP_BETWEEN)

        if limit_hit:
            return all_articles, True

    if skipped > 0:
        print(f"    Skipped {skipped} months already collected")

    return all_articles, False


def collect_all_news(months=24):
    print(f"Collecting {months} months of news from Marketaux")
    print(f"Today's settings — requests: {MAX_REQUESTS} | pages: {MAX_PAGES} | sleep: {SLEEP_BETWEEN}s")
    print(f"Tomorrow — change MAX_REQUESTS to 2400 and MAX_PAGES to 25\n")

    os.makedirs("data", exist_ok=True)
    existing_path = "data/raw_news.csv"

    existing_df, already_done_months = load_existing(existing_path)
    all_new_articles = []

    for ticker in TICKERS:
        if REQUESTS_MADE >= MAX_REQUESTS:
            print("Daily limit reached — run again tomorrow.")
            break

        articles, limit_hit = fetch_news_for_ticker(ticker, months, already_done_months)
        all_new_articles.extend(articles)

        if articles:
            new_df   = pd.DataFrame(all_new_articles)
            combined = pd.concat([existing_df, new_df], ignore_index=True) \
                       if not existing_df.empty else new_df
            combined.to_csv(existing_path, index=False)
            print(f"  {ticker}: {len(articles)} new articles saved | "
                  f"total in file: {len(combined)}\n")
        else:
            print(f"  {ticker}: nothing new to collect\n")

        time.sleep(3)

        if limit_hit:
            print("Daily limit hit — run again tomorrow to continue.")
            break

    # final dedup and sort
    df = pd.read_csv(existing_path)
    df = df.drop_duplicates(subset=["ticker", "headline"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_csv(existing_path, index=False)

    print(f"\nDone. {len(df)} total articles in raw_news.csv")
    print(f"Requests used today: {REQUESTS_MADE}")
    print(df.groupby("ticker").size().rename("article_count"))
    return df


if __name__ == "__main__":
    collect_all_news(months=24)