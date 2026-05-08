import pandas as pd
import yfinance as yf
import os

TICKERS = ["COIN"]


def get_vix(months=24):
    """Fetch VIX data directly from yfinance."""
    print("Fetching VIX from yfinance...")
    vix = yf.download("^VIX", period=f"{months}mo", interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix["date"] = pd.to_datetime(vix["Date"]).dt.strftime("%Y-%m-%d")
    vix = vix[["date", "Close"]].rename(columns={"Close": "vix"})
    print(f"  VIX data: {len(vix)} days ({vix['date'].min()} to {vix['date'].max()})")
    return vix


def shift_to_next_trading_day(scored_df, price_df):
    """Move weekend/holiday news to the next trading day."""
    trading_days = set(price_df["date"].unique())

    def next_trading_day(date):
        if date in trading_days:
            return date
        d = pd.Timestamp(date) + pd.Timedelta(days=1)
        for _ in range(10):
            if d.strftime("%Y-%m-%d") in trading_days:
                return d.strftime("%Y-%m-%d")
            d += pd.Timedelta(days=1)
        return None

    scored_df = scored_df.copy()
    scored_df["date"] = scored_df["date"].apply(next_trading_day)
    scored_df = scored_df.dropna(subset=["date"])

    scored_df = scored_df.groupby(["ticker", "date"]).agg(
        sentiment_score=("sentiment_score", "mean"),
        article_count=("article_count", "sum")
    ).reset_index()

    return scored_df


def merge_data():
    # load datasets
    price_df  = pd.read_csv("data/price_data.csv")
    scored_df = pd.read_csv("data/scored_news.csv")
    vix_df    = get_vix(months=24)

    # filter to target tickers
    price_df  = price_df[price_df["ticker"].isin(TICKERS)]
    scored_df = scored_df[scored_df["ticker"].isin(TICKERS)]

    print(f"Price data:  {len(price_df)} rows")
    print(f"Scored news: {len(scored_df)} days (before shift)")

    # keep only what we need from scored news
    news_agg = scored_df[["ticker", "date", "sentiment_score", "article_count"]].copy()

    # shift weekend/holiday news to next trading day
    news_agg = shift_to_next_trading_day(news_agg, price_df)
    print(f"Scored news: {len(news_agg)} days (after shift)")

    # normalize GPT sentiment per ticker
    news_agg["sentiment_score"] = news_agg.groupby("ticker")["sentiment_score"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # merge price + sentiment
    merged = pd.merge(price_df, news_agg, on=["ticker", "date"], how="left")

    # merge VIX
    merged = pd.merge(merged, vix_df, on="date", how="left")

    # forward fill VIX for missing days (holidays etc)
    merged["vix"] = merged["vix"].ffill()

    # fill missing news days
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0)
    merged["article_count"]   = merged["article_count"].fillna(0)
    merged["has_news"]        = (merged["article_count"] > 0).astype(int)

    # drop raw price columns
    merged = merged.drop(columns=["close", "bb_upper", "bb_lower"], errors="ignore")

    # sort
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)

    # sanity checks
    print(f"\nMerged:             {len(merged)} rows")
    print(f"Days with news:     {(merged['article_count'] > 0).sum()}")
    print(f"Days without news:  {(merged['article_count'] == 0).sum()}")
    print(f"Sentiment coverage: {(merged['sentiment_score'] != 0).sum() / len(merged):.1%}")
    print(f"VIX nulls:          {merged['vix'].isna().sum()}")
    print(f"VIX range:          {merged['vix'].min():.1f} to {merged['vix'].max():.1f}")
    print(f"\nLabel distribution:")
    print(merged.groupby("ticker")["label"].value_counts().unstack())
    print(f"\nCorrelation with label:")
    numeric = merged.select_dtypes(include="number").drop(columns=["label"])
    print(numeric.corrwith(merged["label"]).sort_values(ascending=False).round(3))
    print(f"\nFeature columns: {list(merged.columns)}")

    # save
    os.makedirs("data", exist_ok=True)
    merged.to_csv("data/training_data.csv", index=False)
    print(f"\nDone. Saved to data/training_data.csv")

    return merged


if __name__ == "__main__":
    merge_data()