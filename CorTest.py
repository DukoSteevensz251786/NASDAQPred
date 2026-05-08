import pandas as pd
import yfinance as yf
import openai
import json
import os
from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load recent COIN news
news = pd.read_csv("data/raw_news.csv")
coin_news = news[news["ticker"] == "COIN"].copy()
coin_news["date"] = pd.to_datetime(coin_news["date"])

# get last 30 days
recent = coin_news[coin_news["date"] >= "2026-03-01"].copy()
print(f"Recent COIN articles: {len(recent)}")

# get price data
price = yf.download("COIN", start="2026-03-01", progress=False)
if isinstance(price.columns, pd.MultiIndex):
    price.columns = price.columns.get_level_values(0)
price["date"]      = pd.to_datetime(price.index)
price["return_1d"] = price["Close"].pct_change(1)
price["label"]     = (price["Close"].shift(-1) > price["Close"]).astype(int)

# score each day with GPT
def score_day(ticker, articles):
    headlines = "\n".join([f"- {a}" for a in articles[:10]])
    prompt = f"""You are a financial analyst. Rate how bullish or bearish 
these {ticker} headlines are for tomorrow's stock price.
Scale: -10 (very bearish) to +10 (very bullish).

{headlines}

Reply ONLY with JSON: {{"score": <number>, "reason": "<one sentence>"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except:
        return {"score": 0, "reason": "error"}

# group by date and score
daily = recent.groupby("date")["headline"].apply(list).reset_index()
results = []

for _, row in daily.iterrows():
    result = score_day("COIN", row["headline"])
    results.append({
        "date":      row["date"],
        "gpt_score": result["score"],
        "reason":    result["reason"],
        "n_articles": len(row["headline"])
    })
    print(f"{str(row['date'])[:10]}: {result['score']:+.1f} — {result['reason'][:60]}")

scores_df = pd.DataFrame(results)
scores_df["date"] = pd.to_datetime(scores_df["date"])

# merge with price
merged = scores_df.merge(
    price[["date","return_1d","label"]].assign(date=lambda x: pd.to_datetime(x["date"])),
    on="date", how="inner"
)

print(f"\nDays scored: {len(merged)}")
print(f"\nCorrelation with next-day direction:")
print(merged[["gpt_score","n_articles","label"]].corr()["label"].round(3))

print(f"\nMarketaux vs GPT sentiment comparison:")
marketaux_daily = recent.groupby("date")["sentiment_score"].mean().reset_index()
marketaux_daily["date"] = pd.to_datetime(marketaux_daily["date"])
comp = merged.merge(marketaux_daily, on="date", how="inner")
print(comp[["gpt_score","sentiment_score","label"]].corr()["label"].round(3))