import pandas as pd

scored = pd.read_csv("data/scored_news.csv")
raw    = pd.read_csv("data/raw_news.csv")
price  = pd.read_csv("data/price_data.csv")

coin_gpt   = scored[scored["ticker"]=="COIN"][["date","sentiment_score"]].rename(columns={"sentiment_score":"gpt_score"})
coin_price = price[price["ticker"]=="COIN"][["date","label"]]
merged     = coin_gpt.merge(coin_price, on="date", how="inner")

print("GPT accuracy at different confidence thresholds:")
print()
print("Strong BEARISH predictions (score < threshold):")
for t in [-1, -2, -3, -4, -5]:
    subset = merged[merged["gpt_score"] < t]
    if len(subset) > 0:
        acc = (subset["label"] == 0).mean()
        print(f"  score < {t:>3}: {acc:.1%} DOWN accuracy on {len(subset):>3} days")

print()
print("Strong BULLISH predictions (score > threshold):")
for t in [1, 2, 3, 4, 5]:
    subset = merged[merged["gpt_score"] > t]
    if len(subset) > 0:
        acc = (subset["label"] == 1).mean()
        print(f"  score > {t:>2}: {acc:.1%} UP accuracy on {len(subset):>3} days")