import openai
import pandas as pd
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Set which tickers to score ─────────────────────────────────────────────
TICKERS_TO_SCORE = ["COIN"]  # change this to score different tickers
# ──────────────────────────────────────────────────────────────────────────


def build_prompt(ticker, articles):
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"""
Article {i}:
Headline: {article['headline']}
Summary:  {article['summary']}
"""

    return f"""You are a senior quantitative analyst at a hedge fund specializing in short-term price prediction.

Analyze the following articles/news about {ticker} to estimate likely impact on tomorrow’s price action.

{articles_text}



Instructions:
1. Read all articles and extract key catalysts (earnings, guidance, macro, analyst upgrades/downgrades, litigation, product news, sector trends, insider activity, etc.).

2. For each article, assign:
- Sentiment Score from -10 to +10
  (-10 = extremely bearish, 0 = neutral, +10 = extremely bullish)

- Impact Weight from 1–5
  (1 = low relevance, 5 = market-moving)

- Time Horizon:
  - Immediate (tomorrow)
  - Short-term (days/weeks)
  - Long-term

3. Calculate a weighted aggregate sentiment score:
Weighted Sentiment = Σ(Sentiment × Impact Weight) / Σ(Impact Weights)

4. Estimate probable effect for tomorrow:
- Bullish / Bearish / Neutral bias
- Probability % for Up / Down / Flat session
- Expected magnitude:
  - Mild move (0–1%)
  - Moderate move (1–3%)
  - Large move (3%+)

5. Identify whether sentiment is driven by:
- Fundamentals
- Market positioning
- Momentum/speculation
- Macro factors
- Headline risk

6. Flag conflicting signals or articles that may be overhyped/noise.

Respond ONLY with a JSON object, no other text:
{{"score": <decimal from -10 to 10>, "reason": "<one sentence explaining the key driver>"}}"""


def score_articles(ticker, articles):
    if not articles:
        return {"score": 0, "reason": "No articles for this day"}

    prompt = build_prompt(ticker, articles)

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=100,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["score"] = max(-10, min(10, float(result["score"])))
        return result

    except json.JSONDecodeError:
        print(f"    Warning: could not parse GPT response: {raw}")
        return {"score": 0, "reason": "Parse error"}

    except Exception as e:
        print(f"    Error calling GPT: {e}")
        return {"score": 0, "reason": "API error"}


def score_all_news():
    raw_path = Path(__file__).parent.parent / "data" / "raw_news.csv"
    df = pd.read_csv(raw_path)

    # filter to only the tickers we want to score
    df = df[df["ticker"].isin(TICKERS_TO_SCORE)]
    print(f"Scoring tickers: {TICKERS_TO_SCORE}")
    print(f"Articles to process: {len(df)}")
    print()

    if len(df) == 0:
        print("No articles found for the specified tickers.")
        print(f"Available tickers in raw_news.csv: {pd.read_csv(raw_path)['ticker'].unique().tolist()}")
        return pd.DataFrame()

    # resume support
    scored_path = Path(__file__).parent.parent / "data" / "scored_news.csv"
    if scored_path.exists():
        scored_df    = pd.read_csv(scored_path)
        already_scored = set(zip(scored_df["ticker"], scored_df["date"]))
        print(f"Resuming — {len(scored_df)} days already scored")
        # only count already scored days for our target tickers
        target_scored = len(scored_df[scored_df["ticker"].isin(TICKERS_TO_SCORE)])
        print(f"Already scored for {TICKERS_TO_SCORE}: {target_scored} days\n")
    else:
        scored_df      = pd.DataFrame()
        already_scored = set()

    results = []
    grouped = df.groupby(["ticker", "date"])
    total   = len(grouped)

    # estimate cost
    avg_articles  = len(df) / max(total, 1)
    est_tokens    = total * avg_articles * 150
    est_cost      = est_tokens / 1_000_000 * 0.15
    print(f"Estimated GPT calls: {total}")
    print(f"Estimated cost: ${est_cost:.3f}")
    print()

    for idx, ((ticker, date), group) in enumerate(grouped, 1):

        if (ticker, date) in already_scored:
            continue

        articles = group[["headline", "summary"]].to_dict("records")
        print(f"[{idx}/{total}] {ticker} on {date} ({len(articles)} articles)...")

        result = score_articles(ticker, articles)

        results.append({
            "ticker":          ticker,
            "date":            date,
            "sentiment_score": result["score"],
            "article_count":   len(articles),
            "reason":          result["reason"],
        })

        print(f"  Score: {result['score']:+.1f} — {result['reason']}")

        # save every 10 calls
        if len(results) % 10 == 0:
            save_progress(scored_df, results, scored_path)

        time.sleep(0.5)

    final_df = save_progress(scored_df, results, scored_path)

    # show summary for scored tickers only
    target_df = final_df[final_df["ticker"].isin(TICKERS_TO_SCORE)]
    print(f"\nDone. {len(target_df)} days scored for {TICKERS_TO_SCORE}")
    print(target_df.groupby("ticker")["sentiment_score"].describe().round(2))

    return final_df


def save_progress(existing_df, new_results, path):
    new_df = pd.DataFrame(new_results)

    if not existing_df.empty and not new_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    elif not new_df.empty:
        combined = new_df
    else:
        combined = existing_df

    combined.to_csv(path, index=False)
    return combined


if __name__ == "__main__":
    score_all_news()