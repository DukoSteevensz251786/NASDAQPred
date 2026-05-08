import pandas as pd
import pickle
import numpy as np
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "volume_change",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "sentiment_score", "article_count",
    "has_news", "vix"
]


def backtest_ticker(ticker, df):
    # get test set (last 20%)
    data = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    split = int(len(data) * 0.8)
    test = data.iloc[split:].copy()

    # load model
    model_path = Path(f"models/{ticker}_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = test[FEATURES]
    y_test = test["label"]

    # predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    test = test.copy()
    test["predicted"] = y_pred
    test["confidence"] = y_prob.max(axis=1)
    test["correct"] = (y_pred == y_test).astype(int)

    # overall accuracy
    accuracy = test["correct"].mean()
    baseline = y_test.mean()  # always predict UP

    # accuracy on high confidence predictions only (>60%, >70%, >80%)
    results = {"ticker": ticker, "accuracy": accuracy, "baseline": baseline, "total_days": len(test)}

    for threshold in [0.60, 0.70, 0.80]:
        high_conf = test[test["confidence"] >= threshold]
        if len(high_conf) > 0:
            results[f"acc_{int(threshold*100)}pct_conf"] = high_conf["correct"].mean()
            results[f"days_{int(threshold*100)}pct_conf"] = len(high_conf)
        else:
            results[f"acc_{int(threshold*100)}pct_conf"] = None
            results[f"days_{int(threshold*100)}pct_conf"] = 0

    # month by month breakdown
    test["month"] = pd.to_datetime(test["date"]).dt.to_period("M")
    monthly = test.groupby("month")["correct"].mean()

    return results, test, monthly


def run_backtest():
    df = pd.read_csv("data/training_data.csv")

    # add has_news if missing
    if "has_news" not in df.columns:
        df["has_news"] = (df["article_count"] > 0).astype(int)

    all_results = []
    all_monthly = {}

    print("=" * 60)
    print("BACKTEST RESULTS — Dec 2025 to Apr 2026")
    print("=" * 60)

    for ticker in TICKERS:
        results, test_df, monthly = backtest_ticker(ticker, df)
        all_results.append(results)
        all_monthly[ticker] = monthly

        print(f"\n{ticker}")
        print(f"  Overall accuracy:     {results['accuracy']:.1%}  (baseline: {results['baseline']:.1%})")
        print(f"  Beat baseline:        {'✅ Yes' if results['accuracy'] > results['baseline'] else '❌ No'}")
        print(f"  Total days tested:    {results['total_days']}")
        print(f"  High confidence (>60%): {results['acc_60pct_conf']:.1%} on {results['days_60pct_conf']} days" if results['days_60pct_conf'] > 0 else "  High confidence (>60%): no predictions")
        print(f"  High confidence (>70%): {results['acc_70pct_conf']:.1%} on {results['days_70pct_conf']} days" if results['days_70pct_conf'] > 0 else "  High confidence (>70%): no predictions")
        print(f"  High confidence (>80%): {results['acc_80pct_conf']:.1%} on {results['days_80pct_conf']} days" if results['days_80pct_conf'] > 0 else "  High confidence (>80%): no predictions")

        print(f"\n  Monthly breakdown:")
        for month, acc in monthly.items():
            bar = "█" * int(acc * 20)
            print(f"    {month}  {acc:.1%}  {bar}")

    # summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(all_results).set_index("ticker")
    print(f"{'Ticker':<8} {'Accuracy':<12} {'Baseline':<12} {'Beat?':<8} {'>60% conf':<12} {'>80% conf'}")
    print("-" * 60)
    for ticker, row in results_df.iterrows():
        beat = "✅" if row["accuracy"] > row["baseline"] else "❌"
        acc_60 = f"{row['acc_60pct_conf']:.1%}" if row["days_60pct_conf"] > 0 else "n/a"
        acc_80 = f"{row['acc_80pct_conf']:.1%}" if row["days_80pct_conf"] > 0 else "n/a"
        print(f"{ticker:<8} {row['accuracy']:.1%}{'':>6} {row['baseline']:.1%}{'':>6} {beat}{'':>4} {acc_60:<12} {acc_80}")

    # save detailed results
    pd.DataFrame(all_results).to_csv("data/backtest_results.csv", index=False)
    print(f"\nDetailed results saved to data/backtest_results.csv")


if __name__ == "__main__":
    run_backtest()