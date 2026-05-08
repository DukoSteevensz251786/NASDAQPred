import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier

TICKERS = ["COIN"]

FEATURES = [
    "return_1d", "return_5d",  "return_20d",
    "volume_change",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "sentiment_score", "article_count",
    "has_news", "vix"
]

def train_ticker(ticker, df):
    print(f"\n{'='*50}")
    print(f"Training model for {ticker}")
    print(f"{'='*50}")

    # filter to this ticker and sort by date
    data = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    print(f"Total rows: {len(data)}")

    X = data[FEATURES]
    y = data["label"]

    # chronological 80/20 split — never shuffle time series
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows ({data['date'].iloc[0]} → {data['date'].iloc[split_idx-1]})")
    print(f"Test:  {len(X_test)} rows ({data['date'].iloc[split_idx]} → {data['date'].iloc[-1]})")

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    # train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # evaluate
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    baseline  = y_test.mean()  # always predict UP baseline

    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.2%}  (baseline: {baseline:.2%})")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")

    # feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)
    print(f"\nTop 5 features:")
    for feat, score in importance.head(5).items():
        print(f"  {feat:<20} {score:.4f}")

    return model, {
        "ticker":    ticker,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "baseline":  baseline,
    }


def train_all():
    df = pd.read_csv("data/training_data.csv")
    print(f"Loaded {len(df)} rows\n")

    os.makedirs("models", exist_ok=True)

    all_results = []

    for ticker in TICKERS:
        model, results = train_ticker(ticker, df)
        all_results.append(results)

        # save model
        model_path = f"models/{ticker}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {model_path}")

    # summary table
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index("ticker")
    results_df = results_df.map(lambda x: f"{x:.2%}")
    print(results_df.to_string())


if __name__ == "__main__":
    train_all()