import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

TICKERS = ["COIN"]


def fetch_price_data(ticker, months=24):
    """Fetch OHLCV data for a ticker and calculate technical indicators."""
    print(f"  Fetching {ticker}...")

    df = yf.download(ticker, period=f"{months}mo", interval="1d", progress=False)

    if df.empty:
        print(f"  No data returned for {ticker}")
        return None

    # flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    })

    df["ticker"] = ticker
    df["date"]   = df.index.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)

    # ── Price returns ──────────────────────────────────────────
    df["return_1d"]  = df["close"].pct_change(1)
    df["return_5d"]  = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["return_10d"] = df["close"].pct_change(10)

    # ── Volume change ──────────────────────────────────────────
    df["volume_change"] = df["volume"].pct_change(1)

    # ── RSI (14 periods) ───────────────────────────────────────
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # ── MACD ──────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20 periods) ──────────────────────────
    rolling_mean      = df["close"].rolling(20).mean()
    rolling_std       = df["close"].rolling(20).std()
    df["bb_upper"]    = rolling_mean + (2 * rolling_std)
    df["bb_lower"]    = rolling_mean - (2 * rolling_std)
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / rolling_mean
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Target label ──────────────────────────────────────────
    # 1 if tomorrow's close is higher than today's, else 0
    df["label"] = (df["close"].shift(-5) > df["close"]).astype(int)

    # drop last row — it has no label (no tomorrow)
    df = df.iloc[:-1]

    # drop rows with NaN from indicator warmup period
    df = df.dropna().reset_index(drop=True)

    print(f"  {ticker}: {len(df)} trading days")
    return df


def compute_rsi(series, period=14):
    """Calculate RSI indicator."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_vix():
    """Fetch today's VIX value."""
    vix = yf.download("^VIX", period="5d", interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    return float(vix["Close"].dropna().iloc[-1])

def collect_all_prices(months=24):
    """Fetch and calculate price features for all tickers."""
    print(f"Collecting {months} months of price data\n")

    all_dfs = []

    for ticker in TICKERS:
        df = fetch_price_data(ticker, months=months)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No price data collected.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # keep only relevant columns
    cols = [
        "ticker", "date", "close",
        "return_1d", "return_5d", "return_20d",
        "volume_change",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_position",
        "label"
    ]
    combined = combined[cols]

    os.makedirs("data", exist_ok=True)
    combined.to_csv("data/price_data.csv", index=False)

    print(f"\nDone. {len(combined)} rows saved to data/price_data.csv")
    print(combined.groupby("ticker").size().rename("trading_days"))
    print("\nSample row:")
    print(combined.iloc[0])

    return combined


if __name__ == "__main__":
    collect_all_prices(months=24)