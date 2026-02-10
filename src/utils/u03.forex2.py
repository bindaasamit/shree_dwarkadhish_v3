import pandas as pd
import numpy as np
from datetime import timedelta

# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel("C:\\Users\\Amit\\Downloads\\Abhishek\\forex_input_2022_2026.xlsx")
df = df.sort_values("date_ist").reset_index(drop=True)

# -----------------------------
# Candle metrics
# -----------------------------
df["range"] = df["high"] - df["low"]
df["body"]  = (df["close"] - df["open"]).abs()

# -----------------------------
# Rolling stats
# -----------------------------
df["avg_range_20"] = df["range"].rolling(20).mean()

# 10-candle compression range
df["comp_high"] = df["high"].rolling(10).max()
df["comp_low"]  = df["low"].rolling(10).min()
df["comp_range"] = df["comp_high"] - df["comp_low"]

# Global compression threshold
compression_threshold = df["comp_range"].quantile(0.20)

# Compression flag
df["is_compressed"] = df["comp_range"] < compression_threshold

# -----------------------------
# Expansion conditions
# -----------------------------
df["bull_expansion"] = (
    (df["range"] > 1.5 * df["avg_range_20"]) &
    (df["close"] > df["comp_high"].shift(1)) &
    ((df["close"] - df["low"]) / df["range"] > 0.8)
)

df["bear_expansion"] = (
    (df["range"] > 1.5 * df["avg_range_20"]) &
    (df["close"] < df["comp_low"].shift(1)) &
    ((df["high"] - df["close"]) / df["range"] > 0.8)
)

# Only trade expansions after compression
df["long_signal"]  = df["bull_expansion"] & df["is_compressed"].shift(1)
df["short_signal"] = df["bear_expansion"] & df["is_compressed"].shift(1)

# -----------------------------
# Trade construction
# -----------------------------
df["direction"] = np.where(df["long_signal"], "LONG",
                   np.where(df["short_signal"], "SHORT", None))

df["entry"] = np.where(df["direction"].notna(), df["close"], np.nan)

df["stop"] = np.where(
    df["direction"] == "LONG", df["comp_low"].shift(1),
    np.where(df["direction"] == "SHORT", df["comp_high"].shift(1), np.nan)
)

df["risk"] = (df["entry"] - df["stop"]).abs()

# -----------------------------
# Outcome simulation (1-bar lookahead for loss control)
# -----------------------------
df["result"] = None

for i in range(len(df)-1):
    if df.loc[i, "direction"] == "LONG":
        if df.loc[i+1, "low"] <= df.loc[i, "stop"]:
            df.loc[i, "result"] = -1
        else:
            df.loc[i, "result"] = 1

    elif df.loc[i, "direction"] == "SHORT":
        if df.loc[i+1, "high"] >= df.loc[i, "stop"]:
            df.loc[i, "result"] = -1
        else:
            df.loc[i, "result"] = 1

# -----------------------------
# Final trade table
# -----------------------------
trades = df[df["direction"].notna()].copy()

summary = {
    "Total Trades": len(trades),
    "Win Rate": trades["result"].mean(),
    "Avg Risk": trades["risk"].mean()
}

summary, trades.head()


trades.to_excel('C:\\Users\\Amit\\Downloads\\Abhishek\\forex_output_2022_2026.xlsx', index=False)