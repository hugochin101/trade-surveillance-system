"""
Step 2: Detection engine.

Three detectors, each targeting one abuse pattern from data_generator.py.
None of them are allowed to know which account IDs were used to inject
the abuse (that would make this a lookup table, not a detector) --
every flag has to come from a genuine behavioral signal: matched
round-trip trading, one-sided volume concentrated at the close, or an
unusual volume spike relative to a ticker's own history. That's the
whole point of measuring precision/recall in evaluate.py instead of
just eyeballing a few flagged rows.
"""

import numpy as np
import pandas as pd

# --- Wash trading ---------------------------------------------------

WASH_TIME_WINDOW = pd.Timedelta(minutes=5)
WASH_PRICE_TOLERANCE = 0.005  # 0.5%
WASH_VOLUME_TOLERANCE = 0.10  # 10%
WASH_MIN_ROUND_TRIPS = 3  # same account pair, same day: flag at this many matches


def detect_wash_trades(trades: pd.DataFrame) -> set[int]:
    """Flags trades that are part of a repeated buy/sell round-trip
    between the same two accounts, at matched size and near-identical
    price, within a tight time window -- the signature of artificial
    volume with no real change of ownership."""
    flagged: set[int] = set()
    trades = trades.copy()
    trades["day"] = trades["timestamp"].dt.date

    for (ticker, day), group in trades.groupby(["ticker", "day"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        buys = group[group["side"] == "BUY"]
        sells = group[group["side"] == "SELL"]
        if buys.empty or sells.empty:
            continue

        # every buy vs every sell in this (ticker, day) -- small groups
        # (~100-150 trades/day/ticker), so this stays fast.
        pair_matches: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for b in buys.itertuples():
            for s in sells.itertuples():
                if b.account_id == s.account_id:
                    continue
                time_gap = abs(b.timestamp - s.timestamp)
                if time_gap > WASH_TIME_WINDOW:
                    continue
                price_gap = abs(b.price - s.price) / max(b.price, s.price)
                if price_gap > WASH_PRICE_TOLERANCE:
                    continue
                volume_gap = abs(b.volume - s.volume) / max(b.volume, s.volume)
                if volume_gap > WASH_VOLUME_TOLERANCE:
                    continue

                key = tuple(sorted((b.account_id, s.account_id)))
                pair_matches.setdefault(key, []).append((b.trade_id, s.trade_id))

        for _pair, matches in pair_matches.items():
            if len(matches) >= WASH_MIN_ROUND_TRIPS:
                for buy_id, sell_id in matches:
                    flagged.add(buy_id)
                    flagged.add(sell_id)

    return flagged


# --- Marking the close ----------------------------------------------

CLOSE_WINDOW_MINUTES = 10
CLOSE_VOLUME_SHARE_THRESHOLD = 0.15  # one account's share of the day's total volume in that window
CLOSE_DIRECTIONALITY_THRESHOLD = 0.80  # how one-sided that account's own trading is in the window


def detect_marking_the_close(trades: pd.DataFrame) -> set[int]:
    """Flags an account whose trading in the last few minutes of the
    session is both unusually large relative to that ticker's whole-day
    volume AND overwhelmingly one-directional -- ordinary closing
    activity is neither this concentrated nor this one-sided."""
    flagged: set[int] = set()
    trades = trades.copy()
    trades["day"] = trades["timestamp"].dt.date

    for (ticker, day), group in trades.groupby(["ticker", "day"]):
        day_total_volume = group["volume"].sum()
        if day_total_volume == 0:
            continue

        session_end = group["timestamp"].max()
        window_start = session_end - pd.Timedelta(minutes=CLOSE_WINDOW_MINUTES)
        closing_window = group[group["timestamp"] > window_start]

        for account_id, acc_group in closing_window.groupby("account_id"):
            account_close_volume = acc_group["volume"].sum()
            share_of_day = account_close_volume / day_total_volume
            if share_of_day < CLOSE_VOLUME_SHARE_THRESHOLD:
                continue

            side_counts = acc_group["side"].value_counts(normalize=True)
            directionality = side_counts.max()
            if directionality < CLOSE_DIRECTIONALITY_THRESHOLD:
                continue

            flagged.update(acc_group["trade_id"].tolist())

    return flagged


# --- Statistical volume/price anomaly (catches pre-news-style spikes) ---

ANOMALY_LOOKBACK_DAYS = 15
ANOMALY_Z_THRESHOLD = 3.0
ANOMALY_WINDOW = pd.Timedelta(hours=1)  # "last hour before close" -- must match the window scored below


def detect_volume_anomalies(trades: pd.DataFrame) -> set[int]:
    """Z-score anomaly detection, but on each ticker's LAST-HOUR-BEFORE-
    CLOSE volume specifically, compared against its own history of that
    same window -- not on whole-day totals.

    FIRST ATTEMPT (kept here as the honest record, see README): scored
    Z-scores on whole-day total volume. That measured 3.3% precision --
    daily totals in this dataset have up to ~24% natural
    coefficient-of-variation from ordinary Poisson trade-count and
    log-normal per-trade-size randomness alone, so plenty of
    perfectly ordinary days cross a Z>=3 threshold by chance, and
    every one of those false alarms drags a whole hour of innocent
    trades into the flagged set. The injected spike itself only ever
    lives in the last hour, so scoring the whole day was diluting the
    one signal that actually mattered with the rest of the day's noise.

    FIX: compute each day's last-hour volume, and Z-score that specific
    number against the ticker's own history of ITS last-hour volumes.
    Same technique as the World Cup project's exception report (Z-score
    vs. a rolling baseline), just localized to where the real signal is."""
    flagged: set[int] = set()
    trades = trades.copy()
    trades["day"] = trades["timestamp"].dt.date

    last_hour_volumes = []
    for (ticker, day), group in trades.groupby(["ticker", "day"]):
        session_end = group["timestamp"].max()
        last_hour = group[group["timestamp"] > session_end - ANOMALY_WINDOW]
        last_hour_volumes.append({"ticker": ticker, "day": day, "last_hour_volume": last_hour["volume"].sum()})

    last_hour_df = pd.DataFrame(last_hour_volumes).sort_values(["ticker", "day"])

    for ticker, ticker_group in last_hour_df.groupby("ticker"):
        ticker_group = ticker_group.reset_index(drop=True)
        for i in range(len(ticker_group)):
            lookback = ticker_group.iloc[max(0, i - ANOMALY_LOOKBACK_DAYS): i]
            if len(lookback) < 5:  # not enough history yet to judge an outlier
                continue
            mean = lookback["last_hour_volume"].mean()
            std = lookback["last_hour_volume"].std()
            if std == 0:
                continue
            z = (ticker_group.iloc[i]["last_hour_volume"] - mean) / std
            if z >= ANOMALY_Z_THRESHOLD:
                day = ticker_group.iloc[i]["day"]
                day_trades = trades[(trades["ticker"] == ticker) & (trades["day"] == day)]
                session_end = day_trades["timestamp"].max()
                last_hour = day_trades[day_trades["timestamp"] > session_end - ANOMALY_WINDOW]
                flagged.update(last_hour["trade_id"].tolist())

    return flagged


def run_all_detectors(trades: pd.DataFrame) -> pd.DataFrame:
    """Runs every detector and returns trades.csv with one boolean
    column added per detector plus a combined `flagged` column."""
    wash = detect_wash_trades(trades)
    marking = detect_marking_the_close(trades)
    anomaly = detect_volume_anomalies(trades)

    result = trades.copy()
    result["flag_wash_trade"] = result["trade_id"].isin(wash)
    result["flag_marking_close"] = result["trade_id"].isin(marking)
    result["flag_volume_anomaly"] = result["trade_id"].isin(anomaly)
    result["flagged"] = result["flag_wash_trade"] | result["flag_marking_close"] | result["flag_volume_anomaly"]
    return result


if __name__ == "__main__":
    trades = pd.read_csv("../data/trades.csv", parse_dates=["timestamp"])
    result = run_all_detectors(trades)

    print(f"Total trades: {len(result)}")
    print(f"Flagged (any detector): {result['flagged'].sum()}")
    print(f"  wash_trade:      {result['flag_wash_trade'].sum()}")
    print(f"  marking_close:   {result['flag_marking_close'].sum()}")
    print(f"  volume_anomaly:  {result['flag_volume_anomaly'].sum()}")
