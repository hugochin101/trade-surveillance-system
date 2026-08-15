"""
Step 1: Generate a labeled synthetic trade dataset.

Real trade data isn't public, so this builds a plausible stand-in: a
few weeks of ordinary trading across several tickers and accounts, with
three specific market-abuse patterns deliberately injected on top and
labeled. The label file is kept SEPARATE from the trade file on
purpose -- the detection engine (detectors.py) only ever sees
trades.csv, the same way a real surveillance system doesn't get told
the answer in advance. ground_truth.csv exists purely so evaluate.py
can measure real precision/recall afterward, mirroring how
eval_compliance.py in the RAG project never let the system see its own
answer key either.

Patterns injected:
  - wash_trade: a small ring of related accounts trades back and forth
    with matched size/price and no real net position change, in a
    tight time window -- the point is to create artificial volume/
    liquidity with no real economic transfer.
  - marking_close: one account concentrates unusually heavy,
    one-directional volume in the last few minutes of the trading
    session to push that day's closing price.
  - pre_news_spike: a small number of accounts trade an unusually large
    position shortly before a scheduled news event for that ticker --
    the classic shape of trading on material non-public information.

Everything is seeded (RANDOM_SEED) so the dataset -- and therefore the
detection results in evaluate.py -- is reproducible.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "XOM", "NFLX"]
N_NORMAL_ACCOUNTS = 60
N_TRADING_DAYS = 25
MARKET_OPEN_MINUTE = 0
MARKET_CLOSE_MINUTE = 390  # 9:30-16:00 as minutes-from-open

rng = np.random.default_rng(RANDOM_SEED)


def _trading_days(n_days: int) -> list[pd.Timestamp]:
    days = pd.bdate_range(end=pd.Timestamp("2026-06-30"), periods=n_days)
    return list(days)


def _day_start_prices() -> dict[str, float]:
    return {t: float(p) for t, p in zip(TICKERS, rng.uniform(50, 400, size=len(TICKERS)))}


def generate_baseline_trades(days: list[pd.Timestamp]) -> list[dict]:
    """Ordinary trading: random accounts, random minute, small random
    walk in price per ticker per day, log-normal volume."""
    trades = []
    trade_id = 0
    account_ids = [f"ACC{i:04d}" for i in range(N_NORMAL_ACCOUNTS)]

    prices = _day_start_prices()
    for day in days:
        for ticker in TICKERS:
            n_trades_today = int(rng.poisson(120))
            minutes = np.sort(rng.integers(MARKET_OPEN_MINUTE, MARKET_CLOSE_MINUTE, size=n_trades_today))
            price = prices[ticker]
            for minute in minutes:
                price *= 1 + rng.normal(0, 0.0015)
                price = max(price, 1.0)
                volume = int(rng.lognormal(mean=5.5, sigma=0.8))
                account = account_ids[rng.integers(0, len(account_ids))]
                side = "BUY" if rng.random() < 0.5 else "SELL"
                trades.append(
                    {
                        "trade_id": trade_id,
                        "timestamp": day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=int(minute)),
                        "ticker": ticker,
                        "account_id": account,
                        "side": side,
                        "price": round(price, 2),
                        "volume": max(volume, 1),
                    }
                )
                trade_id += 1
            prices[ticker] = price
    return trades


def inject_wash_trades(trades: list[dict], next_trade_id: int, days: list[pd.Timestamp]) -> tuple[list[dict], dict]:
    """A ring of 3 related accounts trades back and forth in a tight
    window with matched volume and near-identical prices -- artificial
    volume, no real change of beneficial ownership."""
    injected = []
    labels = {}
    ring_accounts = ["ACC9001", "ACC9002", "ACC9003"]

    for i, day in enumerate(days[::5]):  # every 5th day, a new ring episode
        ticker = TICKERS[i % len(TICKERS)]
        base_minute = int(rng.integers(60, 300))
        base_price = round(float(rng.uniform(50, 400)), 2)
        n_round_trips = int(rng.integers(6, 12))

        for j in range(n_round_trips):
            minute = base_minute + j * 2  # every 2 minutes -- tight window
            price = base_price + rng.normal(0, 0.02)  # near-identical price
            volume = 500  # matched size every time -- the tell
            buyer, seller = rng.choice(ring_accounts, size=2, replace=False)

            trade_id = next_trade_id
            next_trade_id += 1
            record = {
                "trade_id": trade_id,
                "timestamp": day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute),
                "ticker": ticker,
                "account_id": buyer,
                "side": "BUY",
                "price": round(price, 2),
                "volume": volume,
            }
            injected.append(record)
            labels[trade_id] = "wash_trade"

            trade_id = next_trade_id
            next_trade_id += 1
            record = {
                "trade_id": trade_id,
                "timestamp": day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute, seconds=30),
                "ticker": ticker,
                "account_id": seller,
                "side": "SELL",
                "price": round(price, 2),
                "volume": volume,
            }
            injected.append(record)
            labels[trade_id] = "wash_trade"

    return injected, labels


def inject_marking_the_close(trades: list[dict], next_trade_id: int, days: list[pd.Timestamp]) -> tuple[list[dict], dict]:
    """One account dumps unusually heavy, one-directional volume into
    the last few minutes of the session to move the closing price."""
    injected = []
    labels = {}
    marker_account = "ACC9004"

    for i, day in enumerate(days[2::6]):
        ticker = TICKERS[(i + 2) % len(TICKERS)]
        base_price = round(float(rng.uniform(50, 400)), 2)
        direction = "BUY" if rng.random() < 0.5 else "SELL"
        n_trades = int(rng.integers(8, 14))

        for j in range(n_trades):
            minute = MARKET_CLOSE_MINUTE - int(rng.integers(1, 6))  # last few minutes only
            price = base_price * (1 + (0.002 * j if direction == "BUY" else -0.002 * j))
            volume = int(rng.integers(2000, 5000))  # much larger than a normal trade

            trade_id = next_trade_id
            next_trade_id += 1
            injected.append(
                {
                    "trade_id": trade_id,
                    "timestamp": day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute),
                    "ticker": ticker,
                    "account_id": marker_account,
                    "side": direction,
                    "price": round(price, 2),
                    "volume": volume,
                }
            )
            labels[trade_id] = "marking_close"

    return injected, labels


def inject_pre_news_spike(trades: list[dict], next_trade_id: int, days: list[pd.Timestamp]) -> tuple[list[dict], dict, list[dict]]:
    """A small handful of accounts trade an unusually large position
    shortly before a scheduled news event for that ticker."""
    injected = []
    labels = {}
    news_events = []
    tippee_accounts = ["ACC9005", "ACC9006"]

    for i, day in enumerate(days[3::7]):
        if i + 1 >= len(days[3::7]) + 1:
            continue
        ticker = TICKERS[(i + 4) % len(TICKERS)]
        # the "news" lands after this trading day's close
        event_time = day + pd.Timedelta(hours=16, minutes=30)
        news_events.append({"ticker": ticker, "event_timestamp": event_time, "day": day})

        base_price = round(float(rng.uniform(50, 400)), 2)
        n_trades = int(rng.integers(4, 8))

        for j in range(n_trades):
            minute = MARKET_CLOSE_MINUTE - int(rng.integers(10, 60))  # last hour before close, day of the news
            price = base_price * (1 + 0.001 * j)
            volume = int(rng.integers(1500, 4000))
            account = tippee_accounts[rng.integers(0, len(tippee_accounts))]

            trade_id = next_trade_id
            next_trade_id += 1
            injected.append(
                {
                    "trade_id": trade_id,
                    "timestamp": day + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute),
                    "ticker": ticker,
                    "account_id": account,
                    "side": "BUY",
                    "price": round(price, 2),
                    "volume": volume,
                }
            )
            labels[trade_id] = "pre_news_spike"

    return injected, labels, news_events


def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = _trading_days(N_TRADING_DAYS)
    baseline = generate_baseline_trades(days)
    next_id = len(baseline)

    wash, wash_labels = inject_wash_trades(baseline, next_id, days)
    next_id += len(wash)

    marking, marking_labels = inject_marking_the_close(baseline, next_id, days)
    next_id += len(marking)

    spike, spike_labels, news_events = inject_pre_news_spike(baseline, next_id, days)
    next_id += len(spike)

    all_trades = baseline + wash + marking + spike
    trades_df = pd.DataFrame(all_trades).sort_values("timestamp").reset_index(drop=True)

    all_labels = {}
    all_labels.update(wash_labels)
    all_labels.update(marking_labels)
    all_labels.update(spike_labels)
    ground_truth_df = pd.DataFrame(
        [{"trade_id": tid, "abuse_type": label} for tid, label in all_labels.items()]
    )

    news_df = pd.DataFrame(news_events)

    return trades_df, ground_truth_df, news_df


if __name__ == "__main__":
    trades_df, ground_truth_df, news_df = generate_dataset()

    trades_df.to_csv("../data/trades.csv", index=False)
    ground_truth_df.to_csv("../data/ground_truth.csv", index=False)
    news_df.to_csv("../data/news_events.csv", index=False)

    print(f"Generated {len(trades_df)} trades across {trades_df['ticker'].nunique()} tickers "
          f"over {trades_df['timestamp'].dt.date.nunique()} trading days")
    print(f"Injected {len(ground_truth_df)} abusive trades: "
          f"{ground_truth_df['abuse_type'].value_counts().to_dict()}")
    print(f"Scheduled {len(news_df)} news events")
