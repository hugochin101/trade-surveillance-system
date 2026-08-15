"""
Unit tests for detectors.py. Each test builds a small, hand-crafted
trade DataFrame -- not the full generated dataset -- so a failure
points at exactly which rule broke, not "something in 24k rows".
"""

import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors import detect_wash_trades, detect_marking_the_close, detect_volume_anomalies


def make_trade(trade_id, minute, ticker, account, side, price, volume, day="2026-06-01"):
    return {
        "trade_id": trade_id,
        "timestamp": pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute),
        "ticker": ticker,
        "account_id": account,
        "side": side,
        "price": price,
        "volume": volume,
    }


# --- wash trades ---

def test_detects_matched_round_trip_wash_trades():
    rows = []
    for i in range(4):
        rows.append(make_trade(i * 2, 10 + i * 2, "AAPL", "ACC_A", "BUY", 100.0, 500))
        rows.append(make_trade(i * 2 + 1, 10 + i * 2, "AAPL", "ACC_B", "SELL", 100.01, 500))
    trades = pd.DataFrame(rows)

    flagged = detect_wash_trades(trades)

    assert flagged == set(range(8))


def test_does_not_flag_ordinary_scattered_trading():
    rows = [
        make_trade(0, 5, "AAPL", "ACC_A", "BUY", 100.0, 300),
        make_trade(1, 90, "AAPL", "ACC_B", "SELL", 101.5, 450),
        make_trade(2, 200, "AAPL", "ACC_C", "BUY", 99.8, 220),
        make_trade(3, 300, "AAPL", "ACC_D", "SELL", 102.0, 610),
    ]
    trades = pd.DataFrame(rows)

    flagged = detect_wash_trades(trades)

    assert flagged == set()


def test_wash_trade_requires_the_price_to_actually_match():
    """Same accounts, same tight window, matched volume -- but prices
    are 5% apart, which is real price discovery, not a wash trade."""
    rows = []
    for i in range(5):
        rows.append(make_trade(i * 2, 10 + i * 2, "AAPL", "ACC_A", "BUY", 100.0, 500))
        rows.append(make_trade(i * 2 + 1, 10 + i * 2, "AAPL", "ACC_B", "SELL", 105.0, 500))
    trades = pd.DataFrame(rows)

    flagged = detect_wash_trades(trades)

    assert flagged == set()


def test_wash_trade_needs_the_minimum_repeat_count():
    """Two round trips isn't enough repetition to distinguish from
    coincidence -- WASH_MIN_ROUND_TRIPS is 3. The two round trips are
    placed far enough apart (minute 10 vs. minute 300) that they can't
    cross-match each other within the 5-minute window and inflate the
    count -- that cross-matching is itself correct detector behavior,
    caught by test_detects_matched_round_trip_wash_trades separately."""
    rows = [
        make_trade(0, 10, "AAPL", "ACC_A", "BUY", 100.0, 500),
        make_trade(1, 10, "AAPL", "ACC_B", "SELL", 100.0, 500),
        make_trade(2, 300, "AAPL", "ACC_A", "BUY", 100.0, 500),
        make_trade(3, 300, "AAPL", "ACC_B", "SELL", 100.0, 500),
    ]
    trades = pd.DataFrame(rows)

    flagged = detect_wash_trades(trades)

    assert flagged == set()


# --- marking the close ---

def test_detects_one_sided_volume_concentrated_at_close():
    rows = [make_trade(i, 385 + 0, "AAPL", "ACC_X", "BUY", 100.0 + i * 0.1, 3000) for i in range(5)]
    # a little ordinary background trading earlier in the day
    rows += [make_trade(100 + i, 50 * i, "AAPL", f"ACC_{i}", "SELL" if i % 2 else "BUY", 100.0, 200) for i in range(6)]
    trades = pd.DataFrame(rows)

    flagged = detect_marking_the_close(trades)

    assert flagged == set(range(5))


def test_does_not_flag_ordinary_closing_activity():
    """Several different accounts trading both directions near the
    close -- normal end-of-day liquidity, not one account marking it."""
    rows = [
        make_trade(0, 385, "AAPL", "ACC_A", "BUY", 100.0, 200),
        make_trade(1, 386, "AAPL", "ACC_B", "SELL", 100.1, 210),
        make_trade(2, 387, "AAPL", "ACC_C", "BUY", 100.0, 190),
        make_trade(3, 388, "AAPL", "ACC_D", "SELL", 99.9, 205),
    ]
    rows += [make_trade(100 + i, 50 * i, "AAPL", f"ACC_{i}", "BUY", 100.0, 300) for i in range(6)]
    trades = pd.DataFrame(rows)

    flagged = detect_marking_the_close(trades)

    assert flagged == set()


# --- volume anomalies ---

def test_detects_a_last_hour_volume_spike_against_history():
    rows = []
    trade_id = 0
    # 10 quiet days of history: small last-hour volume with a little
    # natural variance (never literally identical -- zero-variance
    # history is a degenerate case the detector deliberately skips
    # rather than divide by a zero std, see test below).
    quiet_volumes = [190, 205, 195, 210, 200, 198, 202, 192, 208, 200]
    for day_offset, vol in enumerate(quiet_volumes):
        day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
        rows.append(make_trade(trade_id, 385, "AAPL", "ACC_A", "BUY", 100.0, vol, day=str(day.date())))
        trade_id += 1
    # day 11: a genuine spike in the last hour
    spike_day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=10)
    for i in range(5):
        rows.append(make_trade(trade_id, 385 + i, "AAPL", "ACC_B", "BUY", 100.0, 5000, day=str(spike_day.date())))
        trade_id += 1
    trades = pd.DataFrame(rows)

    flagged = detect_volume_anomalies(trades)

    spike_ids = set(range(10, 15))
    assert spike_ids.issubset(flagged)
    assert flagged.isdisjoint(set(range(10)))  # none of the quiet history days got flagged


def test_does_not_flag_before_enough_history_exists():
    """Only 3 days of history -- the detector requires 5 before it will
    judge anything an outlier, so even an extreme day should pass through
    unflagged rather than false-alarm on a tiny sample."""
    rows = []
    trade_id = 0
    for day_offset in range(3):
        day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=day_offset)
        rows.append(make_trade(trade_id, 385, "AAPL", "ACC_A", "BUY", 100.0, 200, day=str(day.date())))
        trade_id += 1
    huge_day = pd.Timestamp("2026-06-01") + pd.Timedelta(days=3)
    rows.append(make_trade(trade_id, 385, "AAPL", "ACC_B", "BUY", 100.0, 50000, day=str(huge_day.date())))
    trades = pd.DataFrame(rows)

    flagged = detect_volume_anomalies(trades)

    assert flagged == set()
