"""
Step 4: Turn flagged trades into exception reports an analyst could
actually review.

A real surveillance analyst doesn't review individual trade rows one
at a time -- they review CASES: "this account pair round-tripped the
same size 8 times in 10 minutes," not eight separate unrelated alerts.
This groups flagged trades from detectors.py into one case per episode
and writes a templated narrative for each, citing only numbers that are
actually in the underlying trade data -- no free-text LLM generation,
so there's nothing here to hallucinate.
"""

import pandas as pd


def build_wash_trade_cases(result: pd.DataFrame) -> list[dict]:
    flagged = result[result["flag_wash_trade"]].copy()
    if flagged.empty:
        return []
    flagged["day"] = flagged["timestamp"].dt.date

    cases = []
    case_id = 1
    for (ticker, day), group in flagged.groupby(["ticker", "day"]):
        accounts = sorted(group["account_id"].unique())
        cases.append(
            {
                "case_id": f"WASH-{case_id:04d}",
                "pattern": "wash_trade",
                "ticker": ticker,
                "day": str(day),
                "accounts_involved": accounts,
                "trade_ids": sorted(group["trade_id"].tolist()),
                "total_volume": int(group["volume"].sum()),
                "narrative": (
                    f"Accounts {', '.join(accounts)} executed {len(group)} matched round-trip trades in "
                    f"{ticker} on {day}, totaling {int(group['volume'].sum())} shares, at prices within "
                    f"0.5% of each other and volumes within 10% of each other, with no net position change "
                    f"across the group -- consistent with wash trading."
                ),
            }
        )
        case_id += 1
    return cases


def build_marking_close_cases(result: pd.DataFrame) -> list[dict]:
    flagged = result[result["flag_marking_close"]].copy()
    if flagged.empty:
        return []
    flagged["day"] = flagged["timestamp"].dt.date

    cases = []
    case_id = 1
    for (ticker, day, account), group in flagged.groupby(["ticker", "day", "account_id"]):
        side = group["side"].mode().iloc[0]
        cases.append(
            {
                "case_id": f"CLOSE-{case_id:04d}",
                "pattern": "marking_close",
                "ticker": ticker,
                "day": str(day),
                "accounts_involved": [account],
                "trade_ids": sorted(group["trade_id"].tolist()),
                "total_volume": int(group["volume"].sum()),
                "narrative": (
                    f"Account {account} placed {len(group)} predominantly {side}-side trades in {ticker} "
                    f"within the last {10} minutes of the session on {day}, totaling "
                    f"{int(group['volume'].sum())} shares -- a volume and directional concentration "
                    f"consistent with an attempt to influence the closing price."
                ),
            }
        )
        case_id += 1
    return cases


def build_volume_anomaly_cases(result: pd.DataFrame) -> list[dict]:
    flagged = result[result["flag_volume_anomaly"]].copy()
    if flagged.empty:
        return []
    flagged["day"] = flagged["timestamp"].dt.date

    cases = []
    case_id = 1
    for (ticker, day), group in flagged.groupby(["ticker", "day"]):
        accounts = sorted(group["account_id"].unique())
        cases.append(
            {
                "case_id": f"ANOM-{case_id:04d}",
                "pattern": "volume_anomaly",
                "ticker": ticker,
                "day": str(day),
                "accounts_involved": accounts,
                "trade_ids": sorted(group["trade_id"].tolist()),
                "total_volume": int(group["volume"].sum()),
                "narrative": (
                    f"{ticker}'s trading volume in the final hour of the session on {day} was a statistical "
                    f"outlier (Z-score >= 3) against its own recent history, involving {len(accounts)} "
                    f"account(s) and {int(group['volume'].sum())} shares -- flagged for review; note this "
                    f"detector has known low precision (see README) and every case here needs manual "
                    f"triage, not automatic escalation."
                ),
            }
        )
        case_id += 1
    return cases


def build_all_cases(result: pd.DataFrame) -> list[dict]:
    return (
        build_wash_trade_cases(result)
        + build_marking_close_cases(result)
        + build_volume_anomaly_cases(result)
    )


def write_markdown_report(cases: list[dict], path: str) -> None:
    lines = [f"# Surveillance Exception Report\n\n{len(cases)} case(s) flagged for review.\n"]
    for case in cases:
        lines.append(f"## {case['case_id']} -- {case['pattern']}\n")
        lines.append(f"- **Ticker:** {case['ticker']}")
        lines.append(f"- **Date:** {case['day']}")
        lines.append(f"- **Account(s):** {', '.join(case['accounts_involved'])}")
        lines.append(f"- **Trade IDs:** {case['trade_ids']}")
        lines.append(f"- **Total volume:** {case['total_volume']:,} shares")
        lines.append(f"- **Narrative:** {case['narrative']}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import json

    result = pd.read_csv("../data/detection_results.csv", parse_dates=["timestamp"])
    cases = build_all_cases(result)

    with open("../reports/exception_cases.json", "w") as f:
        json.dump(cases, f, indent=2, default=str)

    write_markdown_report(cases, "../reports/exception_report.md")

    print(f"Generated {len(cases)} exception case(s):")
    by_pattern = {}
    for c in cases:
        by_pattern[c["pattern"]] = by_pattern.get(c["pattern"], 0) + 1
    for pattern, count in by_pattern.items():
        print(f"  {pattern}: {count}")
    print("\nSaved reports/exception_cases.json and reports/exception_report.md")
