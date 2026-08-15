"""
Step 3: Measure real precision/recall against the injected ground truth.

Same discipline as the RAG project's eval_compliance.py: the detectors
never see ground_truth.csv, and this script is the only place the two
files ever get joined, purely to score the detectors after the fact.
Reports per-detector recall against ITS OWN target pattern (does
detect_wash_trades actually catch wash_trade-labeled rows?) as well as
overall precision across all detectors combined, since in a real
review queue an analyst sees "flagged," not which detector fired.
"""

import pandas as pd

from detectors import run_all_detectors

DETECTOR_TARGETS = {
    "flag_wash_trade": "wash_trade",
    "flag_marking_close": "marking_close",
    "flag_volume_anomaly": "pre_news_spike",
}


def evaluate(result: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    labels = dict(zip(ground_truth["trade_id"], ground_truth["abuse_type"]))
    result = result.copy()
    result["true_abuse_type"] = result["trade_id"].map(labels)
    result["is_abusive"] = result["true_abuse_type"].notna()

    report = {"per_detector": {}, "overall": {}}

    for flag_col, target_type in DETECTOR_TARGETS.items():
        target_rows = result[result["true_abuse_type"] == target_type]
        flagged_rows = result[result[flag_col]]

        tp = (flagged_rows["true_abuse_type"] == target_type).sum()
        fp = (flagged_rows["true_abuse_type"] != target_type).sum()  # includes both "not abusive" and "wrong type"
        fn = len(target_rows) - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        missed = target_rows[~target_rows["trade_id"].isin(flagged_rows["trade_id"])]

        report["per_detector"][flag_col] = {
            "target_pattern": target_type,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "missed_trade_ids": missed["trade_id"].tolist(),
        }

    flagged_any = result[result["flagged"]]
    tp_any = flagged_any["is_abusive"].sum()
    fp_any = len(flagged_any) - tp_any
    fn_any = result["is_abusive"].sum() - tp_any

    report["overall"] = {
        "total_trades": len(result),
        "total_flagged": len(flagged_any),
        "total_injected_abusive": int(result["is_abusive"].sum()),
        "true_positives": int(tp_any),
        "false_positives": int(fp_any),
        "false_negatives": int(fn_any),
        "precision": round(tp_any / len(flagged_any), 3) if len(flagged_any) else 0.0,
        "recall": round(tp_any / result["is_abusive"].sum(), 3) if result["is_abusive"].sum() else 0.0,
    }

    return report


def print_report(report: dict) -> None:
    print("=== Per-detector performance (against its own target pattern) ===\n")
    for flag_col, stats in report["per_detector"].items():
        print(f"{flag_col} -> target: {stats['target_pattern']}")
        print(f"  precision: {stats['precision']:.1%}   recall: {stats['recall']:.1%}")
        print(f"  TP={stats['true_positives']}  FP={stats['false_positives']}  FN={stats['false_negatives']}")
        if stats["missed_trade_ids"]:
            print(f"  missed trade_ids: {stats['missed_trade_ids']}")
        print()

    o = report["overall"]
    print("=== Overall (any detector flagging counts as a catch) ===\n")
    print(f"Total trades: {o['total_trades']}, injected abusive: {o['total_injected_abusive']}, flagged: {o['total_flagged']}")
    print(f"Precision: {o['precision']:.1%}   Recall: {o['recall']:.1%}")
    print(f"TP={o['true_positives']}  FP={o['false_positives']}  FN={o['false_negatives']}")


if __name__ == "__main__":
    trades = pd.read_csv("../data/trades.csv", parse_dates=["timestamp"])
    ground_truth = pd.read_csv("../data/ground_truth.csv")

    result = run_all_detectors(trades)
    report = evaluate(result, ground_truth)
    print_report(report)

    result.to_csv("../data/detection_results.csv", index=False)

    import json
    with open("../reports/eval_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved detection_results.csv and reports/eval_report.json")
