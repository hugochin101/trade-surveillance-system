# Trade Surveillance & Market Abuse Detection System

A trade surveillance pipeline that generates a labeled synthetic trading dataset, detects three real market-abuse patterns against it, and measures actual precision/recall rather than eyeballing a handful of flagged rows — built to demonstrate the same evaluation discipline a real compliance surveillance function needs, not just a working demo.

## Why this project

Real trade data isn't public, so this builds its own: several weeks of ordinary trading across 8 tickers and 60 accounts, with three specific abuse patterns deliberately injected on top and labeled with ground truth. The detection engine never sees the labels — they exist purely so `evaluate.py` can score the detectors afterward, the same separation between "what the system sees" and "the answer key" used to evaluate a RAG retrieval system.

## Patterns detected

- **Wash trading** — a small ring of related accounts trades back and forth with matched size, near-identical price, and no real net position change, in a tight time window. Detected by matching opposite-side trades within a 5-minute window, 0.5% price tolerance, and 10% volume tolerance, then flagging any account pair that repeats this 3+ times in a day.
- **Marking the close** — one account concentrates unusually heavy, one-directional volume into the last 10 minutes of the session to influence the closing price. Detected by flagging any account whose closing-window volume is both a large share (15%+) of that ticker's whole-day volume AND heavily one-sided (80%+ same direction).
- **Volume anomalies (a proxy for trading ahead of material news)** — statistical outlier detection (Z-score ≥ 3) on each ticker's own last-hour-before-close volume, compared against its recent history of that same window.

## Real results (not cherry-picked)

Run against 23,947 trades with 142 injected abusive trades:

| Detector | Precision | Recall |
|---|---|---|
| Wash trade | 100% | 98.7% |
| Marking the close | 100% | 100% |
| Volume anomaly | 8.4% | 77.3% |

**Overall (any detector firing counts as a catch): 47.1% precision, 95.8% recall.**

The first two detectors are strong. The third is reported honestly because it's genuinely weak, and the weakness is understood, not hand-waved:

### Root cause: the volume anomaly detector

**First attempt** scored Z-scores on each ticker's *whole-day* total volume. That measured **3.3% precision** — daily totals in this dataset carry up to ~24% natural coefficient-of-variation purely from ordinary Poisson trade-count and log-normal per-trade-size randomness, so plenty of perfectly innocent days cross a Z≥3 threshold by chance, and each false alarm drags a full hour of innocent trades into the flagged set with it.

**Fix applied**: localize the comparison to the last-hour-before-close volume specifically (where the injected signal actually lives) instead of the whole day. This moved recall from 22.7% → 77.3% — a real improvement, not a rounding change — but precision only moved from 3.3% → 8.4%. Last-hour volume alone is still a noisy univariate signal against naturally volatile baseline trading.

**What this means, and what I didn't do about it**: I stopped tuning here rather than keep adjusting thresholds until the number looked better on this one synthetic dataset — that would be curve-fitting to a fixed random seed, not building a better detector. A real fix needs a second, independent signal combined with volume (e.g., cross-referencing price direction, or the number of distinct accounts involved) rather than a single univariate Z-score, and that's the honest next step, not something claimed as already done.

## What's in the exception reports

Flagged trades are grouped into cases (one case per episode, not one alert per trade row) with a templated narrative that only ever cites numbers pulled directly from the underlying trade data — no free-text LLM generation here, so there's nothing to fabricate. See `reports/exception_report.md` after running the pipeline.

## Known limitations

- All data is synthetic. Real market microstructure (order books, cancellations, multi-venue trading) isn't modeled — this only sees executed trades, not orders, so genuine spoofing/layering (which lives in the cancel behavior, not the trade tape) isn't actually detected here despite being a common real-world pattern.
- The volume anomaly detector's precision (8.4%) is not production-grade; see root-cause section above.
- Thresholds (price/volume tolerance, Z-score cutoff, minimum repeat count) were picked to be reasonable, not calibrated against a real regulatory standard — a production system would tune these against actual historical alert/investigation outcomes.
- The dataset only has 25 trading days and 4 injected news-adjacent events; the first event has too little lookback history to be judged by the anomaly detector at all (needs 5+ prior days), which is a real, understood limitation of a small demo dataset, not a bug.

## Project structure

```
src/
  data_generator.py   # generates trades.csv, ground_truth.csv, news_events.csv
  detectors.py         # the three detection rules
  evaluate.py           # scores detectors against ground truth (precision/recall)
  reports.py            # groups flagged trades into reviewable exception cases
tests/
  test_detectors.py    # unit tests against small hand-crafted cases, not the full dataset
data/                    # generated trades, ground truth, and detection results -- committed as real evidence, not gitignored
reports/                 # generated exception reports and eval results -- also committed
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
cd src
python data_generator.py   # generates data/trades.csv, ground_truth.csv, news_events.csv
python evaluate.py         # runs all detectors, prints + saves precision/recall
python reports.py          # builds reports/exception_report.md
```

## Run tests

```bash
pytest tests/ -v
```

## Tech stack

Python · pandas · NumPy · pytest
