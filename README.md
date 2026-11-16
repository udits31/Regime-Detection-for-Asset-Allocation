# Market regime detection on Indian data: an out-of-sample evaluation

The original pipeline in `deepregime/` was developed between May and September 2026 and moved to this repository. The out-of-sample evaluation in `regime_eval/` was built in November 2026.

This project asks one question: does a regime model improve an equity and cash
timing rule on Indian market data, once the evaluation uses only information
available at the time of each decision?

On the data and settings used here, the answer is no. A hidden Markov model
(HMM) and a hidden semi-Markov model both reduce drawdown relative to holding
the NIFTY 50 ETF, but neither beats a static mix or plain volatility targeting,
and no Sharpe ratio difference is distinguishable from zero.

## What the strategy is

An equity and cash timing rule with a constant gold holding.

- The equity weight is between 20% and 80%. It is the probability-weighted
  average of per-regime weights, evenly spaced from 80% in the calmest regime
  to 20% in the most volatile.
- Gold is a constant 20%. Only the equity and cash split responds to the regime.
- The rest is in cash, earning the short rate.

Two regime models drive the rule:

- **Plain HMM.** Gaussian emissions with diagonal covariances on four features:
  daily log return, 21-day realised volatility, log India VIX, and drawdown from
  the 252-day high.
- **Explicit-duration hidden semi-Markov model.** The same emissions, but each
  regime has its own duration distribution (shifted negative binomial, at most
  252 days), estimated jointly with the emissions. The plain HMM is the special
  case of geometric durations.

## Results (real data, out of sample)

- Data: `NIFTYBEES.NS` and `GOLDBEES.NS` from Yahoo Finance (adjusted close,
  with the corrections listed below); cash and risk-free rate from FRED series
  `IRSTCI01INM156N`.
- Out-of-sample window: 5 January 2015 to 30 October 2026, 2,918 trading days.
- Models are refitted every 63 trading days on an expanding window that ends the
  day before each block. The first training window is 19 January 2010 to
  31 December 2014.
- Figures are after costs of 10 basis points per unit of ETF turnover. Signals
  are computed at the close of day t and traded at the close of day t+1.

| Strategy | Annual return | Volatility | Sharpe ratio | Maximum drawdown | Turnover per year |
|---|---|---|---|---|---|
| Buy and hold `NIFTYBEES.NS` | 10.3% | 14.7% | 0.33 | −36.3% | 0.09 |
| Static mix 50/20/30 | 10.4% | 7.9% | 0.54 | −18.9% | 0.26 |
| 200-day moving average | 9.7% | 9.0% | 0.41 | −15.8% | 6.09 |
| Volatility targeting | 10.3% | 8.5% | 0.49 | −13.4% | 2.41 |
| Plain HMM strategy | 9.3% | 7.9% | 0.41 | −12.8% | 2.21 |
| Semi-Markov strategy | 9.3% | 7.9% | 0.41 | −12.1% | 2.13 |

Sharpe ratio difference, strategy minus benchmark, with a 95% interval from a
paired stationary block bootstrap (mean block 21 days, 10,000 resamples):

| Benchmark | Plain HMM strategy | Semi-Markov strategy |
|---|---|---|
| Buy and hold `NIFTYBEES.NS` | +0.08 (−0.27 to +0.42) | +0.07 (−0.28 to +0.41) |
| Static mix 50/20/30 | −0.13 (−0.37 to +0.13) | −0.13 (−0.38 to +0.13) |
| 200-day moving average | 0.00 (−0.25 to +0.25) | 0.00 (−0.26 to +0.26) |
| Volatility targeting | −0.08 (−0.21 to +0.04) | −0.09 (−0.21 to +0.03) |
| Plain HMM strategy | | −0.005 (−0.027 to +0.019) |

Every interval includes zero.

### What the evaluation showed

- **No improvement over simple rules.** Both regime strategies have a lower
  Sharpe ratio than the static mix and volatility targeting, and about the same
  as the moving-average rule.
- **Lower drawdown is the one visible effect.** The regime strategies have the
  smallest maximum drawdown in the table (about −12% to −13%, against −36% for
  buy and hold and −19% for the static mix). Volatility targeting gets most of
  that reduction without a regime model.
- **The semi-Markov change did not matter for allocation.** The two models'
  target weights have a correlation of 0.99, and the Sharpe difference between
  them is −0.005 (−0.027 to +0.019).
- **Costs.** The plain HMM strategy's Sharpe ratio is 0.44 before costs, 0.41 at
  10 basis points and 0.37 at 25. The semi-Markov strategy's is 0.43, 0.41 and
  0.36. The moving-average rule is the most cost-sensitive (0.48, 0.41, 0.31).
- **Sub-periods.** In 2015 to 2019 both regime strategies had a Sharpe ratio of
  0.09, below buy and hold (0.22). In 2020 they had about 0.99 against 0.50 for
  buy and hold, with a drawdown of −11% against −36%.
- **Number of states.** The in-window selection rule chose four states, the top
  of the allowed range, in 39 of 47 refits for the HMM and 40 of 47 for the
  semi-Markov model.

Tables for each cost level, the sub-periods and the per-refit diagnostics are in
`results/real/`.

### Regime durations (in-sample, descriptive)

`results/real/in_sample_survival_*.csv` hold Kaplan–Meier survival curves of
regime lengths for both models, fitted once on the full sample with three
states. These use the whole sample, including the evaluation window, and are
not trading results. The two models assign the same regime on 98% of days and
give nearly the same median regime lengths.

## How look-ahead is prevented

- **Filtered probabilities only.** `regime_eval/filter.py` computes the
  probability of each regime on day t from data up to day t. Library routines
  that use the whole sample are not used for signals.
- **Everything is fitted inside the training window:** the scaler, the model,
  the number of states (chosen on the last 20% of each window) and the ordering
  of states by volatility.
- **Trades are lagged.** A signal from the close of day t is traded at the close
  of day t+1 and first affects the return of day t+2.
- **A test checks all of this.** `tests/test_leakage.py` recomputes the signals
  with later data removed and requires earlier signals to be unchanged. It
  passes for both pipelines, and it fails, as it must, for a centred average,
  full-sample standardisation and the library's smoothed probabilities.

Where look-ahead could still enter:

- The features, the allowed numbers of states, the allocation weights and the
  cost level were chosen by the author with general knowledge of the period.
  They were fixed and committed before any out-of-sample result was computed
  (`regime_eval/settings.toml`), but they are not the output of a blind process.
- The data corrections for December 2019 sit inside the evaluation window. They
  follow a published split notice and are listed below.
- The earlier pipeline in `deepregime/` was run on the full sample before this
  evaluation was designed. No setting here was taken from its output.

## How many configurations were run

One. The settings were frozen before the out-of-sample period was examined, and
the evaluation was run once for each of the two models. It is reported at three
cost levels (0, 10 and 25 basis points), which were declared in advance. Nothing
was tuned afterwards. `results/config_log.csv` records every data check,
correction, pre-declared choice and run.

## Reproducing the results

Python 3.13.

```
pip install -r requirements-eval.txt
python -m regime_eval.data download
python reproduce.py
pytest -m "not slow"
```

`reproduce.py` rebuilds the panel, reads the committed walk-forward signals and
regenerates every table. `python reproduce.py --refit` refits both models from
scratch, which takes about an hour. `pytest` without the marker also runs the
two look-ahead tests on the real pipeline (about five minutes).

**Market data is not in this repository.** `download` fetches it from Yahoo
Finance and FRED. Yahoo revises adjusted prices, so a later download may not
match the one used here. `data/processed/panel_meta.json` holds SHA-256
fingerprints of the files used; if yours differ, the numbers may differ too.

### Data corrections

Both are applied by `regime_eval/data.py` and logged in `results/config_log.csv`.

- `NIFTYBEES.NS`, 5 and 8 February 2010 and 6 and 7 October 2010: the Yahoo
  price jumps away from the NIFTY 50 index for one day and returns the next. The
  ETF's return is replaced by the index's return on those four days.
- Both ETFs, 19 and 20 December 2019: the Yahoo closes on the two ex-split days
  are one split ratio too low. They are multiplied by the published ratio (10
  for `NIFTYBEES`, 100 for `GOLDBEES`; NSE Clearing circular NCL/CMPT/42889).

India VIX is missing on 21 days; the previous value is carried forward. No gap
is longer than one trading day.

## Which outputs are real and which are synthetic

- Everything in `results/real/` comes from real market data.
- This evaluation writes no synthetic results. Simulated series are used only
  inside the tests (`tests/test_hsmm.py`, `tests/test_decode_survival.py`,
  `tests/test_backtest.py`, `tests/test_evaluate.py`), where the true answer has
  to be known.
- The earlier pipeline in `deepregime/` has a synthetic data generator with a
  hard-coded regime calendar, and falls back to it without warning if a download
  fails. Any output it prints with `--synthetic`, including its regime-duration
  table, describes that simulated series and says nothing about the market.

## Limitations

- **One path.** There is one out-of-sample history of about twelve years. The
  intervals are wide, and a different period could rank the strategies
  differently.
- **Cash rate.** The overnight call-money rate is used as a proxy for the
  91-day Treasury bill yield, monthly, applied with a one-month lag.
- **Duration cap.** Regime lengths are capped at 252 trading days. In 7 of 47
  refits, more than 5% of a regime's expected endings fell at the cap (the
  highest share was 9%).
- **State count at the edge.** The selection rule usually picked the largest
  allowed number of states, so the range itself may be binding.
- **Fitting limits.** The semi-Markov fit stops after 30 iterations, which some
  fits reach before fully converging.
- **Costs.** A flat cost per unit of turnover stands in for brokerage, spread
  and impact. Taxes are ignored.
- **Data.** Prices come from Yahoo Finance and needed the corrections above.
  Other errors may remain.
- **Fixed weights.** The per-regime weights are set by rule, not optimised.

## Layout

| Path | Contents |
|---|---|
| `regime_eval/` | The evaluation: data cache, features, filter, both models, walk-forward loop, backtest, benchmarks, bootstrap |
| `regime_eval/settings.toml` | The frozen settings |
| `tests/` | Known-answer tests and the look-ahead tests |
| `reproduce.py` | Regenerates every reported number |
| `results/real/` | Output tables and walk-forward signals |
| `results/config_log.csv` | Log of every check, correction, choice and run |
| `deepregime/` | The earlier pipeline, kept unchanged at tag `v2-audited`. It is not used by the evaluation and has known defects, including look-ahead in its backtests. |
