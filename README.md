# Maximum-Sharpe versus 1/N: a four-stock out-of-sample test

This repository holds the code, monthly data and saved printouts for an out-of-sample test of whether a long-only maximum-Sharpe portfolio of four stocks (JPM, BAC, PG and MSFT), re-estimated each month, beat equal weighting (1/N); the question comes from the debate between DeMiguel, Garlappi and Uppal (2009) and Kritzman, Page and Turkington (2010).

The research report that goes with this repository, "Optimised or Equal Weights? What 105 Months of Four US Stocks Can and Cannot Decide", is a separate document and is not stored here.

## The question

For JPM, BAC, PG and MSFT over January 2018–September 2026, did a long-only maximum-Sharpe portfolio, re-estimated monthly on the previous M months (M = 60 and M = 120), earn a higher out-of-sample Sharpe ratio (Sharpe ratio of returns in excess of the 3-month Treasury-bill rate) than an equally weighted (1/N) portfolio, by how much, and what range of true differences is consistent with that result? A secondary, exploratory result is the M = 120 gap minus the M = 60 gap.

## The answer in three sentences

Over the 105 test months from January 2018 to September 2026, the long-only maximum-Sharpe portfolio earned annualised out-of-sample Sharpe ratios of 0.7845 (at M = 60) and 0.8855 (at M = 120) compared with 0.7790 for the equally weighted 1/N benchmark, yielding gaps of +0.005 (with a 95% interval of [−0.428, +0.439]) and +0.106 (interval [−0.348, +0.539]), respectively, each interval being the wider of a formula interval and a bootstrap interval (the formula's for M = 60, the bootstrap's for M = 120). Because zero is inside every interval (the exploratory M = 120 minus M = 60 difference, +0.101 [−0.091, +0.292], included) and the precision is limited by the single 105-month path, the data leave open true gaps from about −0.4 to about +0.5 Sharpe units; the smallest gaps this test would detect 80% of the time at the 5% significance level are 0.619 (M = 60) and 0.633 (M = 120), and a smaller true gap would be missed more than one time in five. This result does not show that the optimiser beats 1/N, nor does it prove that the strategies perform equally; it cannot say why the gaps are what they are, and it covers only these four stocks, this period, one optimiser and two window lengths (see Limitations).

## Limitations

Read these limits before relying on any number here. Nine limits bound the result. No check in this project shows how large the three that matter most are: the single 105-month path (2), hindsight in the choice of the four stocks (5) and the narrow reach of the result (1 and 4). Section 8 of the research report gives every limit in full.

1. **Universe.** Four US stocks, and in practice the optimiser used about three of them (BAC almost never appears in its weights). Few assets is the optimiser's best case, so the result says nothing about larger universes.
2. **Period.** One path of 105 months, of which 60 (2019–2023) had been studied before the test was fixed. After the run, the gap against 1/N changed sign between 2018–2023 and 2024–2026: from +0.224 to −0.552 at M = 60 and from +0.334 to −0.392 at M = 120. The data cannot tell that change from noise.
3. **Costs.** No trading costs are charged. At an assumed 25 basis points of the value traded, the gaps against 1/N would fall by about 0.013 (M = 60) and 0.007 (M = 120), small against standard errors of 0.221 and 0.226.
4. **Optimiser class.** One rule (the exact long-only maximum-Sharpe weights from the sample mean and covariance, used as they are) and two window lengths, 60 and 120 months. Other rules and other window lengths were not tested.
5. **Hindsight.** The four stocks were chosen when their history was known, and their 2019–2023 returns had been studied before the test was fixed. This probably pushes the reported gaps too high. Its size cannot be measured inside this design.
6. **Precision.** The smallest gaps the test would detect four times in five are 0.619 (M = 60), 0.633 (M = 120) and 0.274 (M = 120 minus M = 60), so an interval that contains zero is not evidence of equality. The intervals assume that months are independent. The dependence check found no sign of dependence, but it could not rule out a dependence of about 0.2, which would make the intervals too narrow.
7. **Data and risk-free rate.** All prices come from one vendor, whose adjusted prices are recalculated after each dividend. The monthly risk-free rate (TB3MS ÷ 100 ÷ 12) makes three small approximations. Setting the rate to zero raised the Sharpe ratios by 0.149 to 0.164 and moved the gaps by at most 0.016.
8. **Process.** That the question, the design and the predictions were fixed before the first run is the author's own record, because a Git timestamp cannot show it. The same holds for the rule that no 2024–2026 stock return or Sharpe ratio was printed before the run, and the monthly tables saved earlier did contain those months. The tag `primary-result` fixes the code and printouts of the primary result, not when the run was decided.
9. **Comparison with the literature.** The June 2006 working-paper draft of DeMiguel, Garlappi and Uppal (2009), called "the draft" in this README, reports monthly Sharpe ratios from other samples, asset sets, risk-free series and test lengths, and its Table 3 uses M = 120 only, so patterns can be compared but not levels. The draft was read, not the published article, and Kritzman, Page and Turkington (2010) is cited from its abstract only.

The earlier draft of this list, [week8/assumptions_and_limitations.md](week8/assumptions_and_limitations.md), is superseded. The file is now a short note that says where the draft is kept.

## How to run

The commands are for macOS or Linux. Use Python 3.14.6 and the packages pinned in `requirements.txt`. Run every script from the project folder: the scripts open files such as `data/...` by relative path.

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt

The four scripts below reproduce the printed results from the files in `data/`. Each has a saved printout, and a command prints nothing if your output is identical.

    python3 week7/rolling_test.py | diff week7/rolling_test_reference_2026-10-12.txt -
    python3 week7/first_run.py | diff week7/first_run_reference_2026-10-12.txt -
    python3 week8/post_run_checks.py | diff week8/post_run_checks_reference_2026-10-13.txt -
    python3 week8/dependence_check.py | diff week8/dependence_check_reference_2026-10-13.txt -

`dependence_check.py` simulates thousands of invented histories and took about 40 seconds on the author's laptop. With other library versions the last printed digits may differ; compare these numbers instead.

| Script | What to expect |
|---|---|
| `week7/rolling_test.py` | 105 test months, 2018-01 to 2026-09; every check prints `True` |
| `week7/first_run.py` | annualised Sharpe ratios 0.7845 (M = 60), 0.8855 (M = 120), 0.7790 (1/N); gaps against 1/N +0.005 and +0.106; M = 120 minus M = 60 +0.101 |
| `week8/post_run_checks.py` | with r_f = 0 the Sharpe ratios are 0.9353 (M = 60), 1.0497 (M = 120) and 0.9277 (1/N); buy-and-hold 1/N 0.7886 |
| `week8/dependence_check.py` | 2 of 72 autocorrelations outside the band; block-length-6 SE over the locked formula SE 0.980, 1.016 and 0.743 |

Weeks 1 to 6 are the learning scripts, kept as a dated record. Several of them read `data/jpm.csv` and `data/prices.csv`, raw Yahoo Finance downloads that are not in this repository. `week2/get_jpm_data.py`, `week4/get_data.py` and `week7/get_data_rolling.py` download such files again and overwrite files of the same name. A new download may not be identical: Yahoo re-bases adjusted prices after each dividend (on one date range JPM's levels differed by a constant 0.4965% while daily returns agreed to about 1e-6), so later digits of those printouts can differ. Scripts in weeks 3, 4 and 6 open a chart window; close it to let the script continue.

## Data

- `data/monthly_returns_2008-01_to_2026-09.csv`, `data/monthly_rf_2008-01_to_2026-09.csv` and `data/monthly_excess_returns_2008-01_to_2026-09.csv`: monthly simple returns of the four stocks, the monthly risk-free rate (TB3MS ÷ 100 ÷ 12) and the excess returns (returns minus the risk-free rate), January 2008–September 2026 (225 months). Built by `week7/prepare_monthly_data.py` from the daily adjusted-close prices in `data/prices_rolling.csv` (Yahoo Finance, downloaded with yfinance) and the T-bill file below; the script also compares those prices with the older `data/prices.csv`.
- `data/tbill_TB3MS_downloaded_2026-10-08.csv`: FRED series TB3MS (3-month Treasury bill, secondary market rate, percent per year), downloaded by hand. The date is the file's time stamp.
- `data/SHA256SUMS.txt`: checksums of these four files. From the project folder, `shasum -a 256 -c data/SHA256SUMS.txt` prints `OK` for each file identical to the one used.
- Not in the repository: the raw Yahoo price files. Their SHA-256 checksums, so the files used can be recognised:
  - `data/jpm.csv` (Weeks 2–3): `110674bcb0acd4e45834f36549c3f24dfd692b0973050a6e1f45c658602c6d01`
  - `data/prices.csv` (Weeks 4–7): `1aaa617a7ba6a980aa5a02e7395801cdcb90b6643e7d4f8405d1b56761f9eb2e`
  - `data/prices_rolling.csv` (Week 7): `93458fc7d9af91c885f19fa7cddb4ebbee7696bee1e48234ae04564d64f4a344`

The monthly tables are derived from Yahoo Finance prices. Two things were found about the terms that apply. The yfinance package page says that Yahoo's API is "intended for personal use only" and refers users to Yahoo's terms. Yahoo's English terms of service (`legal.yahoo.com/xw/en/yahoo/terms/otos/`, read through a page-summarising tool, so the wording is not verbatim; the version for Singapore may differ) say that content must not be reproduced or distributed without explicit written permission and that data must not be collected by automated means without permission. The author is not a lawyer and has not been able to settle whether publishing monthly returns derived from the prices is allowed. The raw daily price files are not in the repository and have never been committed. The monthly tables are included so that the results can be reproduced, and will be removed on request.

## Method and sources

Each month from January 2018 to September 2026 the portfolio is rebuilt from the previous M months of monthly excess returns (sample mean and covariance, no shrinkage): the exact long-only weights that maximise the Sharpe ratio. 1/N holds a quarter in each stock. Both rebalance monthly, with no trading costs. Sharpe ratios are annualised by √12. Intervals are 95%: the wider of a formula interval and a paired-bootstrap interval (5,000 resamples, seed 42) is quoted.

The standard error of a difference of two Sharpe ratios is computed with the variance in footnote 21 of DeMiguel, Garlappi and Uppal (June 2006 draft, p. 11), rewritten in terms of the Sharpe ratios and their correlation. The draft attributes the test to Jobson and Korkie (1981) with Memmel's (2003) correction. The draft's description of it was read, not the two original papers.

The strategy applies the draft's non-negativity constraint to the same plug-in problem as the draft's sample-based mean-variance strategy with shortsale constraints (mv-c: Section 2.5, p. 8; Table 1, row 8, p. 40). The two are not claimed to be identical: the draft's remark on risk aversion (Section 7.7, p. 29) does not fit a non-negativity constraint alone.

Page numbers are printed pages of the June 2006 working-paper draft; the published version has not been checked.

### References

- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? Review of Financial Studies 22(5), 1915–1953.
- Jobson, J. D. and Korkie, R. (1981). Performance hypothesis testing with the Sharpe and Treynor measures. Journal of Finance 36, 889–908.
- Kritzman, M., Page, S. and Turkington, D. (2010). In defense of optimization: the fallacy of 1/N. Financial Analysts Journal 66(2), 31–39.
- Memmel, C. (2003). Performance hypothesis testing with the Sharpe ratio. Finance Letters 1, 21–23.

Only the June 2006 draft of the first paper has been read. The second paper is cited from its abstract: its full text has not been read, and the results here do not depend on it. Jobson and Korkie (1981) and Memmel (2003) are cited as the draft credits them, and neither has been read.

The research report also cites Efron (1979), Ledoit and Wolf (2008), Lo (2002), Markowitz (1952), Michaud (1989), Sharpe (1994) and White (2000), each for its main point as given in an abstract, a standard reference or the draft, and not from a reading of the full text; and Fama (1970), for two statements (pp. 383–384). The report's References list has the full entries.

## Process notes

- The question and the design were locked on 12 October 2026 at 13:32 SGT, before the first out-of-sample run. That time is the author's own record: a Git timestamp cannot show that it came before the run. The Git tag `primary-result` marks the code and saved printouts of the primary result.
- Everything done after the first run is labelled post-run. `week8/post_run_checks.py` and `week8/dependence_check.py` report their results whatever they show and do not replace the primary result.

## How this was made

This repository was built through an interactive collaboration between the author and a mentor AI (Claude, made by Anthropic). The mentor taught the methods lesson by lesson, set the exercises, reviewed the author's code and text, and wrote the lesson plans and records. In Weeks 7 and 8 it supplied scripts, or skeletons with blanks for the author to fill in; the two Week 8 scripts (`week8/post_run_checks.py` and `week8/dependence_check.py`) were delivered by the mentor without being run by it, and the author pasted them into the project files and ran them. The author ran every script and read its output. The research question and the design were written up by the mentor from the author's choices or from recommendations the author approved; the author confirmed the final wording and locked the design.

The research report and the reflection that go with this repository are separate documents and are not stored here. The mentor wrote the report in the author's voice, section by section, from the lesson records, the saved printouts and the June 2006 draft. In several places the author wrote the answer to a question the mentor had left open. Those passages are based on the author's words, which the mentor corrected or reworded where they claimed more than the results show or were hard to follow. The author read every section, asked for cuts and changes to wording, citations and layout, and approved the final text. The chart code in `week7/first_run.py` and `week8/post_run_checks.py` was supplied by the mentor; the author ran the scripts, and the four charts in the report are the files they saved. The reflection was written the same way from the weekly records, and its closing paragraph is the author's own, written in Chinese and translated by the mentor. The report points to this README for how it was made; this section is that account.

Other AI tools were also used, as the author reported at the time: for the `targets` loop in `week6/portfolio_metrics.py`, for the author's notes on the abstract of Kritzman, Page and Turkington (2010), and for the first version of an audit table of assumptions, which is kept in the Git history in the earlier version of `week8/assumptions_and_limitations.md` (commit `3834a9d`). The mentor checked that table row by row, corrected it and added rows; the top of that earlier version says who wrote each of its parts. The mentor drafted this README and wrote its first sentence, and later rewrote its limits, data note, sources, file list and this section; the answer paragraph and an earlier version of this section were supplied by the author, and the mentor corrected both.

## What is where

- `week1/` to `week6/`: learning scripts, kept as a dated record.
- `week7/`:
  - data preparation: `get_data_rolling.py` and `prepare_monthly_data.py`;
  - the rolling test (`rolling_test.py`) and the primary result (`first_run.py`), with shared code in `portfolio_tools.py`;
  - the precision and power planning scripts (`precision_planning.py`, `power_planning.py`) and the checks of the standard-error formula and of the bootstrap (`formula_check.py`, `bootstrap_check.py`);
  - `estimation_error.py`: the invented worlds behind the table in Section 2 of the report (it has no saved printout);
  - `optimiser.py`: the first optimiser, kept as a dated record; it reads `data/prices.csv`, which is not in the repository.
- `week8/`: post-run checks, the dependence check and `assumptions_and_limitations.md`, a note saying that the earlier limitations draft is superseded.
- `data/`: the committed data (see Data). `figures/` is created when the scripts run and is not committed.
- `requirements.txt` and `LICENSE`.

## Licence

MIT (see `LICENSE`) for the code and text. The data files remain subject to the terms of their sources.
