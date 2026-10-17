# Assumptions and limitations (superseded)

This file was a working draft of 13 October 2026: six limits of the Week 7–8 result and a 20-row audit table of the assumptions behind it. This note replaces it.

## Where the limits are now

- The README lists the nine limits in short form, in the section "Limitations".
- Section 8 of the research report that goes with this repository gives each limit in full. The report is a separate document and is not stored here.
- The earlier version and its audit table are kept in the Git history of this repository: `week8/assumptions_and_limitations.md` at commit `3834a9d`. Where it and the report differ, the report is the later and checked version.

## Corrections already made in the report

Four statements in the earlier version were wrong or loose. The report has the corrected versions.

1. **Cost effect.** The earlier version said that at an assumed 25 basis points of the value traded no gap moves by more than about 0.014. The report gives 0.013 (M = 60 minus 1/N) and 0.007 (M = 120 minus 1/N).
2. **Block intervals.** The earlier version said the block intervals were no wider than the locked ones, except one by under 0.015. That one is 0.020 wider in total, with its ends 0.013 and 0.007 further out.
3. **Proposition 1.** The earlier version used Proposition 1 of the June 2006 draft of DeMiguel, Garlappi and Uppal as if it applied to the long-only rule tested here. The proposition concerns the unconstrained rule.
4. **Number of datasets.** The earlier version spoke of "the draft's six datasets". The June 2006 draft tests seven datasets and reports results for six.

## Who wrote the earlier version

Part 1 (the limits) was drafted by the author and corrected by the mentor, with numbers added from the saved printouts. Part 2 (the audit table) began as a table produced by an outside AI tool, labelled by the author at the time; the mentor checked every row against the project records, corrected the status and wording, and added rows. Part 3 (the ranking of the three limits that matter most) was the mentor's. The README section "How this was made" says who the mentor is.
