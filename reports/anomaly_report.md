# Bucket-Count Publication Analysis
Generated on: 2026-04-07 22:26:26

## ML Strategy
- **Feature**: `pub_count` — publications per 1-minute window (0 = silence).
- **Grid**: Full timeline zero-filled so silent minutes appear explicitly.
- **Goal**: Learn the normal publication frequency per App/RIC/FID/hour and flag deviations.

## Summary
- **Total 1-min Windows Analyzed:** 157,272
- **Anomalous Windows:** 784
- **Silent Windows (pub_count=0):** 141,569

## Top Anomalous Windows (Lowest Counts)
| App | RIC | FID | Minute | Count | Score |
|-----|-----|-----|--------|-------|-------|
| 101 | AAPL.O | LAST | 2026-04-06 10:22:00+00:00 | 52 | -1 |
| 101 | AAPL.O | LAST | 2026-04-05 10:02:00+00:00 | 54 | -1 |
| 101 | AAPL.O | LAST | 2026-04-05 10:03:00+00:00 | 55 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:08:00+00:00 | 55 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:40:00+00:00 | 55 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:32:00+00:00 | 56 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 09:21:00+00:00 | 56 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 10:25:00+00:00 | 56 | -1 |
| 102 | AAPL.O | BID | 2026-04-06 10:17:00+00:00 | 56 | -1 |
| 102 | AAPL.O | BID | 2026-04-06 10:25:00+00:00 | 56 | -1 |
| 102 | AAPL.O | LAST | 2026-04-06 10:15:00+00:00 | 56 | -1 |
| 101 | AAPL.O | BID | 2026-04-06 10:20:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:19:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:24:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 08:27:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 09:29:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 09:42:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 10:01:00+00:00 | 57 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 10:12:00+00:00 | 57 | -1 |
| 102 | AAPL.O | BID | 2026-04-06 10:20:00+00:00 | 57 | -1 |
