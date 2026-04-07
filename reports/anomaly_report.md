# Bucket-Count Publication Analysis
Generated on: 2026-04-07 23:31:07

## ML Strategy
- **Feature**: `pub_count` — publications per 1-minute window (0 = silence).
- **Grid**: Full timeline zero-filled so silent minutes appear explicitly.
- **Goal**: Learn the normal publication frequency per App/RIC/FID/hour and flag deviations.

## Summary
- **Total 1-min Windows Analyzed:** 518,400
- **Anomalous Windows:** 2,592
- **Silent Windows (pub_count=0):** 68,922

## Top Anomalous Windows (Lowest Counts)
| App | RIC | FID | Minute | Count | Score |
|-----|-----|-----|--------|-------|-------|
| 102 | AAPL.O | LAST | 2026-03-30 16:02:00+00:00 | 195 | -1 |
| 102 | AAPL.O | LAST | 2026-03-16 09:50:00+00:00 | 207 | -1 |
| 102 | TRI.N | BID | 2026-03-23 09:53:00+00:00 | 210 | -1 |
| 102 | TRI.N | BID | 2026-03-30 09:35:00+00:00 | 211 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 16:35:00+00:00 | 213 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 16:01:00+00:00 | 214 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 16:31:00+00:00 | 214 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 16:20:00+00:00 | 215 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 09:35:00+00:00 | 218 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 16:26:00+00:00 | 218 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 09:22:00+00:00 | 219 | -1 |
| 101 | AAPL.O | LAST | 2026-03-30 09:31:00+00:00 | 219 | -1 |
| 101 | AAPL.O | LAST | 2026-03-30 16:12:00+00:00 | 219 | -1 |
| 101 | AAPL.O | LAST | 2026-03-30 16:19:00+00:00 | 219 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 15:01:00+00:00 | 219 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 16:39:00+00:00 | 220 | -1 |
| 101 | AAPL.O | LAST | 2026-03-16 16:40:00+00:00 | 220 | -1 |
| 101 | AAPL.O | LAST | 2026-03-23 09:42:00+00:00 | 220 | -1 |
| 101 | AAPL.O | LAST | 2026-03-23 16:01:00+00:00 | 220 | -1 |
| 101 | AAPL.O | LAST | 2026-04-06 09:17:00+00:00 | 220 | -1 |
