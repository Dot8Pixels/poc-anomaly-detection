# Bucket-Count Publication Analysis
Generated on: 2026-04-07 22:08:59

## ML Strategy
- **Feature**: `pub_count` — publications per 1-minute window (0 = silence).
- **Grid**: Full timeline zero-filled so silent minutes appear explicitly.
- **Goal**: Learn the normal publication frequency per App/RIC/FID/hour and flag deviations.

## Summary
- **Total 1-min Windows Analyzed:** 13,461
- **Anomalous Windows:** 66
- **Silent Windows (pub_count=0):** 9,226

## Top Anomalous Windows (Lowest Counts)
| App | RIC | FID | Minute | Count | Score |
|-----|-----|-----|--------|-------|-------|
| 101 | TRI.N | LAST | 2026-04-06 12:27:00+00:00 | 60 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 08:42:00+00:00 | 73 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 15:13:00+00:00 | 83 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 10:53:00+00:00 | 96 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 10:33:00+00:00 | 104 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 15:33:00+00:00 | 112 | -1 |
| 101 | TRI.N | LAST | 2026-04-04 16:20:00+00:00 | 121 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 09:02:00+00:00 | 130 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 12:47:00+00:00 | 135 | -1 |
| 101 | TRI.N | LAST | 2026-04-07 16:18:00+00:00 | 168 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 16:19:00+00:00 | 174 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 13:58:00+00:00 | 175 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 09:22:00+00:00 | 179 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 15:25:00+00:00 | 180 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 08:11:00+00:00 | 185 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 15:59:00+00:00 | 186 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 08:44:00+00:00 | 187 | -1 |
| 101 | TRI.N | LAST | 2026-04-04 16:13:00+00:00 | 188 | -1 |
| 101 | TRI.N | LAST | 2026-04-06 15:04:00+00:00 | 188 | -1 |
| 101 | TRI.N | LAST | 2026-04-05 15:56:00+00:00 | 189 | -1 |
