# Liveness & Expectation Report
Generated on: 2026-04-04 23:19:49

## Learned Profile Criteria
- **Dynamic Baseline**: The system learned which Hours each App/RIC/FID is supposed to be active.
- **Criteria**: App Number, RIC Name, FID Name.
- **Expectation**: If a stream was active during an hour in the baseline, it is required to publish every minute.

## Summary
- **Total Expected Windows:** 5,360
- **Missing Publications Detected:** 1,086

## Sample Missing Data (Data should have published but did not)
| App | RIC | FID | Expected Hour | Window Missing | Status |
|-----|-----|-----|---------------|----------------|--------|
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:02:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:03:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:04:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:05:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:06:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:07:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:08:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:09:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:10:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:11:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:12:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:13:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:14:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:15:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:16:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:17:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:18:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:19:00 | MISSING |
| 101 | TRI.N | LAST | 13:00 | 2026-04-01 13:20:00 | MISSING |
| 101 | TRI.N | LAST | 14:00 | 2026-04-01 14:44:00 | MISSING |
