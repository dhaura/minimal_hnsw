# Row layout full sweep: slot320

2 CSV files, 108 configs measured in both layouts (same graph, alternated runs, identical predictions).

Paired per-config ratio (slotted / original QPS): min 1.005x, p5 1.023x, median 1.072x, max 1.319x; 0 of 108 configs slower.

Best QPS among configs with recall >= r:

| recall >= | original QPS | slotted QPS | slotted / original | original config | slotted config |
|---|---|---|---|---|---|
| 0.85 | 210,015 | 260,265 | 1.24x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.9 | 210,015 | 260,265 | 1.24x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.95 | 96,772 | 112,975 | 1.17x | M=32 efC=200 ef=50 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=30 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.97 | 43,982 | 49,334 | 1.12x | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.98 | 23,982 | 26,423 | 1.10x | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.99 | 12,387 | 12,633 | 1.02x | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
