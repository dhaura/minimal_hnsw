# Row layout full sweep: slot448

2 CSV files, 108 configs measured in both layouts (same graph, alternated runs, identical predictions).

Paired per-config ratio (slotted / original QPS): min 1.148x, p5 1.163x, median 1.264x, max 1.454x; 0 of 108 configs slower.

Best QPS among configs with recall >= r:

| recall >= | original QPS | slotted QPS | slotted / original | original config | slotted config |
|---|---|---|---|---|---|
| 0.85 | 207,792 | 284,476 | 1.37x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.9 | 207,792 | 284,476 | 1.37x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.95 | 96,517 | 123,127 | 1.28x | M=32 efC=200 ef=50 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.97 | 43,667 | 53,513 | 1.23x | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.98 | 23,786 | 28,179 | 1.18x | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.99 | 12,429 | 14,731 | 1.19x | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
