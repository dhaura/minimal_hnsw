# Row layout full sweep: slot384

2 CSV files, 108 configs measured in both layouts (same graph, alternated runs, identical predictions).

Paired per-config ratio (slotted / original QPS): min 1.124x, p5 1.131x, median 1.207x, max 1.328x; 0 of 108 configs slower.

Best QPS among configs with recall >= r:

| recall >= | original QPS | slotted QPS | slotted / original | original config | slotted config |
|---|---|---|---|---|---|
| 0.85 | 211,162 | 270,911 | 1.28x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.9 | 211,162 | 270,911 | 1.28x | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=10 alpha=0.7 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.95 | 96,369 | 116,322 | 1.21x | M=32 efC=200 ef=10 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=30 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.97 | 43,694 | 50,851 | 1.16x | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=200 alpha=0.7 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.98 | 23,752 | 27,290 | 1.15x | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=400 alpha=0.7 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
| 0.99 | 12,324 | 14,835 | 1.20x | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=32 efC=200 ef=1600 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
