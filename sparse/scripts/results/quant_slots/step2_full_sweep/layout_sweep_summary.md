# Row layout full sweep: step2_full_sweep

8 CSV files, 2200 configs measured in both layouts (same graph, alternated runs, identical predictions).

Paired per-config ratio (slotted / original QPS): min 1.155x, p5 1.291x, median 1.431x, max 1.602x; 0 of 2200 configs slower.

Best QPS among configs with recall >= r:

| recall >= | original QPS | slotted QPS | slotted / original | original config | slotted config |
|---|---|---|---|---|---|
| 0.8 | 278,238 | 364,123 | 1.31x | M=32 efC=200 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=20 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.85 | 278,238 | 364,123 | 1.31x | M=32 efC=200 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=20 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.9 | 278,238 | 364,123 | 1.31x | M=32 efC=200 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=20 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.91 | 278,238 | 364,123 | 1.31x | M=32 efC=200 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=20 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.92 | 238,903 | 318,984 | 1.34x | M=32 efC=200 ef=40 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=200 ef=40 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.93 | 225,065 | 311,342 | 1.38x | M=32 efC=400 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 pat=512 | M=32 efC=400 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.94 | 225,065 | 311,342 | 1.38x | M=32 efC=400 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 pat=512 | M=32 efC=400 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.95 | 194,467 | 277,979 | 1.43x | M=32 efC=800 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=800 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.96 | 180,392 | 259,496 | 1.44x | M=32 efC=1600 ef=10 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=1600 ef=20 alpha=0.85 beta=2 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.97 | 149,525 | 219,266 | 1.47x | M=32 efC=1600 ef=40 alpha=0.85 beta=3 q=u8 seedT=8 seedH=8 seedS=4 | M=32 efC=1600 ef=40 alpha=0.85 beta=3 q=u8 seedT=8 seedH=8 seedS=4 |
| 0.98 | 102,243 | 152,123 | 1.49x | M=48 efC=800 ef=30 alpha=0.85 beta=5 q=u8 seedT=8 seedH=8 seedS=4 | M=48 efC=800 ef=40 alpha=0.85 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=2048 |
| 0.985 | 80,193 | 117,382 | 1.46x | M=32 efC=1600 ef=50 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 | M=64 efC=1600 ef=50 alpha=0.85 beta=3 q=u8 seedT=8 seedH=8 seedS=4 pat=128 |
| 0.99 | 59,316 | 85,264 | 1.44x | M=48 efC=1600 ef=50 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 | M=48 efC=1600 ef=50 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=128 |
| 0.995 | 36,925 | 54,975 | 1.49x | M=32 efC=1600 ef=400 alpha=0.85 beta=4 q=u8 seedT=8 seedH=8 seedS=4 pat=128 | M=32 efC=1600 ef=400 alpha=0.85 beta=4 q=u8 seedT=8 seedH=8 seedS=4 pat=128 |
| 0.997 | 22,454 | 33,392 | 1.49x | M=64 efC=1600 ef=400 alpha=0.85 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=128 | M=64 efC=1600 ef=400 alpha=0.85 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=128 |
| 0.998 | 14,265 | 20,371 | 1.43x | M=48 efC=1600 ef=800 alpha=0.85 beta=8 q=u8 seedT=8 seedH=8 seedS=4 pat=256 | M=64 efC=1600 ef=800 alpha=0.85 beta=5 q=u8 seedT=8 seedH=8 seedS=4 pat=256 |
