MS MARCO Full, pruned document bitmaps; M=32, efC=200, ef=1600, alpha=0.85, 64 threads. Recall/time/QPS are from counter-free searches; rejection statistics are from separate audit searches on the same graph, with identical predictions. Missed rejection rate = exact-only rejections / all exact rejections.

| Bitmap bytes | Threshold (%) | Recall (%) | Search time (s) | QPS | Skipped candidates | Exact rejections | Missed rejection (%) |
|---|---|---|---|---|---|---|---|
| 128 | 5 | 98.4957 | 1.2551 | 5,561.11 | 468,737 | 32,904,562 | 98.5790 |
| 128 | 10 | 98.4957 | 1.2344 | 5,654.53 | 4,843,229 | 63,375,926 | 92.3786 |
| 128 | 15 | 98.4255 | 1.1659 | 5,986.86 | 18,984,241 | 92,395,657 | 79.5102 |
| 256 | 5 | 98.4957 | 1.4594 | 4,782.88 | 3,495,715 | 32,847,178 | 89.3702 |
| 256 | 10 | 98.4513 | 1.3786 | 5,063.29 | 19,059,759 | 62,982,930 | 69.7709 |
| 256 | 15 | 98.2880 | 1.2397 | 5,630.52 | 45,012,157 | 92,030,431 | 51.1545 |
| 512 | 5 | 98.4814 | 1.9121 | 3,650.38 | 10,765,761 | 32,601,455 | 66.9963 |
| 512 | 10 | 98.3911 | 1.7800 | 3,921.30 | 35,198,233 | 61,939,370 | 43.2277 |
| 512 | 15 | 98.1877 | 1.6072 | 4,343.01 | 64,690,307 | 91,155,262 | 29.1241 |
