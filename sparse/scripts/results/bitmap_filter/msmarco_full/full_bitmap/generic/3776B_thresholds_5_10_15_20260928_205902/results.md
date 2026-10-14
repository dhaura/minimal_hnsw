MS MARCO Full, pruned document bitmaps; M=32, efC=200, ef=1600, alpha=0.85, 64 threads. Recall/time/QPS are from counter-free searches; rejection statistics are from separate audit searches on the same graph, with identical predictions. Missed rejection rate = exact-only rejections / all exact rejections.

| Bitmap bytes | Threshold (%) | Recall (%) | Search time (s) | QPS | Skipped candidates | Exact rejections | Missed rejection (%) |
|---|---|---|---|---|---|---|---|
| 3776 | 5 | 98.4713 | 6.6604 | 1,047.98 | 30,824,807 | 30,824,807 | 0.0000 |
| 3776 | 10 | 98.2851 | 6.7277 | 1,037.51 | 58,926,044 | 58,926,044 | 0.0000 |
| 3776 | 15 | 97.9871 | 6.8607 | 1,017.39 | 89,087,926 | 89,087,926 | 0.0000 |
