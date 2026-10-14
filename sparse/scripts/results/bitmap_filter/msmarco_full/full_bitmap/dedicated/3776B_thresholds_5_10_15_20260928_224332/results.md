MS MARCO Full, pruned document bitmaps; M=32, efC=200, ef=1600, alpha=0.85, 64 threads. Recall/time/QPS are from counter-free searches; rejection statistics are from separate audit searches on the same graph, with identical predictions. Missed rejection rate = exact-only rejections / all exact rejections.

| Bitmap bytes | Threshold (%) | Recall (%) | Search time (s) | QPS | Skipped candidates | Exact rejections | Missed rejection (%) |
|---|---|---|---|---|---|---|---|
| 3776 | 5 | 98.4628 | 1.7201 | 4,057.95 | 30,820,761 | 30,820,761 | 0.0000 |
| 3776 | 10 | 98.2751 | 1.9548 | 3,570.71 | 58,925,541 | 58,925,541 | 0.0000 |
| 3776 | 15 | 97.9542 | 2.1494 | 3,247.44 | 89,084,556 | 89,084,556 | 0.0000 |
