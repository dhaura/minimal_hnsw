# Query terms among the top-K terms of true top-10 docs, msmarco_full (alpha 0.85)

69,800 query/GT pairs, mean pruned nnz 62.3; rows shorter than 16 / 24 / 32 terms: 0.3% / 2.5% / 6.7%.

| terms | mean hits | 0 hits | <=1 | <=2 | top-1 with 0 | recall@10 ceiling | queries with all 10 |
|---|---|---|---|---|---|---|---|
| top-16 | 6.16 | 0.01% | 0.46% | 4.28% | 0.00% | 0.9999 | 99.90% |
| top-24 | 8.39 | 0.00% | 0.06% | 0.98% | 0.00% | 1.0000 | 99.99% |
| top-32 | 10.60 | 0.00% | 0.01% | 0.28% | 0.00% | 1.0000 | 99.99% |
| full pruned row | 19.23 | 0.00% | 0.00% | 0.01% | 0.00% | 1.0000 | 100.00% |
