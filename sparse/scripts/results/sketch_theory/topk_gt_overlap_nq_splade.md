# Query terms among the top-K terms of true top-10 docs, nq_splade (alpha 0.85)

34,520 query/GT pairs, mean pruned nnz 80.3; rows shorter than 16 / 24 / 32 terms: 0.0% / 0.4% / 1.4%.

| terms | mean hits | 0 hits | <=1 | <=2 | top-1 with 0 | recall@10 ceiling | queries with all 10 |
|---|---|---|---|---|---|---|---|
| top-16 | 4.78 | 0.48% | 3.78% | 13.58% | 0.38% | 0.9952 | 95.94% |
| top-24 | 6.74 | 0.05% | 0.84% | 4.11% | 0.06% | 0.9995 | 99.48% |
| top-32 | 8.63 | 0.01% | 0.17% | 1.27% | 0.00% | 0.9999 | 99.88% |
| full pruned row | 18.82 | 0.00% | 0.00% | 0.00% | 0.00% | 1.0000 | 100.00% |
