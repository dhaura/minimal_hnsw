# Online stopping rules, msmarco_full (195 points)

Ratio = QPS / fixed-ef QPS at the same recall (fixed-ef curve interpolated in log QPS).
Fixed ef spans recall 0.9735-0.9974.

## Per rule: recall reached (largest ef cap) and best matched-recall ratio

| rule | family | recall at cap | best ratio (at recall, ef cap) | median ratio over caps |
|---|---|---|---|---|
| 128 | patience | 0.9936 | 1.319 (0.9935, ef=400) | 1.222 |
| 64 | patience | 0.9903 | 1.170 (0.9903, ef=300) | 1.091 |
| g0.02 | score gap | 0.9770 | 1.154 (0.9767, ef=100) | 1.060 |
| g0.05 | score gap | 0.9852 | 1.239 (0.9847, ef=200) | 1.146 |
| g0.05:m10:p64 | combo | 0.9828 | 1.138 (0.9826, ef=150) | 1.066 |
| g0.075 | score gap | 0.9893 | 1.389 (0.9892, ef=300) | 1.269 |
| g0.1 | score gap | 0.9915 | 1.427 (0.9915, ef=400) | 1.270 |
| g0.15 | score gap | 0.9937 | 1.437 (0.9934, ef=400) | 1.284 |
| g0.1:m10:p128 | combo | 0.9905 | 1.323 (0.9905, ef=300) | 1.213 |
| g0.2 | score gap | 0.9952 | 1.397 (0.9947, ef=600) | 1.235 |
| g0.2:m10:p512 | combo | 0.9948 | 1.406 (0.9948, ef=800) | 1.208 |
| g0.3 | score gap | 0.9969 | 1.805 (0.9969, ef=2400) | 1.094 |
| g0.3:m10:p512 | combo | 0.9957 | 1.322 (0.9957, ef=1600) | 1.057 |

## Best matched-recall ratio per family by recall window (setting, ef cap)

| recall window | patience | score gap | combo |
|---|---|---|---|
| [0.975, 0.985) | 0.98x (128, ef=75) | 1.24x (g0.05, ef=200) | 1.14x (g0.05:m10:p64, ef=150) |
| [0.985, 0.99) | 1.16x (64, ef=200) | 1.39x (g0.075, ef=300) | 1.27x (g0.1:m10:p128, ef=200) |
| [0.99, 0.993) | 1.17x (64, ef=300) | 1.43x (g0.1, ef=400) | 1.32x (g0.1:m10:p128, ef=300) |
| [0.993, 0.995) | 1.32x (128, ef=400) | 1.44x (g0.15, ef=400) | 1.41x (g0.2:m10:p512, ef=800) |
| [0.995, 0.997) | - | 1.80x (g0.3, ef=2400) | 1.32x (g0.3:m10:p512, ef=1600) |
| [0.997, 1.0) | - | - | - |

## Best QPS at recall >= r

| recall | fixed ef | best rule | rule QPS | speedup |
|---|---|---|---|---|
| 0.975 | 110,725 | g0.02 (ef=75) | 158,951 | 1.44x |
| 0.98 | 110,725 | g0.05 (ef=75) | 128,958 | 1.16x |
| 0.985 | 86,590 | g0.05 (ef=300) | 114,635 | 1.32x |
| 0.99 | 61,579 | g0.1 (ef=200) | 78,707 | 1.28x |
| 0.9925 | 34,650 | g0.15 (ef=300) | 53,887 | 1.56x |
| 0.995 | 19,033 | g0.2 (ef=1200) | 32,174 | 1.69x |
| 0.996 | 14,408 | g0.3 (ef=800) | 19,309 | 1.34x |
