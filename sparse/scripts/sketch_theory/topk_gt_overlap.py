#!/usr/bin/env python3
"""Top-K sketch theory, step 3: how many of a true neighbor's top-K terms are in
the query.

For every query and each of its ten exact ground-truth documents, the document row is
pruned as the engine prunes it (fp16 weights, mass ratio alpha, sparse/prune.h) and its
top-K terms are taken as the top-K-first slot writer takes them (QuantCSR::writeRow:
largest weight first, ties in dimension order). hits_K = query terms among those K. The
inline gate scores a neighbor
only when hits_K >= 1, and the gate is the same on every edge, so a true neighbor with
hits_K = 0 can only be found as a seed or an entry point: one minus that share is the
recall@10 ceiling of the gate. full = query terms in the whole pruned row, for contrast.
Per-pair hits, pruned nnz and query nnz also go to topk_gt_overlap_<ds>_pairs.npz
(plot_topk_overlap.py draws the document- and query-normalized overlap from them).

  module load python/3.11-24.1.0
  python topk_gt_overlap.py msmarco_full      # -> results/sketch_theory/topk_gt_overlap_<ds>.{md,json}
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dimension_overlap"))
from analyze_dimension_overlap import DATASET_DEFAULTS, BigAnnCSR, GroundTruth  # noqa: E402

KS = (16, 24, 32)  # 16 and 24: the inline layouts swept; 32: past line 0 (30 IDs)
GT_K = 10


def summary(hits, k):
    hist = np.bincount(hits, minlength=k + 1)[: k + 1] if k else np.bincount(hits)
    per_query = (hits.reshape(-1, GT_K) > 0).mean(axis=1)
    top1 = hits.reshape(-1, GT_K)[:, 0]
    return {"hist": hist.tolist(), "zero": float((hits == 0).mean()),
            "le1": float((hits <= 1).mean()), "le2": float((hits <= 2).mean()),
            "mean": float(hits.mean()), "top1_zero": float((top1 == 0).mean()),
            "recall_ceiling": float(per_query.mean()),
            "queries_all_hit": float((per_query == 1).mean())}


def main(ds):
    spec = DATASET_DEFAULTS[ds]
    d = Path(spec["directory"])
    alpha = np.float32(spec["alpha"])
    with BigAnnCSR(d / spec["base"]) as base, BigAnnCSR(d / spec["queries"]) as q, \
            GroundTruth(d / spec["gt"]) as gt:
        n = q.nrow * GT_K
        hits = {k: np.empty(n, np.int32) for k in KS}
        full = np.empty(n, np.int32)
        nnz = np.empty(n, np.int32)
        qnnz = np.empty(n, np.int32)
        mask = np.zeros(base.ncol, np.bool_)
        for qi in range(q.nrow):
            qd, _ = q.row(qi)
            mask[qd] = True
            for r in range(GT_K):
                dims, vals = base.row(int(gt.ids[qi, r]))
                w = np.asarray(vals, np.float16).astype(np.float32)
                order = np.lexsort((dims, -w))  # largest first, ties in dimension order
                prefix = np.cumsum(w[order], dtype=np.float32)
                kept = min(int(np.searchsorted(prefix, alpha * prefix[-1], side="left")) + 1, dims.size)
                inq = mask[dims[order[:kept]]]
                i = qi * GT_K + r
                full[i], nnz[i], qnnz[i] = inq.sum(), kept, qd.size
                for k in KS:
                    hits[k][i] = inq[:k].sum()
            mask[qd] = False
    out = {"dataset": ds, "alpha": float(alpha), "queries": int(q.nrow), "pairs": int(n),
           "mean_pruned_nnz": float(nnz.mean()), "short_rows": {k: float((nnz < k).mean()) for k in KS},
           "full": summary(full, 0)}
    out.update({"top%d" % k: summary(hits[k], k) for k in KS})
    res = HERE.parent / "results" / "sketch_theory"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"topk_gt_overlap_{ds}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(res / f"topk_gt_overlap_{ds}_pairs.npz", full=full, nnz=nnz, qnnz=qnnz,
                        **{f"hits{k}": hits[k] for k in KS})
    rows = [(f"top-{k}", out[f"top{k}"]) for k in KS] + [("full pruned row", out["full"])]
    md = [f"# Query terms among the top-K terms of true top-10 docs, {ds} (alpha {alpha:g})", "",
          f"{n:,} query/GT pairs, mean pruned nnz {nnz.mean():.1f}; rows shorter than "
          + " / ".join(map(str, KS)) + " terms: " + " / ".join(f"{out['short_rows'][k]:.1%}" for k in KS) + ".", "",
          "| terms | mean hits | 0 hits | <=1 | <=2 | top-1 with 0 | recall@10 ceiling | queries with all 10 |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {name} | {s['mean']:.2f} | {s['zero']:.2%} | {s['le1']:.2%} | {s['le2']:.2%} | "
           f"{s['top1_zero']:.2%} | {s['recall_ceiling']:.4f} | {s['queries_all_hit']:.2%} |" for name, s in rows]
    (res / f"topk_gt_overlap_{ds}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
