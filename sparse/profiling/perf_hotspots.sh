#!/bin/bash
# E6b: Profile the non-distance time.
#
#   ./perf_hotspots.sh <command> [args...]
#   ./perf_hotspots.sh ../../build-release/bin/sparse_profile 16 200 150 1 0 0 0.8 3 \
#        base.csr queries.csr gt batch
set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
perf_hotspots.sh: `perf` is not installed on this cluster, so E6b (the ranked
list of where non-distance cycles go) cannot run here.

E6 still answers the question that matters -- HOW MUCH of the time is not the
distance kernel -- via the record/replay ablation in sparse_profile, which needs
no PMU at all. Only the breakdown of that remainder into named functions is
lost.
EOF
    exit 127
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkfifo "$WORK/ctl.fifo" "$WORK/ack.fifo"

PERF_CTL_FIFO="$WORK/ctl.fifo" PERF_ACK_FIFO="$WORK/ack.fifo" \
perf record -D -1 --control="fifo:$WORK/ctl.fifo,$WORK/ack.fifo" \
    -e cycles:u -F 999 --call-graph fp \
    -o "$WORK/search.data" -- "$@"

echo
echo "================ where the search phase's cycles go ================"
perf report -i "$WORK/search.data" --stdio --no-children --percent-limit 0.5 2>/dev/null \
    | grep -E '^\s+[0-9]+\.[0-9]+%' \
    | sed -e 's/  */ /g' \
    | head -25

echo
echo "(--no-children = self cycles. distanceDense() is the production kernel;"
echo " everything else -- priority_queue sift, visited bits, neighbor walks, the"
echo " per-query q_dense scatter -- is overhead that no amount of distance-kernel"
echo " optimization will remove.)"
