#!/bin/bash
# E6b: Profile the non-distance time.
#
#   ./perf_hotspots.sh <command> [args...]
#   ./perf_hotspots.sh ../../build/bin/sparse_profile 16 200 150 1 0 0 1 0 \
#        base.csr queries.csr gt batch
set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
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
echo "(--no-children = self cycles. distance() is the merge kernel; everything"
echo " else -- priority_queue sift, visited bits, neighbor walks -- is overhead"
echo " that no amount of distance-kernel optimization will remove.)"
