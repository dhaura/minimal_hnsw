#!/bin/bash
# E2: run the Zen3 counter groups against a binary that marks its region of
# interest with perf_ctl (sparse_profile: search phase; bench_distance: the
# selected variant's kernel; bench_scale: the merge kernel).
#
#   ./perf_groups.sh <command> [args...]
#   ./perf_groups.sh taskset -c 8 ../../build/bin/sparse_profile 16 200 150 ... batch
#
# Groups are capped at 5 events: Zen3 has 6 core PMCs and perf starts
# multiplexing (rescaling = estimating) at 6. Everything is :u because
# perf_event_paranoid=2 on Perlmutter. If the program prints
# "PROF ndist=<N>", counters are also normalized per distance call.
set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
fi

# G1 is it memory at all: IPC + L1 miss rate + demand DRAM/L2 fills
# G2 where fills come from: local/remote DRAM (NUMA), same/other-CCX cache
# G3 translation + speculation: page walks (per page size) + branch mispredicts
EVG_NAMES=(G1_overview G2_fill_source G3_tlb_branch)
EVG_EVENTS=(
  "cycles:u,instructions:u,ls_mab_alloc.loads:u,ls_dmnd_fills_from_sys.mem_io_local:u,ls_dmnd_fills_from_sys.lcl_l2:u"
  "cycles:u,ls_dmnd_fills_from_sys.mem_io_local:u,ls_dmnd_fills_from_sys.mem_io_remote:u,ls_dmnd_fills_from_sys.ext_cache_local:u,ls_dmnd_fills_from_sys.ext_cache_remote:u"
  "cycles:u,ls_l1_d_tlb_miss.all:u,ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss:u,ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss:u,ex_ret_brn_misp:u"
)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for gi in "${!EVG_NAMES[@]}"; do
    name=${EVG_NAMES[$gi]}
    events=${EVG_EVENTS[$gi]}
    echo
    echo "================ $name ================"
    mkfifo "$WORK/ctl.fifo" "$WORK/ack.fifo"

    PERF_CTL_FIFO="$WORK/ctl.fifo" PERF_ACK_FIFO="$WORK/ack.fifo" \
    perf stat -D -1 --control="fifo:$WORK/ctl.fifo,$WORK/ack.fifo" \
        -x, -o "$WORK/$name.csv" -e "$events" -- "$@" | tee "$WORK/$name.out"

    rm -f "$WORK/ctl.fifo" "$WORK/ack.fifo"

    ndist=$(grep -oE 'PROF ndist=[0-9]+' "$WORK/$name.out" | tail -1 | cut -d= -f2)
    echo "---- counts (gated region only) ----"
    awk -F, -v nd="${ndist:-0}" '
        /^#/ || NF < 3 { next }
        {
            val = $1; evt = $3
            if (val == "<not counted>" || val == "<not supported>") {
                printf "%20s  %-55s\n", val, evt; next
            }
            if (nd > 0) printf "%20d  %-55s %12.3f /dist-call\n", val, evt, val / nd
            else        printf "%20d  %-55s\n", val, evt
        }' "$WORK/$name.csv"
    [ -n "${ndist:-}" ] && echo "(normalized by ndist=$ndist from program output)"
done
