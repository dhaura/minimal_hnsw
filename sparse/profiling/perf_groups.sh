#!/bin/bash
# E2: run the hardware-counter groups against a binary that marks its region of
# interest with perf_ctl (sparse_profile: search phase; bench_distance: the
# selected variant's kernel; bench_scale: the dense kernel).
#
#   ./perf_groups.sh <command> [args...]
#   ./perf_groups.sh taskset -c 8 ../../build-release/bin/bench_distance base.csr q.csr 2000000 3
#
set -u

if [ $# -lt 1 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 1
fi

if ! command -v perf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
perf_groups.sh: `perf` is not on $PATH, so E2 (hardware counters) and E6b
(cycle hotspots) cannot run here. Grace has no perf at all; Perlmutter DOES
(/usr/bin/perf), so on Perlmutter this message means the environment is wrong,
not that the rung is impossible.

The rest of the ladder does not depend on them: E3/E4 (ablation, working-set
sweep), E5 (thread scaling) and E1/E3b/E6 (the real driver) are all wall-clock
and in-code counter based by design, precisely so the analysis survives a
machine with no PMU access.
EOF
    exit 127
fi

#   G1 is it memory at all:  IPC + the demand-load miss ladder
#   G2 where fills come from: local vs remote DRAM (the NUMA diagnosis)
#   G3 translation + speculation: page walks by page size + branch mispredicts
EVG_NAMES=(G1_overview G2_fill_source G3_tlb_branch)

if grep -qi 'AuthenticAMD' /proc/cpuinfo 2>/dev/null; then
    CPU_EVENTS=AMD_Zen   # Perlmutter: EPYC 7763 / 7713 (Zen3)
    EVG_EVENTS=(
      "cycles:u,instructions:u,ls_mab_alloc.loads:u,ls_dmnd_fills_from_sys.mem_io_local:u,ls_dmnd_fills_from_sys.lcl_l2:u"
      "cycles:u,ls_dmnd_fills_from_sys.mem_io_local:u,ls_dmnd_fills_from_sys.mem_io_remote:u,ls_dmnd_fills_from_sys.ext_cache_local:u,ls_dmnd_fills_from_sys.ext_cache_remote:u"
      "cycles:u,ls_l1_d_tlb_miss.all:u,ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss:u,ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss:u,ex_ret_brn_misp:u"
    )
else
    CPU_EVENTS=Intel     # Grace: Xeon Gold 6248R (Cascade Lake)
    EVG_EVENTS=(
      "cycles:u,instructions:u,mem_load_retired.l1_miss:u,mem_load_retired.l2_miss:u,mem_load_retired.l3_miss:u"
      "cycles:u,mem_load_l3_miss_retired.local_dram:u,mem_load_l3_miss_retired.remote_dram:u,mem_load_l3_miss_retired.remote_hitm:u,mem_load_l3_miss_retired.remote_fwd:u"
      "cycles:u,dtlb_load_misses.miss_causes_a_walk:u,dtlb_load_misses.walk_completed_4k:u,dtlb_load_misses.walk_completed_2m_4m:u,br_misp_retired.all_branches:u"
    )
fi
echo "perf_groups.sh: using $CPU_EVENTS event set" >&2

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
