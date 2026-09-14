#!/bin/bash
#SBATCH --job-name=v2_splade_build
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Build the msmarco_v2 SPLADE++ ED dataset in the .csr/.gt format the sparse
# drivers read, from the pre-encoded corpus:
#   https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco_v2_passage_splade_pp_ed.tar
# and the pre-encoded topics from castorini/eval (topics/, formerly anserini-tools).
#
#   sbatch build_msmarco_v2_splade_perlmutter.sh
#   sbatch --export=ALL,LIMIT_SHARDS=100 build_msmarco_v2_splade_perlmutter.sh
#
# LIMIT_SHARDS=N converts only the first N shards (138,364 passages each), which
# is how you get a smaller subset to prototype on.

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts/msmarco_v2"
mkdir -p logs

module load python/3.11-24.1.0

DATA=${DATA:-$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_v2_splade}
WORKERS=${WORKERS:-64}
K=${K:-10}
BLOCK_ROWS=${BLOCK_ROWS:-250000}
QUERY_SETS=${QUERY_SETS:-"dev dev2 dl21 dl22 dl23"}
BASE=$DATA/base_v2_splade${LIMIT_SHARDS:+_${LIMIT_SHARDS}shard}.csr

echo "data=$DATA workers=$WORKERS k=$K base=$BASE"
df -h "$DATA" | tail -1

echo "########## queries ##########"
for q in $QUERY_SETS; do
    case $q in
        dev|dev2) topics=topics.msmarco-v2-passage.$q.splade-pp-ed.tsv.gz ;;
        *)        topics=topics.$q.splade-pp-ed.tsv.gz ;;
    esac
    python3 convert_splade_topics_to_csr.py \
        --topics "$DATA/$topics" --vocab "$DATA/bert_vocab.txt" \
        --out "$DATA/queries.$q.csr"
done

echo "########## corpus ##########"
python3 convert_splade_jsonl_to_csr.py \
    --shards "$DATA/msmarco_v2_passage_splade_pp_ed/*.jsonl.gz" \
    --vocab "$DATA/bert_vocab.txt" \
    --out "$BASE" \
    --workers "$WORKERS" \
    ${LIMIT_SHARDS:+--limit-shards "$LIMIT_SHARDS"}

echo "########## ground truth ##########"
for q in $QUERY_SETS; do
    python3 compute_gt_streaming.py \
        --base "$BASE" --queries "$DATA/queries.$q.csr" \
        --out "${BASE%.csr}.$q.gt" \
        --k "$K" --workers "$WORKERS" --block-rows "$BLOCK_ROWS"
done

echo "########## done ##########"
ls -lh "$DATA"/*.csr "$DATA"/*.gt
