#!/bin/bash
# Parses the recall/timing lines that main.cpp, grassRMA_main.cpp, and
# sindi_main.cpp all print in the same format --
#   "Added N points to the index in T microseconds."
#   "Total Query time: T microseconds"
#   "Recall@k: R%"
# -- out of a demo binary's log, and appends one row to a shared
# competitor-results CSV (method, params, dataset_size, threads,
# indexing_time_sec, searching_time_sec, recall). Writing the header only
# the first time so multiple competitors can share one file.
#
# Usage: append_competitor_result.sh <log_file> <method> <params> <threads> <csv_out>

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <log_file> <method> <params> <threads> <csv_out>" >&2
  exit 1
fi

LOG="$1"; METHOD="$2"; PARAMS="$3"; THREADS="$4"; CSV="$5"

dataset_size=$(grep -oP 'Added \K[0-9]+(?= points to the index)' "$LOG" | head -1)
indexing_us=$(grep -oP 'points to the index in \K[0-9]+(?= microseconds)' "$LOG" | head -1)
search_us=$(grep -oP 'Total Query time: \K[0-9]+(?= microseconds)' "$LOG" | head -1)
recall=$(grep -oP 'Recall@k: \K[0-9.]+(?=%)' "$LOG" | head -1)

if [ -z "${dataset_size:-}" ] || [ -z "${indexing_us:-}" ] || [ -z "${search_us:-}" ] || [ -z "${recall:-}" ]; then
  echo "append_competitor_result.sh: could not parse one or more fields from $LOG -- not appending (check the log for a crash)." >&2
  echo "  dataset_size='${dataset_size:-}' indexing_us='${indexing_us:-}' search_us='${search_us:-}' recall='${recall:-}'" >&2
  exit 1
fi

indexing_sec=$(echo "scale=6; $indexing_us/1000000" | bc -l)
search_sec=$(echo "scale=6; $search_us/1000000" | bc -l)

mkdir -p "$(dirname "$CSV")"
if [ ! -s "$CSV" ]; then
  echo "method,params,dataset_size,threads,indexing_time_sec,searching_time_sec,recall" > "$CSV"
elif [ -n "$(tail -c1 "$CSV")" ]; then
  # Last append left no trailing newline -- fix it up first or the next row
  # concatenates onto the same line instead of starting a new one.
  echo >> "$CSV"
fi
echo "${METHOD},\"${PARAMS}\",${dataset_size},${THREADS},${indexing_sec},${search_sec},${recall}" >> "$CSV"
echo "Appended ${METHOD} result to $CSV"
