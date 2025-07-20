#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <num_nodes> <node_num>"
  exit 1
fi

num_nodes="$1"
node_num="$2"
local_procs=8

[[ "$node_num" =~ ^[0-9]+$ ]] || { echo "node_num must be int"; exit 1; }
[[ "$num_nodes" =~ ^[0-9]+$ ]] || { echo "num_nodes must be int"; exit 1; }

world_size=$(( num_nodes * local_procs ))

for r in $(seq 0 $(( local_procs - 1 ))); do
  global_rank=$(( node_num * local_procs + r ))
  python voxtral_submit_audio.py -r "$global_rank" -w "$world_size" &
done

wait