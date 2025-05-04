#!/usr/bin/env bash
set -euo pipefail

# init tmp dir
mkdir -p ./tmp
export TMPDIR="$(pwd)/tmp"

# ---------- helpers ----------
train() {
    local gid="$1"      # group_id
    local tid="$2"      # target_id
    local kmax="$3"     # K_max

    python train.py \
        --group_id  "$gid" \
        --target_id "$tid" \
        --epochs    128 \
        --hd1       1024 \
        --hd2       256 \
        --nrd       1 \
        --batch_size 10000 \
        --K_max     "$kmax" \
        --device_id 1
}

test() {
    local gid="$1"
    local tid="$2"
    local kmax="$3"

    local model_id
    model_id="$(grep -v '^$' logs/model_id.txt | tail -n 1 | tr -d '\r\n')"

    python test.py \
        --group_id   "$gid" \
        --target_id  "$tid" \
        --tests_num  10 \
        --dataset    rnd \
        --epoch      16 \
        --model_id   "$model_id" \
        --num_steps  $((2 * kmax)) \
        --num_attempts 1 \
        --verbose    0 \
        --B          1048576 \
        --device_id 1

    python test.py \
        --group_id   "$gid" \
        --target_id  "$tid" \
        --tests_num  10 \
        --dataset    rnd \
        --epoch      128 \
        --model_id   "$model_id" \
        --num_steps  $((2 * kmax)) \
        --num_attempts 1 \
        --verbose    0 \
        --B          1048576 \
        --device_id 0

}

# ---------- job list ----------
triples=(
  "026 0 30"
  "027 0 80"
  "028 0 150"
  "029 0 200"
  "034 0 50"
  "035 0 100"
  "036 0 200"
  "037 0 300"
  "044 0 15"
  "045 0 25"
  "046 0 30"
  "047 0 40"
  "048 0 45"
  "049 0 55"
  "050 0 60"
  "030 0 250"
  "031 0 300"
  "038 0 300"
  "051 0 65"
  "052 0 70"
)

# ---------- main loop ----------
for triple in "${triples[@]}"; do
    IFS=' ' read -r gid tid kmax <<< "$triple"
    echo "===== group_id=$gid  target_id=$tid  K_max=$kmax ====="

    echo "[TRAIN]"
    train "$gid" "$tid" "$kmax"

    echo "[TEST ]"
    test  "$gid" "$tid" "$kmax"

    echo
done