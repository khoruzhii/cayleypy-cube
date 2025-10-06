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
        --device_id 0
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
        --tests_num  100 \
        --dataset    santa \
        --epoch      16 \
        --model_id   "$model_id" \
        --num_steps  $((2 * kmax)) \
        --num_attempts 1 \
        --verbose    0 \
        --B          1048576 \
        --device_id 0

    python test.py \
        --group_id   "$gid" \
        --target_id  "$tid" \
        --tests_num  100 \
        --dataset    santa \
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
  "011 0 10"
  "012 0 10"
  "013 0 20"
  "014 0 35"
  "015 0 75"
  "017 0 45"  "017 1 60"
  "018 0 110"
  "019 0 25"
  "020 0 30"  "020 1 40"
  "021 0 40"
  "022 0 165"
  "023 0 170"
  "024 0 500"
  "025 0 700"
  "000 0 15"  "000 1 15"  "000 2 15"
  "001 0 25"  "001 1 25"  "001 2 25"
  "002 0 60"  "002 1 60"  "002 2 60"
  "003 0 70"  "003 1 70"  "003 2 70"
  "004 0 100" "004 1 150"
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