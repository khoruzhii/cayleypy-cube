#!/usr/bin/env bash
set -euo pipefail

# ---------- defaults ----------
A_default=2            # builds count
TESTS_NUM_default=4    # tests_num for test.py
EPOCH_default=16       # train up to this epoch and test on it
B_default=65536        # parameter B for test.py

# ---------- external overrides ----------
A="${A:-${1:-$A_default}}"
TESTS_NUM="${TESTS_NUM:-${2:-$TESTS_NUM_default}}"
EPOCH="${EPOCH:-${3:-$EPOCH_default}}"
B="${B:-${4:-$B_default}}"

# ---------- single triple ----------
gid="054"
tid="0"
kmax="25"

# ---------- helpers ----------
train() {
    local gid="$1"      # group_id
    local tid="$2"      # target_id
    local kmax="$3"     # K_max

    python train.py \
        --group_id   "$gid" \
        --target_id  "$tid" \
        --epochs     "$EPOCH" \
        --hd1        1024 \
        --hd2        256 \
        --nrd        1 \
        --batch_size 10000 \
        --K_max      "$kmax" \
        --device_id  0
}

test() {
    local gid="$1"
    local tid="$2"
    local kmax="$3"

    local model_id
    model_id="$(grep -v '^$' logs/model_id.txt | tail -n 1 | tr -d '\r\n')"
    [[ -z "$model_id" ]] && { echo "model_id not found"; exit 1; }

    python test.py \
        --group_id     "$gid" \
        --target_id    "$tid" \
        --tests_num    "$TESTS_NUM" \
        --dataset      deepcubea \
        --epoch        "$EPOCH" \
        --model_id     "$model_id" \
        --num_steps    $((2 * kmax)) \
        --num_attempts 1 \
        --verbose      0 \
        --B            "$B" \
        --device_id    0
}

# ---------- main loop ----------
for ((i=1; i<=A; i++)); do
    echo "===== agent $i/$A   group_id=$gid   target_id=$tid   K_max=$kmax   epoch=$EPOCH   B=$B   tests_num=$TESTS_NUM ====="

    echo "[TRAIN]"
    train "$gid" "$tid" "$kmax"

    echo "[TEST ]"
    test  "$gid" "$tid" "$kmax"

    echo
done


python scripts/read-test-logs-multiagent.py "$A" "$EPOCH" "$B"