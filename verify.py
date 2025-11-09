# verify.py

import argparse
import json
import os
import re
import sys
import torch


def parse_log_filename(log_path):
    """Extract group_id, target_id, dataset from test log filename."""
    filename = os.path.basename(log_path)

    # Example:
    # test_p001-t000-rnd_3_00050_B262144.json
    # test_p001-t000-rnd_3_00050_B262144_shift10_skip[2, 5].json
    pattern = r"^test_p(\d+)-t(\d+)-([^-_]+)_(\d+)_(\d+)_B(\d+)"
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(
            "Cannot parse log filename. Expected format like "
            "'test_p001-t000-rnd_3_00050_B262144.json', got: '{}'".format(filename)
        )

    group_id = int(m.group(1))
    target_id = int(m.group(2))
    dataset = m.group(3)
    model_id = int(m.group(4))
    epoch = int(m.group(5))
    B = int(m.group(6))

    return {
        "group_id": group_id,
        "target_id": target_id,
        "dataset": dataset,
        "model_id": model_id,
        "epoch": epoch,
        "B": B,
    }


def load_moves(group_id):
    """Load generators and move_names from generators/pXXX.json."""
    path = "generators/p{:03d}.json".format(group_id)
    if not os.path.exists(path):
        raise FileNotFoundError("Generators file not found: {}".format(path))

    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict) and "moves" in data and "move_names" in data:
            all_moves = data["moves"]
            move_names = data["move_names"]
        else:
            # Backward-compatible: assume two values in order (moves, move_names)
            all_moves, move_names = data.values()

    all_moves = torch.tensor(all_moves, dtype=torch.int64, device="cpu")
    return all_moves, move_names


def load_target(group_id, target_id):
    """Load target state V0 from targets/pXXX-tYYY.pt."""
    path = "targets/p{:03d}-t{:03d}.pt".format(group_id, target_id)
    if not os.path.exists(path):
        raise FileNotFoundError("Target file not found: {}".format(path))

    V0 = torch.load(path, weights_only=True, map_location="cpu")
    return V0


def load_tests(group_id, target_id, dataset):
    """Load full test dataset from datasets/pXXX-tYYY-dataset.pt."""
    path = "datasets/p{:03d}-t{:03d}-{}.pt".format(group_id, target_id, dataset)
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset file not found: {}".format(path))

    tests = torch.load(path, weights_only=False, map_location="cpu")
    return tests


def apply_moves(state, moves, all_moves):
    """Apply sequence of generator indices to a single state."""
    # state: 1D tensor [state_size]
    # all_moves: [n_gens, state_size] of indices (permutations)
    s = state.clone()
    n_gens = all_moves.size(0)

    for m in moves:
        if m < 0 or m >= n_gens:
            raise ValueError("Invalid move index {} (n_gens = {})".format(m, n_gens))
        perm = all_moves[m]
        s = s[perm]

    return s


def verify_log(log_path):
    """Verify that recorded move sequences indeed reach V0 from initial states."""
    info = parse_log_filename(log_path)
    group_id = info["group_id"]
    target_id = info["target_id"]
    dataset = info["dataset"]

    print("Verifying log:")
    print("  file      {}".format(log_path))
    print("  group_id  {}".format(group_id))
    print("  target_id {}".format(target_id))
    print("  dataset   {}".format(dataset))

    all_moves, move_names = load_moves(group_id)
    V0 = load_target(group_id, target_id)
    tests = load_tests(group_id, target_id, dataset)

    with open(log_path, "r") as f:
        results = json.load(f)

    total = 0
    verified_ok = 0
    not_found = 0
    mismatch = 0
    out_of_range = 0
    errors = 0

    for entry in results:
        total += 1
        test_num = entry.get("test_num")
        moves = entry.get("moves")

        if test_num is None:
            print("Entry without test_num, skipping.")
            errors += 1
            continue

        if not isinstance(test_num, int):
            print("Entry with non-integer test_num {}, skipping.".format(test_num))
            errors += 1
            continue

        if test_num < 0 or test_num >= len(tests):
            print("Test {}: index out of range for dataset (size {}).".format(test_num, len(tests)))
            out_of_range += 1
            continue

        state0 = tests[test_num]

        if moves is None:
            # No solution recorded (search did not find path)
            not_found += 1
            continue

        if not isinstance(moves, list):
            print("Test {}: moves is not a list, skipping.".format(test_num))
            errors += 1
            continue

        try:
            state_final = apply_moves(state0, moves, all_moves)
        except Exception as e:
            print("Test {}: error applying moves: {}".format(test_num, e))
            errors += 1
            continue

        if state_final.shape != V0.shape:
            print("Test {}: shape mismatch: final {} vs V0 {}.".format(
                test_num, tuple(state_final.shape), tuple(V0.shape)
            ))
            mismatch += 1
            continue

        if torch.equal(state_final, V0):
            verified_ok += 1
        else:
            mismatch += 1
            print("Test {}: mismatch between final state and V0.".format(test_num))

    print()
    print("Verification summary:")
    print("  total entries     {}".format(total))
    print("  verified ok       {}".format(verified_ok))
    print("  not found (moves=None) {}".format(not_found))
    print("  mismatches        {}".format(mismatch))
    print("  out of range      {}".format(out_of_range))
    print("  errors            {}".format(errors))

    if mismatch == 0 and errors == 0 and out_of_range == 0:
        print("All recorded solutions are consistent with V0.")
    else:
        print("Some recorded solutions are inconsistent with V0 or could not be verified.")


def main():
    parser = argparse.ArgumentParser(description="Verify test log solutions against V0.")
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to test log JSON file (e.g. logs/test_p001-t000-rnd_3_00050_B262144.json).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print("Log file does not exist: {}".format(args.log_file))
        sys.exit(1)

    verify_log(args.log_file)


if __name__ == "__main__":
    main()
