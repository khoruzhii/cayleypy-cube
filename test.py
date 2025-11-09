# test.py

import argparse
import json
import os
import time
import torch
from pilgrim import Pilgrim, Searcher
from pilgrim import generate_inverse_moves  # kept if you need inverse moves

def parse_skip_list(s):
    """Parse skip list provided as JSON (e.g., '[2,5]') or as comma-separated string '2,5'."""
    if s is None or str(s).strip() == "":
        return None
    s = str(s).strip()
    try:
        if s.startswith("["):
            return json.loads(s)
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Test Pilgrim Model")
    parser.add_argument("--group_id", type=int, required=True, help="Group ID.")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID.")
    parser.add_argument("--dataset", type=str, default="rnd", help="Dataset type: 'santa' or 'rnd'.")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load.")
    parser.add_argument("--B", type=int, default=2**18, help="Beam size.")
    parser.add_argument("--num_steps", type=int, default=200, help="Max steps per beam search run.")
    parser.add_argument("--tests_num", type=int, default=10, help="Number of tests to run.")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--verbose", type=int, default=0, help="Use tqdm if verbose > 0.")
    parser.add_argument("--shift", type=int, default=0, help="Shift part of the dataset.")
    parser.add_argument("--skip_list", type=str, help="IDs to skip, e.g. '[2, 5]' or '2,5'.")
    parser.add_argument("--cpu_beam", type=int, default=0, help="Store beam on CPU (0 or 1).")
    parser.add_argument("--no_path", type=int, default=0, help="Don't store path, only length (0 or 1).")

    args = parser.parse_args()
    args.skip_list = parse_skip_list(args.skip_list)
    args.cpu_beam = bool(args.cpu_beam)
    args.no_path = bool(args.no_path)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Load model info produced during training
    with open(f"{log_dir}/model_p{int(args.group_id):03d}-t{int(args.target_id):03d}_{args.model_id}.json", "r") as json_file:
        info = json.load(json_file)

    # Select device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] Start testing with device: {device}.")
    if args.cpu_beam:
        print(f"[{timestamp}] CPU beam mode enabled.")
    if args.no_path:
        print(f"[{timestamp}] No path mode enabled.")

    # Load group data (moves, names, target)
    with open(f"generators/p{int(args.group_id):03d}.json", "r") as f:
        data = json.load(f)
        if isinstance(data, dict) and "moves" in data and "move_names" in data:
            all_moves = data["moves"]
            move_names = data["move_names"]
        else:
            all_moves, move_names = data.values()
        all_moves = torch.tensor(all_moves, dtype=torch.int64, device=device)

    V0 = torch.load(
        f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt",
        weights_only=True,
        map_location=device,
    )

    # Derived parameters
    n_gens = all_moves.size(0)
    state_size = all_moves.size(1)
    num_classes = torch.unique(V0).numel()

    print("Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")

    # Inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)

    # Build model
    model = Pilgrim(
        num_classes=num_classes,
        state_size=state_size,
        hd1=info["hd1"],
        hd2=info["hd2"],
        nrd=info["nrd"],
        dropout_rate=info.get("dropout", 0.0),
    )

    # Load weights
    weights_path = f"weights/p{int(args.group_id):03d}-t{int(args.target_id):03d}_{args.model_id}_e{args.epoch:05d}.pth"
    state = torch.load(weights_path, weights_only=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Mixed precision: use float16 only on CUDA
    if device.type == "cuda":
        model.half()
        model.dtype = torch.float16
    else:
        model.dtype = torch.float32

    model.to(device)

    # Shift for negative labels if needed
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    # Load test dataset
    tests_path = f"datasets/p{int(args.group_id):03d}-t{int(args.target_id):03d}-{args.dataset}.pt"
    tests = torch.load(tests_path, weights_only=False, map_location=device)
    tests = tests[args.shift : args.shift + args.tests_num]
    args.tests_num = tests.size(0)
    print(f"Test dataset size: {args.tests_num}")

    # Initialize searcher
    searcher = Searcher(
        model=model, 
        all_moves=all_moves, 
        V0=V0, 
        device=device, 
        verbose=args.verbose,
        cpu_beam=args.cpu_beam
    )

    # Prepare log file path
    log_file_add = ""
    if args.shift > 0:
        log_file_add += f"_shift{args.shift}"
    if args.skip_list is not None:
        log_file_add += f"_skip{args.skip_list}"
    if args.cpu_beam:
        log_file_add += "_cpubeam"
    if args.no_path:
        log_file_add += "_nopath"
    
    log_file = (
        f"{log_dir}/test_p{int(args.group_id):03d}-t{int(args.target_id):03d}-"
        f"{args.dataset}_{args.model_id}_{args.epoch}_B{args.B}{log_file_add}.json"
    )

    results = []
    total_length = 0
    t1 = time.time()

    for i, state in enumerate(tests, start=0):
        # Skip by provided ids if needed
        if args.skip_list is not None and (i + args.shift) in args.skip_list:
            continue

        solution_time_start = time.time()
        moves, length = searcher.get_solution(
            state,
            B=args.B,
            num_steps=args.num_steps,
            no_path=args.no_path,
        )

        solution_time_end = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Compute average vertices per level (3 counters)
        vertex_num = searcher.counter[:, 0] / searcher.counter[:, 1]
        searcher.counter = torch.zeros((3, 2), dtype=torch.int64)

        if length is not None:
            total_length += length
            entry = {
                "test_num": i + args.shift,
                "solution_length": length,
                "time": round(solution_time_end - solution_time_start, 2),
                "moves": moves.tolist() if moves is not None else None,
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            print(f"[{timestamp}] Solution {i + args.shift}: Length = {length}")
        else:
            entry = {
                "test_num": i + args.shift,
                "solution_length": None,
                "time": round(solution_time_end - solution_time_start, 2),
                "moves": None,
                "vertex_num": f"[{vertex_num[0]:.2e}, {vertex_num[1]:.2e}, {vertex_num[2]:.2e}]",
            }
            print(f"[{timestamp}] Solution {i + args.shift} not found")

        results.append(entry)

        # Persist results after each test
        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)

    t2 = time.time()

    # Aggregate stats
    solved_results = [r for r in results if r["solution_length"] is not None]
    avg_length = (total_length / len(solved_results)) if solved_results else 0.0

    print(f"Test completed in {(t2 - t1):.2f}s.")
    print(f"Average solution length: {avg_length:.2f}.")
    print(f"Solved {len(solved_results)}/{args.tests_num} scrambles.")
    print(f"Results saved to {log_file}.")

if __name__ == "__main__":
    main()