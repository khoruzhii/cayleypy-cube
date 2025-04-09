import argparse
import torch
import os
import json
import time
from pilgrim import Pilgrim, Searcher
from pilgrim import count_parameters, generate_inverse_moves, load_cube_data

def main():
    parser = argparse.ArgumentParser(description="Test Pilgrim Model")
    parser.add_argument("--group_id", type=int, help="Group ID.")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID.")
    parser.add_argument("--dataset", type=str, default='rnd', help="Type of dataset, 'santa' or 'rnd'.")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID.")
    parser.add_argument("--epoch", type=int, required=True, help="Number of epochs to train model.")
    parser.add_argument("--B", type=int, default=2**18, help="Beam size")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of allowed restarts.")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of allowed steps in one beam search run.")
    parser.add_argument("--tests_num", type=int, default=10, help="Number of tests to run")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--verbose", type=int, default=0, help="Use tqdm if verbose > 0.")
    parser.add_argument("--shift", type=int, default=0, help="Shift part of the dataset.")
    parser.add_argument("--skip_list", type=str, help="List of ids, that should be skipped, e.g. '[2, 5]'.")
    parser.add_argument("--return_tree", type=int, default=0, help="Save beam seach tree to 'forest' folder.")
    
    args = parser.parse_args()
    if args.skip_list is not None:
        args.skip_list = eval(args.skip_list)

    log_dir = "logs"
    forest_dir = "forest"
    
    # Load model info
    with open(f"{log_dir}/model_p{int(args.group_id):03d}-t{int(args.target_id):03d}_{args.model_id}.json", "r") as json_file:
        info = json.load(json_file)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device_id)
#     device = torch.device("cpu")
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"[{timestamp}] Start testing with {device}.")

    # Load group data (moves, names, target)
    with open(f'generators/p{int(args.group_id):03d}.json', 'r') as f:
        all_moves, move_names = json.load(f).values()
        all_moves = torch.tensor(all_moves, dtype=torch.int64, device=device)
    V0 = torch.load(f"targets/p{int(args.group_id):03d}-t{int(args.target_id):03d}.pt", weights_only=True, map_location=device)

    # Derive important group parameters from the loaded data
    n_gens = all_moves.size(0)  # Number of moves
    state_size = all_moves.size(1)  # Size of the state representation
    num_classes = torch.unique(V0).size(0)
    print(f"Group info:")
    print(f"  # generators   {n_gens}")
    print(f"  # classes      {num_classes}")
    print(f"  state size     {state_size}")

    # Generate inverse moves
    inverse_moves = torch.tensor(generate_inverse_moves(move_names), dtype=torch.int64, device=device)

    # Load model and weights
    model = Pilgrim(num_classes=num_classes, state_size=state_size, 
                    hd1=info['hd1'], hd2=info['hd2'], nrd=info['nrd'], 
                    activation_function=info.get('activation', 'relu'), 
                    use_batch_norm=info.get('use_batch_norm', True))
    model.load_state_dict(torch.load(
        f"weights/p{int(args.group_id):03d}-t{int(args.target_id):03d}_{args.model_id}_e{args.epoch:05d}.pth", 
        weights_only=False, map_location=device
    ))
    model.eval()
    
    # Fix float16
    model = model.half()
    model.dtype = torch.float16
    
    # Load test dataset
    tests_path = f"datasets/p{int(args.group_id):03d}-t{int(args.target_id):03d}-{args.dataset}.pt"
    tests = torch.load(tests_path, weights_only=False, map_location=device)
    tests = tests[args.shift:args.shift+args.tests_num]
    args.tests_num = tests.size(0)
    print(f"Test dataset size: {args.tests_num}")

    # Initialize Searcher object
    searcher = Searcher(model=model, all_moves=all_moves, V0=V0, device=device, verbose=args.verbose)

    # Prepare log file
    os.makedirs(log_dir, exist_ok=True)
    log_file_add = ""
    if args.shift > 0:
        log_file_add = log_file_add + f"_shift{args.shift}"
    if args.skip_list is not None:
        log_file_add = log_file_add + f"_skip{args.skip_list}"
    log_file = f"{log_dir}/test_p{int(args.group_id):03d}-t{int(args.target_id):03d}-{args.dataset}_{args.model_id}_{args.epoch}_B{args.B}{log_file_add}.json"

    results = []
    total_length = 0
    t1 = time.time()
    for i, state in enumerate(tests, start=0):
        if args.skip_list is not None and i+args.shift in args.skip_list:
            continue
        solution_time_start = time.time()
        result = searcher.get_solution(
            state, B=args.B, 
            num_steps=args.num_steps, num_attempts=args.num_attempts, 
            return_tree=args.return_tree
        )
        moves, attempts = result[:2]
        if args.return_tree and moves is not None:
            tree = result[2]
            os.makedirs(forest_dir, exist_ok=True)
            torch.save(tree.cpu(), f"{forest_dir}/tree_{args.cube_type}_{args.cube_size}_i{i+args.shift:04d}_B{args.B:08d}_{info['model_id']}.pt")  
            torch.save(state.cpu(), f"{forest_dir}/state_{args.cube_type}_{args.cube_size}_i{i+args.shift:04d}_B{args.B:08d}_{info['model_id']}.pt") 
    
        solution_time_end = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        if moves is not None:
            solution_length = len(moves)
            total_length += solution_length
            
            result = {
                "test_num": i+args.shift,
                "solution_length": solution_length,
                "attempts": attempts + 1,
                "time": round(solution_time_end - solution_time_start, 2),
                "moves": moves.tolist()
            }
            
            # Print solution length for each solved cube
            print(f"[{timestamp}] Solution {i+args.shift}: Length = {solution_length}")
        else:
            # If no solution is found
            result = {
                "test_num": i+args.shift,
                "solution_length": None,
                "attempts": None,
                "time": round(solution_time_end - solution_time_start, 2),
                "moves": None
            }
            print(f"[{timestamp}] Solution {i+args.shift} not found")
        
        results.append(result)

        # Append new result to the log file
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=4)

    t2 = time.time()

    # Calculate average solution length
    solved_results = [r for r in results if r["solution_length"] is not None]
    avg_length = total_length / len(solved_results) if solved_results else 0

    # Print completion message with average solution length
    print(f"Test completed in {(t2 - t1):.2f}s.")
    print(f"Average solution length: {avg_length:.2f}.")
    print(f"Solved {len(solved_results)}/{args.tests_num} cubes.")
    print(f"Results saved to {log_file}.")

if __name__ == "__main__":
    main()