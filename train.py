import argparse
import torch
import os
import json
from pilgrim import Trainer, Pilgrim
from pilgrim import count_parameters, generate_inverse_moves, load_cube_data

def save_model_id(model_id):
    # Ensure the logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Path to the model_id file
    model_id_file = os.path.join(log_dir, "model_id.txt")

    # Check if the file exists, if not create it and write the model_id
    if not os.path.exists(model_id_file):
        with open(model_id_file, "w") as f:
            f.write(f"{model_id}\n")
    else:
        # Append the model_id to the file
        with open(model_id_file, "a") as f:
            f.write(f"{model_id}\n")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Pilgrim Model")
    
    # Training and architecture hyperparameters
    parser.add_argument("--epochs", type=int, default=256, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "AdamSF"], default="Adam", help="Optimizer (Adam or AdamSF)")
    parser.add_argument("--activation", type=str, choices=["relu", "mish"], default="relu", help="Activation function (relu or mish)")
    parser.add_argument("--use_batch_norm", type=bool, default=True, help="Batch normalization usage (True or False, default True).")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    parser.add_argument("--weights", type=str, default='', help="Path to file with model weights.")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--alpha", type=float, default=1, help="TD-learning parameter, avg 1/α steps.")
    
    # Cube parameters
    parser.add_argument("--group_id", type=int, help="Group ID.")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID.")
    
    # Model architecture parameters
    parser.add_argument("--hd1", type=int, default=1024, help="Size of the first hidden layer")
    parser.add_argument("--hd2", type=int, default=256, help="Size of the second hidden layer (0 means no second layer)")
    parser.add_argument("--nrd", type=int, default=4, help="Number of residual blocks (0 means no residual blocks)")
    
    args = parser.parse_args()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device_id)
    print(f"Start training with {device}.")

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

    # Infer model mode based on hd1, hd2, and nrd
    if args.hd2 == 0 and args.nrd == 0:
        mode = "MLP1"
    elif args.hd2 > 0 and args.nrd == 0:
        mode = "MLP2"
    elif args.hd2 > 0 and args.nrd > 0:
        mode = "MLP2RB"
    else:
        raise ValueError("Invalid combination of hd1, hd2, and nrd.")

    # Initialize the Pilgrim model
    model = Pilgrim(
        num_classes=num_classes,
        state_size=state_size,
        hd1=args.hd1,
        hd2=args.hd2,
        nrd=args.nrd,
        dropout_rate=args.dropout,
        activation_function=args.activation
    ).to(device)
    
    if len(args.weights) > 0:
        model.load_state_dict(torch.load(f"weights/{args.weights}.pth", weights_only=True, map_location=device))
        print(f"Weights {args.weights} loaded.")

    # Calculate the number of model parameters
    num_parameters = count_parameters(model)
    param_million = round(num_parameters / 1_000_000)  # Get the number of parameters in millions

    # Create the training name based on mode, hidden layers, residual blocks, and number of parameters
    name = f"p{int(args.group_id):03d}-t{int(args.target_id):03d}"
    
    # Create the trainer
    trainer = Trainer(
        net=model,
        num_epochs=args.epochs,
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        name=name,
        K_min=args.K_min,
        K_max=args.K_max,
        all_moves=all_moves,
        inverse_moves=inverse_moves,
        V0=V0,
        α=args.alpha
    )
    
    # Save the arguments to a log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args)  # Convert the args object to a dictionary
    args_dict["model_name"] = name
    args_dict["model_mode"] = mode
    args_dict["model_id"] = trainer.id
    args_dict["num_parameters"] = num_parameters

    # Save the args and model information in JSON format
    with open(f"{log_dir}/model_{name}_{trainer.id}.json", "w") as f:
        json.dump(args_dict, f, indent=4)
        
    # Save model_id
    save_model_id(trainer.id)

    # Display model information
    print("Model info:")
    print(f"  mode          {mode}")
    print(f"  name          {name}")
    print(f'  id            {trainer.id}')
    print(f"  # parameters  {num_parameters:_}")

    # Start the training process
    trainer.run()

if __name__ == "__main__":
    main()
