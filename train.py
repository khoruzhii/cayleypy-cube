import argparse
import torch
import os
import json
from pilgrim import Trainer, MLP
from pilgrim import count_parameters

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
        print(f"Created new model_id file and saved model_id: {model_id}")
    else:
        # Append the model_id to the file
        with open(model_id_file, "a") as f:
            f.write(f"{model_id}\n")
        print(f"Appended model_id: {model_id} to existing file")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train Pilgrim Model")
    
    # Training and architecture hyperparameters
    parser.add_argument("--m", type=int, help="Number of qubits for x")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--K_min", type=int, default=1, help="Minimum K value for random walks")
    parser.add_argument("--K_max", type=int, default=30, help="Maximum K value for random walks")
    parser.add_argument("--weights", type=str, default='', help="Path to file with model weights")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--layers", type=int, nargs='+', help="List of layer sizes for MLP")
    parser.add_argument("--gate_cost", type=int, nargs='+', default=[1, 1, 1], help="List of gate costs")
    parser.add_argument("--gate_probs", type=int, nargs='+', default=[1, 1, 1], help="List of gate probabilities for trainset generation")

    args = parser.parse_args()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device_id)
    print(f"Start training with {device}.")

    # Initialize the Pilgrim model
    model = MLP(layers=[2**args.m]+args.layers+[1]).to(device)
    
    if len(args.weights) > 0:
        model.load_state_dict(torch.load(f"weights/{args.weights}.pth", weights_only=True, map_location=device))
        print(f"Weights {args.weights} loaded.")

    # Calculate the number of model parameters
    num_parameters = count_parameters(model)
    param_million = round(num_parameters / 1_000_000)  # Get the number of parameters in millions

    # Create the training name based on mode, hidden layers, residual blocks, and number of parameters
    name = f"m{args.m:02d}_{args.layers}"
    
    # Create the trainer
    trainer = Trainer(m=args.m, net=model, device=device,
        batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs,
        name=name, K_min=args.K_min, K_max=args.K_max,
    )
    
    # Save the arguments to a log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    args_dict = vars(args)  # Convert the args object to a dictionary
    args_dict["model_name"] = name
    args_dict["model_id"] = trainer.id
    args_dict["num_parameters"] = num_parameters

    # Save the args and model information in JSON format
    with open(f"{log_dir}/model_{name}_{trainer.id}.json", "w") as f:
        json.dump(args_dict, f, indent=4)
        
    # Save model_id
    save_model_id(trainer.id)

    # Display model information
    print(f"Model Name: {name}")
    print(f'Model id: {trainer.id}')
    print(f"Model has {num_parameters} parameters")

    # Start the training process
    trainer.run()

if __name__ == "__main__":
    main()