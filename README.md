# Zero-Human-Knowledge NN Training with Random Walks for Puzzle Solving

This program (as part of the CayleyPy project) provides tools to train and test models capable of solving NxNxN Rubik's cubes. It includes two main scripts: `train.py` for training a model and `test.py` for testing cube solutions. 

**`train.py`** trains a model to predict the diffusion distance, which is calculated using random walks from the solved cube state. The diffusion distance serves as a metric that creates a good ordering between cube states, simplifying the search for solutions.

Once the model is trained, **`test.py`** can be used to solve Rubik's cubes using beam search. The model's solutions — including solution lengths and the number of attempts — are logged in result files.

Within 10 minutes of training, the model will be able to:
- Solve a 3x3x3 cube in seconds.
- Solve a 4x4x4 cube in under a minute.

This approach allows for efficient and quick solutions for Rubik's cubes of various sizes, using a model trained to predict diffusion distances. The proposed method achieves state-of-the-art (SOTA) results in solving Rubik's cubes with high efficiency and speed.



## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/k1242/pilgrim.git
    cd pilgrim
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Training Script

You can run the `train.py` script to train a model on cube-based data. Each epoch model see 1M cubes sampled with K ∈ \[1, K_max\]. The model architecture is flexible, allowing different hidden layer sizes and residual blocks to be used.

### Basic Usage

```bash
python train.py --cube_size 4 --cube_type all --K_max 48 --hd1 1000 --hd2 500 --nrd 2 --epochs 256
```

### Parameters:

*   `--hd1`: Size of the first hidden layer (e.g., `2000`).
*   `--hd2`: Size of the second hidden layer (`0` means no second layer).
*   `--nrd`: Number of residual blocks (`0` means no residual blocks).
*   `--epochs`: Number of training epochs.
*   `--batch_size`: Batch size (default `10000`).
*   `--lr`: Learning rate for the optimizer (default `0.001`).
*   `--optimizer`: Optimizer, `Adam` or `AdamSF` means schedulefree module use (default `Adam`).
*   `--activation`: Activation function, `relu` or `mish` (default `relu`).
*   `--use_batch_norm`: Batch normalization usage, `True` or `False` (default `True`).
*   `--K_min` and `--K_max`: Minimum and maximum values for random walks (default `1` and `30`).
*   `--cube_size`: Cube size (e.g., `3` for 3x3x3 or `4` for 4x4x4).
*   `--cube_type`: Cube move set (`qtm` for quarter-turn metric or `all` for all moves).
*   `--device_id`: Device ID to use different graphics card (default `0`).


#### Output

When you run the training script, the following output and files are generated:

1. **Training Logs (CSV)**:
    - A CSV file is created in the `logs/` directory that tracks the training progress. This file logs the following information for each epoch:
        - `epoch`: The current epoch number.
        - `train_loss`: The loss value at the end of each epoch.
        - `vertices_seen`: The number of vertices (data points) seen in each epoch.
        - `data_gen_time`: Time taken to generate the training data for the current epoch.
        - `train_epoch_time`: Time taken to complete the training step for the current epoch.
    - The file is saved with the naming convention: `train_{model_name}_{model_id}.csv`.

2. **Model Weights (Checkpoint Files)**:
    - The model weights are saved periodically during training:
        - **Power of Two Epochs**: Weights are saved at epochs that are powers of two (e.g., epoch 1, 2, 4, 8, 16, ...). These weights are saved in the `weights/` directory with the filename:
          `weights/{model_name}_{model_id}_e{epoch}.pth`.
        - **Epoch 10,000 and 50,000**: If training reaches these epochs, weights are saved with the filename:
          `weights/{model_name}_{model_id}_e10000.pth` and `weights/{model_name}_{model_id}_e50000.pth`.
        - **Final Weights**: After the final epoch, the weights are saved with the filename:
          `weights/{model_name}_{model_id}_e{final_epoch}_final.pth`.

#### Model Name Generation

The `model_name` is automatically generated based on the cube size, cube type, model architecture, and the number of parameters in the model. This helps uniquely identify the model being trained and is used for logging and saving model weights.

The `model_name` is constructed using the following format:

~~~~text
cube{cube_size}_{cube_type}_{mode}_{num_parameters}M
~~~~

Where:

*   **`mode`**: The architecture of the model, determined by the following:
    *   `"MLP1"`: When both `hd2=0` and `nrd=0`.
    *   `"MLP2"`: When `hd2>0` and `nrd=0`.
    *   `"MLP2RB"`: When `hd2>0` and `nrd>0` (i.e., when residual blocks are included).
*   **`num_parameters`**: The total number of trainable parameters in the model, rounded to millions (`M`).

## Testing the Model

You can test a trained **Pilgrim** model using the `test.py` script. This script loads the model, applies it to a set of cube states, and attempts to solve them using a beam search.

### Basic Usage

~~~~bash
python test.py --cube_size 4 --cube_type all --weights weights/cube4_all_MLP2_01M_1728177387_e00256.pth --tests_num 10 --B 65536
~~~~

### Parameters

*   `--cube_size`: The size of the cube (e.g., `4` for 4x4x4 cube).
*   `--cube_type`: The cube type, either `qtm` (quarter-turn metric) or `all` (all moves).
*   `--weights`: Path to the saved model weights.
*   `--tests`: Path to the test dataset (optional). If not provided, it defaults to the dataset in `datasets/{cube_type}_cube{cube_size}.pt`.
*   `--B`: Beam size for the beam search (default `4096`).
*   `--tests_num`: Number of test cases to run (default `10`).
*   `--device_id`: Device ID to use different graphics card.
*   `--verbose`: Each step of beam search printed as tqdm, default is 0.


#### Output


*   **Log File**: The test results, including solution lengths and attempts, are saved to a log file in the `logs/` directory. The log file is named based on the model name, model ID, epoch, and beam size:

    ~~~~text
    logs/test_{model_name}_{model_id}_{epoch}_B{beam_size}.json
    ~~~~

    The log file contains the following information for each test case:
    *   `test_num`: The index of the test case.
    *   `solution_length`: The number of moves in the solution (if found).
    *   `attempts`: The number of attempts made by the searcher to solve the test case.
    *   `moves`: The sequence of moves for solving the cube, stored as a list.
    
    If no solution is found, the `solution_length`, `attempts`, and `moves` will be set to `None`.
   

## Support Us

If you find this project helpful or interesting, please consider giving it a ⭐. It helps others discover the project and motivates to keep improving it. Thank you for your support!

