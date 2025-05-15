# Machine Learning-based Puzzle Solver

This code provides a novel approach using machine learning to solve puzzles represented as large graphs, such as Rubik’s cubes, using neural networks trained to estimate diffusion distances. Solutions are found using beam search.

## Overview

The approach employs trained neural networks to efficiently navigate puzzle state spaces, significantly outperforming traditional algorithms and other ML methods. Specifically, the system:

* Uses beam search guided by neural network predictions.
* Achieves state-of-the-art results on Rubik’s cubes (3x3x3, 4x4x4, 5x5x5).

## Available Groups and Kmax

| Group ID | Puzzle                        | Kmax used |
| -------- | ----------------------------- | --------- |
| 000      | Cube 2x2x2                    | 15        |
| 001      | Cube 3x3x3                    | 26        |
| 002      | Cube 4x4x4                    | 45        |
| 003      | Cube 5x5x5                    | 65        |
| 004      | Cube 6x6x6                    | 150       |
| 011      | Wreath 6/6                    | 10        |
| 012      | Wreath 7/7                    | 10        |
| 013      | Wreath 12/12                  | 20        |
| 014      | Wreath 21/21                  | 35        |
| 015      | Wreath 33/33                  | 75        |
| 017      | Globe 1/8                     | 60        |
| 018      | Globe 1/16                    | 110       |
| 019      | Globe 2/6                     | 25        |
| 020      | Globe 3/4                     | 40        |
| 021      | Globe 6/4                     | 40        |
| 022      | Globe 6/8                     | 165       |
| 023      | Globe 6/10                    | 170       |
| 024      | Globe 3/33                    | 500       |
| 025      | Globe 8/25                    | 700       |
| 026      | Puzzle 8                      | 30        |
| 027      | Puzzle 15                     | 80        |
| 028      | Puzzle 24                     | 150       |
| 029      | Puzzle 35                     | 200       |
| 030      | Puzzle 48                     | 250       |
| 031      | Puzzle 63                     | 300       |
| 034      | LRX 10                        | 50        |
| 035      | LRX 15                        | 100       |
| 036      | LRX 20                        | 200       |
| 037      | LRX 25                        | 300       |
| 044      | Pancake 10                    | 15        |
| 045      | Pancake 15                    | 25        |
| 046      | Pancake 20                    | 30        |
| 047      | Pancake 25                    | 40        |
| 048      | Pancake 30                    | 45        |
| 049      | Pancake 35                    | 55        |
| 050      | Pancake 40                    | 60        |
| 051      | Pancake 45                    | 65        |
| 052      | Pancake 50                    | 70        |
| 053      | Pancake 55                    | 75        |
| 054      | Cube 3x3x3 (DeepCubeA metric) | 26        |

(For reproducing results from Table 4, refer to provided scripts `traintest-tab4-santa.sh` and `traintest-tab4-rnd.sh`.)


## Testing a Pre-trained Model

Run these commands to solve puzzles using preloaded model weights:


**Cube 3x3x3 (UQTM metric)**

```bash
python test.py --group_id 1 --target_id 0 --tests_num 3 --dataset santa --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```

**Cube 4x4x4 (UQTM metric)**

```bash
python test.py --group_id 2 --target_id 0 --tests_num 3 --dataset santa --num_steps 150 --verbose 1 --epoch 8192 --model_id 444 --B 262144 --device_id 0
```

**Cube 5x5x5 (UQTM metric)**

```bash
python test.py --group_id 3 --target_id 0 --tests_num 3 --dataset santa --num_steps 200 --verbose 1 --epoch 8192 --model_id 555 --B 524288 --device_id 0
```

**Cube 3x3x3 (QTM metric)**

```bash
python test.py --group_id 54 --target_id 0 --tests_num 3 --dataset deepcubea --num_steps 100 --verbose 1 --epoch 8192 --model_id 333 --B 262144 --device_id 0
```


### Notes

* `tests_num` — sets an upper limit on the number of scrambles to test from the beginning of the dataset.
* `dataset` — selects the dataset to use:
  * `santa` — official Kaggle Santa 2023 dataset.
  * `rnd` — randomly generated dataset (100 scrambles) using 10,000(+1) random steps from the solved state.
  * `deepcubea` (1000 DeepCubeA scrambles), `deepcubeadifficult` (69 DeepCubeA subset scrambles which was used in the third experiment), `deepcubeahard` (16 DeepCubeA subset scrambles that were not solved optimally with our approach) — available only for group `054` (3x3x3, QTM metric), useful for benchmarking against DeepCubeA and EfficientCube.
* `device_id` — specifies which GPU to use for testing (default is `0`). Useful when multiple GPUs are available.

Optimal solution lengths for the scrambles in the `deepcubeahard`: `[20, 20, 20, 21, 20, 20, 20, 20, 19, 20, 20, 19, 21, 20, 20, 20]`.


## Output

Test results saved in `logs/test_pXXX-tXXX-{dataset}_{model_id}_{epoch}_{B}.json`. Each file is a JSON array where `moves` contain generator indices used to reach the solved state for corresponding scramble. 


## Training Your Model

Run training using your puzzle configuration:

```bash
python train.py --group_id <id> --target_id 0 --epochs <epochs> --hd1 <N_1> --hd2 <N_2> --nrd <N_r> --batch_size 10000 --K_max <K_max> --device_id 0
```

Example for 24-puzzle:
```bash
python train.py --group_id 28 --target_id 0 --epochs 16 --hd1 1024 --hd2 512 --nrd 1 --batch_size 10000 --K_max 100 --device_id 0
```
After training, use the assigned model_id (generated as int(time.time()) during training) to run beam search evaluation to use trained model:
```bash
python test.py --group_id 28 --target_id 0 --tests_num 3 --dataset rnd --num_steps 300 --num_attempts 1 --verbose 1 --epoch 16 --model_id {MODEL_ID} --B 65536 --device_id 0
```
Replace `{MODEL_ID}` with the actual numeric identifier saved in the logs.


## Adding New Puzzle Groups

To add your puzzle:

1. Define your puzzle moves in `generators` folder as `pXXX.json`.
2. Place a torch tensor `.pt` (1D tensor) in `targets`.
3. Add scrambles dataset (2D tensor, each row as a scramble) to `datasets` folder.



## Multi-Agent Evaluation

For **Cube 3x3x3 (QTM)** on the **`deepcubea`** dataset, a multi-agent script `./traintest-multiagent.sh` automates:

* training multiple agents (`group_id=054`, `target_id=0`)
* testing each agent on the same scrambles
* aggregating results: per-agent and ensemble statistics

### What It Does

After training and testing `A` agents, the script calls `read-test-logs-multiagent.py` to compute:

* average solution length per agent
* ensemble stats (shortest solution per scramble)
* solved percentage
* move sequences from the best agent per scramble

### How to Run

Script `./run.sh A TESTS_NUM EPOCH B` runs `A` agents, tests on `EPOCH` train and test, with `B` beam width and `TESTS_NUM` as number of scrambles to test.
```bash
./run.sh  4 10 16 65536
```

### Output Example

```
=== per agent ===
          tests  solved_%  avg_len
123456789   1000     97.3     21.4
123456790   1000     98.2     20.8

=== ensemble (shortest per scramble) ===
solved %           : 99.1
avg solution length: 19.95

=== moves (winning agent) ===
 test_num  solution_length   model_id            moves
        0                20  123456790  [2, 0, 4, 5, 1, ...]
        ...
```

All logs are saved in `logs/`, with results printed at the end.
