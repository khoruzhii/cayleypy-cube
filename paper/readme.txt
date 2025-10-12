This directory contains paper evaluation data.

solver-scrambles/
  Each subdirectory corresponds to a solver model from Table 1 of the paper.

figure-scrambles/
  Each subdirectory (fig-*) corresponds to an experiment from the paper.  

Each figure and solver folder contains:
- generators.json – Defines permutation actions and symbolic names used in searcher.py.  
- data.pt – A 2D Torch tensor (Int8) where each row represents a scrambled Rubik’s Cube state. Values in [0, 6) correspond to face colors.

