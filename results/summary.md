Best cost found on chip-1000 after N placement evaluations (mean ± std over runs):

| Method | N = 64 | N = 1,024 | N = 4,096 | N = 12,800 |
|---|---:|---:|---:|---:|
| Random search | 1.099 | 1.055 | 0.930 | 0.930 |
| Simulated annealing (8 chains) | 1.148 ± 0.067 | 0.958 ± 0.032 | 0.881 ± 0.013 | 0.803 ± 0.007 |
| RL from scratch, no wire mask (3 seeds) | 1.138 ± 0.015 | 1.037 ± 0.015 | 0.931 ± 0.002 | 0.924 ± 0.011 |
| RL from scratch (3 seeds) | 1.138 ± 0.015 | 0.899 ± 0.010 | 0.817 ± 0.017 | 0.812 ± 0.019 |
| RL pre-trained + fine-tuned (3 seeds) | 0.890 ± 0.010 | 0.862 ± 0.003 | 0.847 ± 0.005 | 0.843 ± 0.004 |

Greedy heuristic (1 placement): 1.003.  RL zero-shot argmax (1 placement): 0.919.  Best single run: SA 0.795, RL scratch 0.796, RL fine-tuned 0.838.

Zero-shot on 20 held-out chips (mean cost):

| Method | Mean cost |
|---|---:|
| Random placement | 1.450 |
| Greedy heuristic | 0.976 |
| RL zero-shot, pre-trained on 4 chips | 1.014 |
| RL zero-shot, pre-trained on 16 chips | 0.983 |
| RL zero-shot, pre-trained on 64 chips | 0.981 |
| RL zero-shot, 64 chips, best of 64 samples | 0.902 |

Pre-trained on 64 chips, zero-shot beats the greedy heuristic on 8/20 chips with a single argmax pass and on 19/20 with the best of 64 samples.
Total wall-clock for all runs: 18.4 min.
