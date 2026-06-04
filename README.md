# Lunar Lander using Genetic Algorithm + Cross-Entropy Method

This project trains an autonomous LunarLander-v3 agent using evolutionary optimization techniques instead of gradient-based reinforcement learning.

## Features

- Genetic Algorithm (GA)
  - Elitism
  - Uniform Crossover
  - Gaussian Mutation

- Cross Entropy Method (CEM)
  - Elite sampling
  - Distribution refinement
  - Smoothed parameter updates

## Results

- Average Reward: 276.79
- Evaluation Episodes: 100
- Environment: LunarLander-v3

## Technologies

- Python
- NumPy
- Gymnasium

## Run Evaluation

```bash
python evaluate_agent.py --filename less_luck.npy --policy_module my_policy
```
