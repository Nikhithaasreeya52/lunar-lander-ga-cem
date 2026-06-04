# Lunar Lander using Genetic Algorithm + Cross-Entropy Method

## Overview

This project trains an autonomous LunarLander-v3 agent using evolutionary optimization techniques instead of gradient-based reinforcement learning.

The agent uses a hybrid optimization pipeline:

- Genetic Algorithm (GA)
  - Elitism
  - Uniform Crossover
  - Gaussian Mutation

- Cross-Entropy Method (CEM)
  - Elite sampling
  - Distribution refinement
  - Smoothed parameter updates

## Policy

The policy is a linear controller with:

- 8 state inputs
- 4 possible actions
- 36 trainable parameters

Actions are selected using:

```python
action = argmax(observation @ W + b)
```

## Results

- Average Reward: **275.24**
- Evaluation Episodes: **100**
- Environment: **LunarLander-v3**

The achieved reward exceeds the commonly accepted solved threshold of 200.

## Technologies

- Python
- NumPy
- Gymnasium
- Box2D

## Training

```bash
python3 train_agent.py --train --filename less_luck.npy --warm_start best_trainone.npy
```

## Evaluation

```bash
python3 evaluate_agent.py --filename less_luck.npy --policy_module my_policy
```

## Future Improvements

- Replace linear policy with a neural network policy
- Compare performance against DQN and PPO
- Track training metrics and reward curves
- Add visualization and experiment logging