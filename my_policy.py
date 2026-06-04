import numpy as np

INPUT_SIZE = 8
OUTPUT_SIZE = 4


def policy_action(params, observation):
    """
    Linear policy for LunarLander-v3

    params:
        flat numpy array of length 36
        first 32 values -> W of shape (8, 4)
        last 4 values  -> b of shape (4,)

    observation:
        numpy array of length 8

    returns:
        integer action in {0, 1, 2, 3}
    """
    W = params[:INPUT_SIZE * OUTPUT_SIZE].reshape(INPUT_SIZE, OUTPUT_SIZE)
    b = params[INPUT_SIZE * OUTPUT_SIZE:].reshape(OUTPUT_SIZE)

    logits = np.dot(observation, W) + b
    return int(np.argmax(logits))