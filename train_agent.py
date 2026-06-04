import gymnasium as gym
import numpy as np
import argparse
import os

PARAM_SIZE = 36  # 8*4 + 4

# ---------------- POLICY ----------------
def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    b = params[32:].reshape(4)
    logits = np.dot(observation, W) + b
    return int(np.argmax(logits))

# ---------------- EVALUATION ----------------
def evaluate_policy(params, episodes=10):
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_action(params, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += ep_reward
    return total_reward / episodes

# ---------------- GA PART ----------------
def uniform_crossover(p1, p2):
    mask = np.random.rand(len(p1)) < 0.5
    return np.where(mask, p1, p2)

def mutate(child, rate=0.08, scale=0.12):
    for i in range(len(child)):
        if np.random.rand() < rate:
            child[i] += np.random.randn() * scale
    return np.clip(child, -5, 5)

def run_ga(pop_size=120, generations=90, elite_frac=0.15, warm_start=None):

    if warm_start is not None:
        print("Warm starting GA from loaded warm start parameters")
        noise = np.random.randn(pop_size, PARAM_SIZE) * 0.15
        population = warm_start + noise
        population = np.clip(population, -5, 5)
        population[0] = warm_start.copy()  # keep original untouched
    else:
        print("No warm start found, initializing randomly")
        population = np.random.uniform(-1, 1, (pop_size, PARAM_SIZE))

    num_elites = int(pop_size * elite_frac)
    best_ever = -np.inf
    best_params=None

    for gen in range(generations):
        fitness = np.array([evaluate_policy(ind, 10) for ind in population])
        elite_idx = np.argsort(fitness)[::-1][:num_elites]
        elites = population[elite_idx]

        #if fitness[elite_idx[0]] > best_ever:
            #best_ever = fitness[elite_idx[0]]
        candidate = population[elite_idx[0]]
        candidate_score = fitness[elite_idx[0]]

        if candidate_score > best_ever:
           stable_score = evaluate_policy(candidate, 20)
           if stable_score > best_ever:
                best_ever = stable_score
                best_params=candidate.copy()

        print(f"[GA] Gen {gen+1:>3} | Gen Best: {fitness[elite_idx[0]]:.2f} | Overall Best: {best_ever:.2f}")

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1, p2 = elites[np.random.choice(num_elites, 2, replace=False)]
            child = uniform_crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = np.array(new_pop)

    return population, elites,best_params

# ---------------- CEM PART ----------------
def run_cem(initial_elites,best_from_ga, generations=25, pop_size=60):
    mean = best_from_ga.copy() if best_from_ga is not None else np.mean(initial_elites, axis=0)
    std  = np.std(initial_elites, axis=0)
    std  = np.clip(std, 0.05, 0.6)

    best_params = None
    best_score  = -np.inf

    for gen in range(generations):
        population = np.random.randn(pop_size, PARAM_SIZE) * std + mean
        population = np.clip(population, -5, 5)

        # always include current best mean as a candidate
        population[0] = mean.copy()

        fitness    = np.array([evaluate_policy(ind, 10) for ind in population])
        elite_idx  = np.argsort(fitness)[-int(pop_size * 0.2):]
        elites     = population[elite_idx]

        new_mean = np.mean(elites, axis=0)
        new_std  = np.std(elites, axis=0) + 1e-2

        # smoothed update
        mean = 0.75 * mean + 0.25 * new_mean
        std  = 0.8 * std  + 0.1 * new_std
        std  = np.clip(std, 0.03, 0.5)
        
        #gen_best_score = np.max(fitness)
        #if gen_best_score > best_score:
            #best_score  = gen_best_score
            #best_params = population[np.argmax(fitness)].copy() 
        best_idx = np.argmax(fitness)
        gen_best_score = fitness[best_idx]
        candidate = population[best_idx]

        if gen_best_score > best_score:
            stable_score = evaluate_policy(candidate, 20)
            if stable_score > best_score:
               best_score = stable_score
               best_params = candidate.copy()
        print(f"[CEM] Gen {gen+1:>3} | Gen Best: {gen_best_score:.2f} | Overall Best: {best_score:.2f}")

    return best_params

# ---------------- MAIN ----------------
def train_and_save(filename, seed=None, warm_start_file="best_policy.npy"):
    if seed is not None:
        np.random.seed(seed)

    # load warm start
    warm_start = None
    if os.path.exists(warm_start_file):
        warm_start = np.load(warm_start_file)
        print(f"Loaded warm start from {warm_start_file}")
    else:
        print(f"{warm_start_file} not found — starting from scratch")

    print("\n========== PHASE 1: GA ==========")
    population, elites ,best_from_ga = run_ga(warm_start=warm_start)

    print("\n========== PHASE 2: CEM ==========")
    best_params = run_cem(elites,best_from_ga)

    np.save(filename, best_params)
    print(f"\nDone. Best policy saved to {filename}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--filename",       type=str, default="best_policy.npy")
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--warm_start",     type=str, default="best_policy.npy")
    args = parser.parse_args()

    if args.train:
        train_and_save(args.filename, seed=args.seed, warm_start_file=args.warm_start)
    else:
        print("Use --train to start training.")