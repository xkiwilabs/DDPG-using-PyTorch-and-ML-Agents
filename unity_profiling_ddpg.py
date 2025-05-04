
#!/usr/bin/env python
"""
Profiling framework (old / new / mix policies) for custom Unity-DDPG agent,
without any WandB integration. Results and final weights are saved locally.
"""

import argparse, os, numpy as np, torch, copy, csv
from collections import defaultdict
from unityagents import UnityEnvironment

from ddpg_agent import Agent, device    # uses your existing Agent class

# ──────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent: Agent, env: UnityEnvironment, brain_name: str,
                    n_episodes: int = 5) -> float:
    """Run deterministic episodes (no noise) and return mean reward."""
    scores = []
    for _ in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        states   = env_info.vector_observations
        agent.reset()
        ep_r     = np.zeros(agent.num_agents)
        while True:
            actions    = agent.act(states, add_noise=False)
            env_info   = env.step(actions)[brain_name]
            states     = env_info.vector_observations
            ep_r      += env_info.rewards
            if np.any(env_info.local_done):
                break
        scores.append(np.mean(ep_r))
    return float(np.mean(scores))

def clone_params(agent: Agent):
    """Deep-copy actor & critic state_dicts."""
    return (copy.deepcopy(agent.actor_local.state_dict()),
            copy.deepcopy(agent.critic_local.state_dict()))

def load_params(agent: Agent, actor_sd, critic_sd):
    agent.actor_local.load_state_dict(actor_sd)
    agent.critic_local.load_state_dict(critic_sd)

def mix_params(actor_old, actor_new, critic_old, critic_new, lam):
    actor_mix  = {k: lam*actor_new[k]  + (1-lam)*actor_old[k]  for k in actor_old}
    critic_mix = {k: lam*critic_new[k] + (1-lam)*critic_old[k] for k in critic_old}
    return actor_mix, critic_mix

# ──────────────────────────────────────────────────────────────────────────────
def profiling_run(env_path: str,
                  episodes_total: int,
                  episodes_budget: int,
                  eval_episodes: int,
                  variant: str,
                  seed: int,
                  lambda_alpha: float,
                  lambda_beta: float,
                  threshold_range,
                  log_dir: str):

    # prepare output folder
    os.makedirs(log_dir, exist_ok=True)

    # Unity env
    env = UnityEnvironment(file_name=env_path, seed=seed, no_graphics=True)
    brain_name = env.brain_names[0]
    brain      = env.brains[brain_name]

    # sizes
    action_size = brain.vector_action_space_size
    state_size  = brain.vector_observation_space_size
    num_agents  = len(env.reset(train_mode=True)[brain_name].agents)

    # Agent
    agent = Agent(state_size, action_size, num_agents, random_seed=seed)

    # results store
    results = defaultdict(list)
    episodes_per_round = episodes_total // episodes_budget

    # profiling rounds
    for rnd in range(1, episodes_budget+1):
        print(f"\nRound {rnd}/{episodes_budget}")

        # snapshot old
        actor_old, critic_old = clone_params(agent)
        r_old = evaluate_policy(agent, env, brain_name, eval_episodes)
        print(f"  old reward = {r_old:.2f}")

        # train for N episodes
        for ep in range(episodes_per_round):
            env_info = env.reset(train_mode=True)[brain_name]
            states   = env_info.vector_observations
            agent.reset()
            while True:
                acts = agent.act(states)
                env_info = env.step(acts)[brain_name]
                next_states = env_info.vector_observations
                rewards      = env_info.rewards
                dones        = env_info.local_done
                agent.step(states, acts, rewards, next_states, dones)
                states = next_states
                if np.any(dones):
                    break

        # snapshot new
        actor_new, critic_new = clone_params(agent)
        r_new = evaluate_policy(agent, env, brain_name, eval_episodes)
        print(f"  new reward = {r_new:.2f}")

        # mix
        lam = min(max(np.random.beta(lambda_alpha, lambda_beta), 0.0),
                  np.random.uniform(*threshold_range))
        actor_mix, critic_mix = mix_params(actor_old, actor_new,
                                           critic_old, critic_new, lam)
        load_params(agent, actor_mix, critic_mix)
        r_mix = evaluate_policy(agent, env, brain_name, eval_episodes)
        print(f"  mix λ={lam:.3f}  reward = {r_mix:.2f}")

        # select
        sel_r, sel_p = r_new, "new"
        if variant == "lookback":
            sel_r, sel_p = (r_old, "old") if r_old >= r_new else (r_new, "new")
            sd_actor, sd_critic = (actor_old, critic_old) if sel_p=="old" else (actor_new, critic_new)
            load_params(agent, sd_actor, sd_critic)
        elif variant == "mixup":
            sel_r, sel_p = (r_old, "old") if r_old >= r_mix else (r_mix, "mix")
            sd_actor, sd_critic = (actor_old, critic_old) if sel_p=="old" else (actor_mix, critic_mix)
            load_params(agent, sd_actor, sd_critic)
        elif variant == "three-points":
            vals = [r_old, r_new, r_mix]
            idx  = int(np.argmax(vals))
            sel_r, sel_p = vals[idx], ["old","new","mix"][idx]
            sd_actor = [actor_old, actor_new, actor_mix][idx]
            sd_critic= [critic_old,critic_new,critic_mix][idx]
            load_params(agent, sd_actor, sd_critic)

        print(f"  → selected {sel_p}  (reward={sel_r:.2f})")

        # record
        results["round"].append(rnd)
        results["reward_old"].append(r_old)
        results["reward_new"].append(r_new)
        results["reward_mix"].append(r_mix)
        results["reward_selected"].append(sel_r)
        results["lambda"].append(lam)

    # save results CSV
    csv_path = os.path.join(log_dir, "profiling_results.csv")
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(results.keys())
        for i in range(len(results["round"])):
            writer.writerow([results[k][i] for k in results.keys()])
    print(f"\nSaved profiling results → {csv_path}")

    # save final weights
    torch.save(agent.actor_local.state_dict(),
               os.path.join(log_dir, "ddpg_actor_final.pth"))
    torch.save(agent.critic_local.state_dict(),
               os.path.join(log_dir, "ddpg_critic_final.pth"))
    print("Saved final actor/critic weights.")

    env.close()
    return results

# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-path",      required=True,
                   help="Unity executable path (e.g. Reacher.x86_64)")
    p.add_argument("--episodes-total",type=int, default=500,
                   help="Total training episodes budget")
    p.add_argument("--episodes-budget",type=int, default=10,
                   help="Profiling rounds (>=1)")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="Deterministic eval episodes per round")
    p.add_argument("--variant",       choices=["vanilla","lookback","mixup","three-points"],
                   default="vanilla")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--log-dir",       default="u_logs",
                   help="Folder to save CSV & weights")
    p.add_argument("--lambda-alpha",  type=float, default=2.0)
    p.add_argument("--lambda-beta",   type=float, default=2.0)
    p.add_argument("--threshold-min", type=float, default=0.75)
    p.add_argument("--threshold-max", type=float, default=0.9)
    args = p.parse_args()

    profiling_run(env_path=args.env_path,
                  episodes_total=args.episodes_total,
                  episodes_budget=args.episodes_budget,
                  eval_episodes=args.eval_episodes,
                  variant=args.variant,
                  seed=args.seed,
                  lambda_alpha=args.lambda_alpha,
                  lambda_beta=args.lambda_beta,
                  threshold_range=(args.threshold_min,args.threshold_max),
                  log_dir=args.log_dir)

if __name__ == "__main__":
    main()
