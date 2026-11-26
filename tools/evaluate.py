"""Local evaluation mirroring flatland PolicyRunner + ECML2026 normalization.

Usage: python -m tools.evaluate --policy submission.my_policy.MyPolicy \
          --obs submission.my_observation_builder.MyObservationBuilder [--envs GLOB] [--workers N]
"""
import argparse, glob, importlib, os, time, warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")


def _resolve(path):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def run_one(args):
    env_path, policy_path, obs_path = args
    import torch
    torch.set_num_threads(1)
    from flatland.envs.persistence import RailEnvPersister
    from flatland.envs.rewards import ECML2026Rewards
    from flatland.envs.step_utils.states import TrainState

    obs_builder = _resolve(obs_path)()
    policy = _resolve(policy_path)()
    env, _ = RailEnvPersister.load_new(env_path, obs_builder=obs_builder, rewards=ECML2026Rewards())
    obs = env._get_observations()
    n = env.get_num_agents()
    per_agent = [0.0] * n
    t_obs = t_act = 0.0
    t0 = time.time()
    for _ in range(env._max_episode_steps):
        ta = time.time()
        actions = policy.act_many(env.get_agent_handles(), observations=list(obs.values()))
        t_act += time.time() - ta
        tb = time.time()
        obs, rew, dones, _ = env.step(actions)
        t_obs += time.time() - tb
        for h in range(n):
            per_agent[h] += rew.get(h, 0.0)
        if dones["__all__"]:
            break
    T = env._max_episode_steps
    score = sum(max(r, -T) for r in per_agent) / (T * n) + 1
    arrived = sum(a.state == TrainState.DONE for a in env.agents) / n
    return dict(env=os.path.basename(env_path), n=n, T=T, score=score, arrived=arrived,
                steps=env._elapsed_steps, sec=time.time() - t0, sec_step_obs=t_obs, sec_policy=t_act,
                worst=min(per_agent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="submission.my_policy.MyPolicy")
    p.add_argument("--obs", default="submission.my_observation_builder.MyObservationBuilder")
    p.add_argument("--envs", default="reinforcement_learning/curriculum/curriculum/example_curriculum/*.pkl")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    files = sorted(glob.glob(a.envs))
    with Pool(a.workers) as pool:
        res = pool.map(run_one, [(f, a.policy, a.obs) for f in files], chunksize=1)
    tot = 0
    for r in res:
        tot += r["score"]
        print(f"{r['env']:28s} n={r['n']:3d} T={r['T']:5d} score={r['score']:.3f} arrived={r['arrived']:.2f} "
              f"worst={r['worst']:8.1f} time={r['sec']:6.1f}s (env+obs {r['sec_step_obs']:.1f}s, policy {r['sec_policy']:.1f}s)")
    print(f"TOTAL {tot:.3f} / {len(res)}  mean {tot/len(res):.3f}")


if __name__ == "__main__":
    main()
