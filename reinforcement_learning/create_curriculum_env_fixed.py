"""Seeded env factory built on the starterkit sampling generator (fixed agent count)."""
import os
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards
from reinforcement_learning.sampling.sampling_env_generator import sampling_env_generator

SCENARIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling", "level_0_scenario_1.pkl")


def create_env(scene="scene_5", n_agents=10, line_length=3, seed=None, obs_builder=None):
    env = RailEnvPersister.load_new(SCENARIO, rewards=ECML2026Rewards(), obs_builder=obs_builder)[0]
    env = sampling_env_generator(env, line_length=line_length, scene=scene)
    env.number_of_agents = n_agents
    env.reset(random_seed=seed)
    return env
