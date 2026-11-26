"""Generate envs resembling the hidden ECML2026 test levels (more agents, longer lines)."""
import warnings, numpy as np
from pathlib import Path
from flatland.envs.persistence import RailEnvPersister
from reinforcement_learning.create_curriculum_env_fixed import create_env

warnings.filterwarnings("ignore")
OUT = Path("reinforcement_learning/testlike")
CONFIGS = [  # (scene, n_agents, line_length)
    ("scene_1", 36, 3), ("scene_2", 50, 3), ("scene_3", 62, 4), ("scene_4", 118, 6), ("scene_5", 210, 6),
    ("scene_1", 90, 3), ("scene_2", 125, 3), ("scene_3", 150, 4), ("scene_4", 300, 6), ("scene_5", 532, 6),
]
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (scene, n, ll) in enumerate(CONFIGS):
        np.random.seed(1000 + i)
        env = create_env(scene=scene, n_agents=n, line_length=ll, seed=1000 + i)
        p = OUT / f"t{i:02d}_{scene}_ll-{ll}_a-{n}.pkl"
        RailEnvPersister.save(env, str(p))
        print(p, env.get_num_agents(), env._max_episode_steps)
