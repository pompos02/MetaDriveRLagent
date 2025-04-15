
from metadrive.envs import MetaDriveEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif
from IPython.display import Image
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod


map_config = {BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 30,  # Generate maps with 30 blocks
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 4,
            "exit_length": 50,
            "start_position": [0, 0],}

def create_env(need_monitor=False):
    env = MetaDriveEnv(dict(map="SSySOSCSSXCSSTSCSSCSS",
                      # This policy setting simplifies the task                      
                      discrete_action=True,
                      discrete_throttle_dim=3,
                      discrete_steering_dim=3,
                      horizon=500,
                      random_spawn_lane_index=False,
                      start_seed=0,
                      traffic_density=0,
                      accident_prob=0,
                      log_level=50,
                      map_config=map_config),
                      )
    if need_monitor:
        env = Monitor(env)
    return env


env=create_env()
env.reset(seed=0)
frame_1 = env.render(mode="topdown", window=False,
                     screen_size=(800, 800), scaling=1)
env.close()

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.imshow(frame_1)
plt.axis('off')
plt.title("Seed: 0, Normal")
plt.show()
