#test_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from uav_env import UAV3DEnvironment
import time
import os

max_episodes = 10
render_delay = 0.02

env = UAV3DEnvironment(num_obstacles=5, num_lidar_rays=12, render_mode="human")

stats_path = "./ppo_uav_checkpoints/vecnormalize.pkl"
if os.path.exists(stats_path):
    env = DummyVecEnv([lambda: UAV3DEnvironment(
    num_obstacles=5, 
    num_lidar_rays=12, 
    render_mode="human"
)])

if os.path.exists(stats_path):
    env = VecNormalize.load(stats_path, env)
else:
    env = VecNormalize(env, training=False, norm_reward=False)

env.training = False
env.norm_reward = False

try:
    model = PPO.load("ppo_uav_continuous_model", env=env)
    
    current_episode = 0
    obs = env.reset()
    
    while current_episode < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        time.sleep(render_delay)
        
        if dones[0]:
            current_episode += 1
            print(f"Episode {current_episode} completed")
            print(f"Final Distance: {infos[0]['distance_to_target']:.2f}")
            print(f"Avg Speed: {infos[0]['velocity']:.2f} m/s")
            obs = env.reset()

except KeyboardInterrupt:
    print("Test interrupted by user")
finally:
    env.close()