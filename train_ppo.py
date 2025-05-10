#train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from uav_env import UAV3DEnvironment

def make_env():
    return UAV3DEnvironment(num_obstacles=4, num_lidar_rays=12)

env = make_env()
check_env(env)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./ppo_uav_checkpoints/",
    name_prefix="ppo_uav_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# Optimized training parameters
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=0.00025,
n_steps=2048,
batch_size=512,
n_epochs=10,
gamma=0.99,
gae_lambda=0.95,
clip_range=0.2,
    ent_coef=0.1,
   policy_kwargs={
    "net_arch": [dict(pi=[512, 256], vf=[512, 256])],  # Deeper network
},
    tensorboard_log="./ppo_uav_tensorboard/"
)

model.learn(
    total_timesteps=500000,
    callback=checkpoint_callback,
    progress_bar=True
)

model.save("ppo_uav_continuous_model")
env.close()