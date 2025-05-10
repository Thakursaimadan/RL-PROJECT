# test_a2c.py
import torch
from uav_env import UAV3DEnvironment
from models import Actor
import time

def test(model_path="uav_actor_a2c.pth"):
    env = UAV3DEnvironment(render_mode="human", num_obstacles=5)
    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
    actor.load_state_dict(torch.load(model_path))
    actor.eval()
    
    state, _ = env.reset()
    total_reward = 0
    
    for _ in range(2000):  # Max steps
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = actor(state_tensor).mean.numpy()
            
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        env.render()
        time.sleep(0.02)
        
        if done:
            break
    
    print(f"Total test reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    test("uav_actor_a2c.pth") 