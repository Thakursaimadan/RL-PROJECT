# uav_env.py
import numpy as np
import pybullet as p
import pybullet_data
import time
from gymnasium import Env
from gymnasium.spaces import Box
import math

class UAV3DEnvironment(Env):
    def __init__(self, num_obstacles=3, render_mode=None, num_lidar_rays=12):
        super(UAV3DEnvironment, self).__init__()
        # Environment parameters
        self.x_range = (-2, 2)
        self.y_range = (-2, 2)
        self.z_range = (0.5, 2.5)
        self.num_obstacles = num_obstacles
        self.num_lidar_rays = num_lidar_rays
        self.max_lidar_distance = 3.0
        self.collision_radius = 0.25
        self.max_speed = 0.15

        # Action and observation spaces
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0,
                                     shape=(9 + self.num_lidar_rays,), dtype=np.float32)

        # PyBullet setup
        self.render_mode = render_mode
        self.lidar_debug_lines = []
        self.obstacles = []
        self._init_physics_engine()

    def _init_physics_engine(self):
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=45,
                                         cameraPitch=-30, cameraTargetPosition=[0,0,1])
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self._setup_world()
        self.uav_velocity = np.zeros(3)
        return self._get_observation(), {}

    def _setup_world(self):
        for line_id in self.lidar_debug_lines:
            p.removeUserDebugItem(line_id)
        self.lidar_debug_lines.clear()

        self.plane_id = p.loadURDF("plane.urdf")
        self._create_boundary_markers()

        self.uav_start = [0, 0, 1.0]
        self.uav_id = self._create_uav()

        self.target_pos, self.target_id = self._create_target()
        self.obstacles = self._create_obstacles()

        self.uav_velocity = np.zeros(3)
        self.prev_dist = self._calculate_target_distance()
        for _ in range(50): p.stepSimulation()

    def _create_uav(self):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1,0,0,1])
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.collision_radius)
        uid = p.createMultiBody(1.0, col, vis, self.uav_start)
        p.changeDynamics(uid, -1, linearDamping=0.85, angularDamping=0.85)
        return uid

    def _create_target(self):
        valid = False
        while not valid:
            pos = [np.random.uniform(self.x_range[0]+0.5, self.x_range[1]-0.5),
                   np.random.uniform(self.y_range[0]+0.5, self.y_range[1]-0.5),
                   np.random.uniform(self.z_range[0], self.z_range[1]-0.5)]
            if np.linalg.norm(np.array(pos)-np.array(self.uav_start))>1.5: valid=True
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0,1,0,1])
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        return pos, p.createMultiBody(0, col, vis, pos)

    def _create_obstacles(self):
        obs = []
        for _ in range(self.num_obstacles):
            pos,size = self._generate_valid_obstacle()
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size]*3,
                                      rgbaColor=[0.7,0.2,1,0.8])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3)
            oid = p.createMultiBody(0, col, vis, pos)
            obs.append({'id':oid,'pos':pos,'size':size})
        return obs

    def _generate_valid_obstacle(self, max_attempts=50):
        for _ in range(max_attempts):
            pos = [np.random.uniform(self.x_range[0]+0.5, self.x_range[1]-0.5),
                   np.random.uniform(self.y_range[0]+0.5, self.y_range[1]-0.5),
                   np.random.uniform(self.z_range[0]+0.3, self.z_range[1]-0.3)]
            size = np.random.uniform(0.2,0.4)
            if self._is_valid_position(pos,size): return pos,size
        return [0,0,-10],0.1

    def _is_valid_position(self,pos,size):
        if np.linalg.norm(np.array(pos)-np.array(self.uav_start))<(1.0+size): return False
        if np.linalg.norm(np.array(pos)-np.array(self.target_pos))<(0.8+size): return False
        for o in getattr(self,'obstacles',[]):
            if np.linalg.norm(np.array(pos)-np.array(o['pos']))<(o['size']+size+0.5): return False
        return True

    def _get_observation(self):
        pos,_ = p.getBasePositionAndOrientation(self.uav_id)
        obs = np.zeros(9+self.num_lidar_rays,dtype=np.float32)
        obs[0] = (pos[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        obs[1] = (pos[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        obs[2] = (pos[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 2 - 1
        obs[3:6] = np.clip(self.uav_velocity/self.max_speed,-1,1)
        delta = np.array(self.target_pos)-np.array(pos)
        maxd = np.array([self.x_range[1]-self.x_range[0],
                         self.y_range[1]-self.y_range[0],
                         self.z_range[1]-self.z_range[0]])
        obs[6:9] = np.clip(delta/maxd, -1,1)
        obs[9:] = self._get_lidar_readings()
        return obs

    def _get_lidar_readings(self):
        uav_pos, _ = p.getBasePositionAndOrientation(self.uav_id)
        readings = []

        for i in range(self.num_lidar_rays):
            theta = 2 * math.pi * i / self.num_lidar_rays
            phi   = math.pi / 2  # horizontal plane

            # direction vector
            ray_dir = [
                math.cos(theta) * math.sin(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(phi)
            ]
            ray_end = [uav_pos[j] + ray_dir[j] * self.max_lidar_distance for j in range(3)]

            # Perform raycast
            hit_id, link_idx, hit_fraction, hit_pos, hit_norm = p.rayTest(uav_pos, ray_end)[0]

            # If we hit the target, pretend we hit nothing
            if hit_id == self.target_id:
                hit_fraction = 1.0

            # Convert to distance
            if hit_fraction < 1.0:
                distance = hit_fraction * self.max_lidar_distance
                # Only draw debug line for real obstacles
                if self.render_mode == "human" and hit_id != self.target_id:
                    p.addUserDebugLine(uav_pos, hit_pos, [1, 0, 0], lifeTime=0.1)
            else:
                distance = self.max_lidar_distance

            # Normalize [0, 1]
            readings.append(distance / self.max_lidar_distance)

        return np.array(readings, dtype=np.float32)


    def step(self, action):
        # Apply action and update position
        mv = np.clip(action, -1, 1) * self.max_speed
        self.uav_velocity = 0.8 * self.uav_velocity + 0.2 * mv
        cp = np.array(p.getBasePositionAndOrientation(self.uav_id)[0])
        npv = cp + self.uav_velocity
        npv[:2] = np.clip(npv[:2], [self.x_range[0]+0.1, self.y_range[0]+0.1],
                        [self.x_range[1]-0.1, self.y_range[1]-0.1])
        npv[2] = np.clip(npv[2], self.z_range[0]+0.1, self.z_range[1]-0.1)
        p.resetBasePositionAndOrientation(self.uav_id, npv.tolist(), [0, 0, 0, 1])
        p.stepSimulation()
        
        rew = 0
        done = False
        info = {}
        
        # Check for collisions with obstacles
        if any(p.getContactPoints(self.uav_id, o['id']) for o in self.obstacles):
            rew -= 20
            done = True
            info['collision'] = True
            return self._get_observation(), rew, done, False, info
        
        dist = self._calculate_target_distance()
        if dist < 0.5:
            # Success reward for proximity
            done = True
            info['success'] = True
            return self._get_observation(), rew, done, False, info
        # Check for collision with target - MUCH HIGHER REWARD
        if any(p.getContactPoints(self.uav_id, self.target_id)):
            rew += 200  # Extremely large reward for actual collision
            done = True
            info['success'] = True
            return self._get_observation(), rew, done, False, info
        
        # Only calculate distance-based rewards if no collision detected
        dist = self._calculate_target_distance()
        
        # Smaller approach reward - only a guidance signal
        approach_reward = (self.prev_dist - dist) * 5
        rew += approach_reward
        self.prev_dist = dist
        
        # Add penalty for orbiting behavior - detect if distance isn't changing much
        if abs(approach_reward) < 0.02 and dist < 0.8:
            # If close to target but not making progress, penalize
            rew -= 0.5  # Penalty for orbiting
        
        # Obstacle avoidance using lidar
        rew -= np.sum(np.clip(1-self._get_lidar_readings(), 0, 1)) * 0.3
        
        # Small time penalty
        rew -= 0.1
        
        return self._get_observation(), rew, done, False, info

    def _calculate_target_distance(self):
        return np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.uav_id)[0])-
                              np.array(self.target_pos))

    def _create_boundary_markers(self):
        lc=[0.5,0.5,0.5]
        corners=[[self.x_range[0],self.y_range[0],0],[self.x_range[1],self.y_range[0],0],
                 [self.x_range[1],self.y_range[1],0],[self.x_range[0],self.y_range[1],0]]
        for i in range(4):p.addUserDebugLine(corners[i],corners[(i+1)%4],lc,lineWidth=2)
        ceil=self.z_range[1]
        c2=[c.copy() for c in corners]
        for c in c2:c[2]=ceil
        for i in range(4):p.addUserDebugLine(c2[i],c2[(i+1)%4],lc,lineWidth=2);
        for i in range(4):p.addUserDebugLine(corners[i],c2[i],lc,lineWidth=2)

# train_a2c.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import types
import pybullet as p  # Import pybullet
from uav_env import UAV3DEnvironment
from models import Actor, Critic
from collections import deque

class A2CAgent:
    def __init__(self, env):
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
        self.critic = Critic(env.observation_space.shape[0])
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-5)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.entropy_coef = 0.01

    def update(self, states, actions, rewards, next_states, dones):
      
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        values = self.critic(states)
        next_values = self.critic(next_states)

        td_targets = rewards + self.gamma * next_values * (1 - dones)
        advantages = (td_targets.detach() - values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist = self.actor(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy
        critic_loss = nn.MSELoss()(values, td_targets.detach())

        # Actor update (retain graph for critic)
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        # Critic update
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

def modified_create_target(self, target_radius):
    """Modified target creation function with specified radius"""
    valid = False
    while not valid:
        pos = [np.random.uniform(self.x_range[0]+0.5, self.x_range[1]-0.5),
               np.random.uniform(self.y_range[0]+0.5, self.y_range[1]-0.5),
               np.random.uniform(self.z_range[0], self.z_range[1]-0.5)]
        if np.linalg.norm(np.array(pos)-np.array(self.uav_start)) > 1.5:
            valid = True
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=target_radius, rgbaColor=[0,1,0,1])
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=target_radius)
    return pos, p.createMultiBody(0, col, vis, pos)

def train_a2c():
    # Initialize environment
    env = UAV3DEnvironment(num_obstacles=0, render_mode=None)
    agent = A2CAgent(env)
    max_episodes = 3000
    max_steps = 500
    reward_history = deque(maxlen=100)
    
    # Original target radius
    original_target_radius = 0.3
    
    # Keep original method
    original_create_target = env._create_target
    
    # Training phases
    for phase in range(3):
        # Gradually decrease target radius
        if phase == 0:
            target_radius = 0.6  # Start with larger target
        elif phase == 1:
            target_radius = 0.45  # Medium target
        else:
            target_radius = original_target_radius  # Normal target
            
        print(f"=== Training Phase {phase+1}: Target radius = {target_radius} ===")
        
        # Create a closure with the current target radius
        def create_target_with_radius(self):
            return modified_create_target(self, target_radius)
        
        # Apply the method to the environment
        env._create_target = types.MethodType(create_target_with_radius, env)
        
        # Train for this phase
        phase_episodes = 1000
        for episode in range(phase_episodes):
            state, _ = env.reset()
            episode_reward = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            for step in range(max_steps):
                s_t = torch.FloatTensor(state).unsqueeze(0)
                action_dist = agent.actor(s_t)
                
                # Add more exploration noise during early training
                noise_scale = max(0.1, 0.5 - 0.4 * (episode / phase_episodes))
                action = action_dist.sample().squeeze(0).numpy()
                
                # Add exploration noise
                if phase < 2:  # Only in earlier phases
                    action += np.random.normal(0, noise_scale, size=action.shape)
                    action = np.clip(action, -1, 1)
                    
                next_state, reward, done, _, info = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                state = next_state
                episode_reward += reward
                if done:
                    break
                    
            agent.update(states, actions, rewards, next_states, dones)
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history) if len(reward_history) > 0 else 0
            
            if (episode + 1) % 20 == 0:
                print(f"Phase {phase+1}, Episode {episode+1}, Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}")
                
            # Early stopping condition if performance is good
            if len(reward_history) >= 100 and avg_reward >= 150:
                print(f"Phase {phase+1} solved with average reward {avg_reward:.2f}")
                break
                
        # Save checkpoint for this phase
        torch.save(agent.actor.state_dict(), f"uav_actor_phase{phase+1}.pth")
    
    # Restore original method
    env._create_target = original_create_target
    
    # Final model save
    torch.save(agent.actor.state_dict(), "uav_actor_a2c.pth")
    torch.save(agent.critic.state_dict(), "uav_critic_a2c.pth")

if __name__ == "__main__":
    train_a2c()