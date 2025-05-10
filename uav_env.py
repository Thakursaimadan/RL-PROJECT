# uav_env.py - COMPLETE VERSION
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
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = Box(
    low=np.array([-1.0]*9 + [0.0]*self.num_lidar_rays, dtype=np.float32),
    high=np.array([1.0]*(9 + self.num_lidar_rays), dtype=np.float32),
    dtype=np.float32
)

        # PyBullet setup
        self.render_mode = render_mode
        self.obstacle_ids = []
        self.lidar_debug_lines = []
        self.obstacles = []
        self._init_physics_engine()

    def _init_physics_engine(self):
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 1]
            )
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self._setup_world()
        self.uav_velocity = np.array([0.0, 0.0, 0.0])
        return self._get_observation(), {}

    def _setup_world(self):
        # Clear previous debug lines
        for line_id in self.lidar_debug_lines:
            p.removeUserDebugItem(line_id)
        self.lidar_debug_lines = []

        # Create floor and boundaries
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_boundary_markers()

        # Spawn UAV
        self.uav_start = [0, 0, 1.0]
        self.uav_id = self._create_uav()

        # Generate target and obstacles
        self.target_pos, self.target_id = self._create_target()
        self.obstacles = self._create_obstacles()

        # Initialize UAV state
        self.uav_velocity = np.array([0.0, 0.0, 0.0])
        self.prev_dist = self._calculate_target_distance()

        # Physics stabilization
        for _ in range(50):
            p.stepSimulation()

    def _create_uav(self):
        uav_radius = 0.2
        collision_radius = 0.25
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=uav_radius, rgbaColor=[1, 0, 0, 1])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=collision_radius)
        uav_id = p.createMultiBody(1.0, collision, visual, self.uav_start)
        p.changeDynamics(uav_id, -1, linearDamping=0.85, angularDamping=0.85)
        return uav_id

    def _create_target(self):
        valid = False
        while not valid:
            target_pos = [
                np.random.uniform(self.x_range[0]+0.5, self.x_range[1]-0.5),
                np.random.uniform(self.y_range[0]+0.5, self.y_range[1]-0.5),
                np.random.uniform(self.z_range[0], self.z_range[1]-0.5)
            ]
            if np.linalg.norm(np.array(target_pos)-np.array(self.uav_start)) > 1.5:
                valid = True
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0, 1, 0, 1])
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        return target_pos, p.createMultiBody(0, collision, visual, target_pos)

    def _create_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            pos, size = self._generate_valid_obstacle()
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[size]*3, rgbaColor=[0.7, 0.2, 1.0, 0.8])
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size]*3)
            obstacles.append({
                'id': p.createMultiBody(0, collision, visual, pos),
                'pos': pos,
                'size': size
            })
        return obstacles

    def _generate_valid_obstacle(self, max_attempts=50):
        for _ in range(max_attempts):
            pos = [
                np.random.uniform(self.x_range[0]+0.5, self.x_range[1]-0.5),
                np.random.uniform(self.y_range[0]+0.5, self.y_range[1]-0.5),
                np.random.uniform(self.z_range[0]+0.3, self.z_range[1]-0.3)
            ]
            size = np.random.uniform(0.2, 0.4)
            if self._is_valid_position(pos, size):
                return pos, size
        return [0, 0, -10], 0.1

    def _is_valid_position(self, pos, size):
        # Check distance from UAV start
        if np.linalg.norm(np.array(pos)-np.array(self.uav_start)) < (1.0 + size):
            return False
        
        # Check distance from target
        if np.linalg.norm(np.array(pos)-np.array(self.target_pos)) < (0.8 + size):
            return False
        
        # Only check against existing obstacles if we have some
        if hasattr(self, 'obstacles'):
            for obstacle in self.obstacles:
                if np.linalg.norm(np.array(pos)-np.array(obstacle['pos'])) < (obstacle['size'] + size + 0.5):
                    return False
        return True

    def _get_observation(self):
        uav_pos, _ = p.getBasePositionAndOrientation(self.uav_id)
        obs = np.zeros(9 + self.num_lidar_rays, dtype=np.float32)
        
        # Normalized position [-1, 1]
        obs[0] = (uav_pos[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        obs[1] = (uav_pos[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        obs[2] = (uav_pos[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 2 - 1
        
        # Normalized velocity [-1, 1]
        obs[3:6] = np.clip(self.uav_velocity / self.max_speed, -1, 1)
        
        # Normalized target position [-1, 1]
        obs[6] = (self.target_pos[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 2 - 1
        obs[7] = (self.target_pos[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 2 - 1
        obs[8] = (self.target_pos[2] - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 2 - 1
        
        # LiDAR readings [0, 1]
        obs[9:] = self._get_lidar_readings()
        
        return obs

    def _get_lidar_readings(self):
        uav_pos, _ = p.getBasePositionAndOrientation(self.uav_id)
        readings = []
        
        for i in range(self.num_lidar_rays):
            theta = 2 * math.pi * i / self.num_lidar_rays
            phi = math.pi * (i % (self.num_lidar_rays//2)) / (self.num_lidar_rays//2)
            
            ray_dir = [
                math.cos(theta) * math.sin(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(phi)
            ]
            
            ray_end = [
                uav_pos[0] + ray_dir[0] * self.max_lidar_distance,
                uav_pos[1] + ray_dir[1] * self.max_lidar_distance,
                uav_pos[2] + ray_dir[2] * self.max_lidar_distance
            ]
            
            result = p.rayTest(uav_pos, ray_end)[0]
            hit_fraction = result[2]
            
            if hit_fraction < 1.0:
                distance = hit_fraction * self.max_lidar_distance
                if self.render_mode == "human":
                    hit_pos = result[3]
                    p.addUserDebugLine(uav_pos, hit_pos, [1, 0, 0], lifeTime=0.1)
            else:
                distance = self.max_lidar_distance
            
            readings.append(distance / self.max_lidar_distance)
        
        return np.array(readings, dtype=np.float32)

    def step(self, action):
        # Get current position FIRST
        current_pos = np.array(p.getBasePositionAndOrientation(self.uav_id)[0])
        
        # Apply action with momentum
        move_vector = np.clip(action, -1, 1) * self.max_speed
        self.uav_velocity = 0.6 * self.uav_velocity + 0.4 * move_vector
        
        # Calculate new position
        new_pos = current_pos + self.uav_velocity
        new_pos = np.clip(new_pos,
                        [self.x_range[0]+0.3, self.y_range[0]+0.3, self.z_range[0]+0.3],
                        [self.x_range[1]-0.3, self.y_range[1]-0.3, self.z_range[1]-0.3])
        
        # Update physics
        p.resetBasePositionAndOrientation(self.uav_id, new_pos.tolist(), [0, 0, 0, 1])
        p.stepSimulation()

        # Calculate rewards
        current_dist = self._calculate_target_distance()
        lidar_readings = self._get_lidar_readings()
        
        reward = -0.05
        distance_reward = (self.prev_dist - current_dist) * 30
        lidar_penalty = np.sum(np.clip(1.0 - lidar_readings, 0, 1)) * 0.15
        reward += distance_reward - lidar_penalty - 0.1

        # Check termination conditions
        done = False
        info = {}
        if self._check_collisions():
            reward -= 20
            done = True
            info['collision'] = True
        elif current_dist < 0.4:
            reward += 50
            done = True
            info['success'] = True

        self.prev_dist = current_dist
        
        return self._get_observation(), reward, done, False, info

    def _check_collisions(self):
        uav_pos = p.getBasePositionAndOrientation(self.uav_id)[0]
        for obstacle in self.obstacles:
            distance = np.linalg.norm(np.array(uav_pos) - np.array(obstacle['pos']))
            if distance < (self.collision_radius + obstacle['size']):
                return True
        return False

    def _calculate_target_distance(self):
        return np.linalg.norm(np.array(p.getBasePositionAndOrientation(self.uav_id)[0]) 
                         - np.array(self.target_pos))

    def _create_boundary_markers(self):
        line_color = [0.5, 0.5, 0.5]
        line_width = 2.0
        
        # Floor boundaries
        corners = [
            [self.x_range[0], self.y_range[0], 0],
            [self.x_range[1], self.y_range[0], 0],
            [self.x_range[1], self.y_range[1], 0],
            [self.x_range[0], self.y_range[1], 0]
        ]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1)%4], line_color, lineWidth=line_width)
        
        # Ceiling boundaries
        ceiling = self.z_range[1]
        ceiling_corners = [c.copy() for c in corners]
        for c in ceiling_corners:
            c[2] = ceiling
        for i in range(4):
            p.addUserDebugLine(ceiling_corners[i], ceiling_corners[(i+1)%4], line_color, lineWidth=line_width)
        
        # Vertical pillars
        for i in range(4):
            p.addUserDebugLine(corners[i], ceiling_corners[i], line_color, lineWidth=line_width)

    def render(self, mode='human'):
        if self.render_mode == "human":
            time.sleep(0.02)

    def close(self):
        p.disconnect(self.client)