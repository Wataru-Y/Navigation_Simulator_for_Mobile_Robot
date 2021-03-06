import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from nsmr.envs.consts import *
from nsmr.envs.renderer import Renderer
from nsmr.envs.nsmr import NSMR

class NsmrGymEnv2(gym.Env):
    def __init__(self,
                 layout=SIMPLE_MAP,
                 reward_params={"goal_reward": 5.0,
                                "collision_penalty": 5.0,
                                "alpha": 0.3,
                                "beta": 0.01,
                                "stop_penalty": 0.05},
                 ):
        # simulator
        self.nsmr = NSMR(layout)

        # gym space
        self.observation_space = spaces.Dict(dict(
            lidar=spaces.Box(low=MIN_RANGE, high=MAX_RANGE, shape=(NUM_LIDAR,), dtype=np.float32),
            target=spaces.Box(np.array([MIN_TARGET_DISTANCE,-1.0,-1.0]), np.array([MAX_TARGET_DISTANCE,1.0,1.0]), dtype=np.float32)
        ))
        self.action_space = spaces.Box(
            low = np.array([MIN_LINEAR_VELOCITY,MIN_ANGULAR_VELOCITY]),
            high = np.array([MAX_LINEAR_VELOCITY,MAX_ANGULAR_VELOCITY]),
            dtype = np.float32
            )

        # renderer
        self.renderer = Renderer(self.nsmr.dimentions)

        # reward params
        self.reward_params = reward_params
        self.mode='rgb_array'

        self.reset()

    def set_reward_params(self, reward_params):
        self.reward_params = reward_params
        self.reset()

    def set_layout(self, layout):
        self.nsmr.set_layout(layout)
        self.renderer = Renderer(self.nsmr.dimentions)
        self.reset()

    def reset(self):
        self.t = 0
        self.nsmr.reset_pose()
        self.nsmr.reset_noise_param()
        observation = self.get_observation_()
        self.pre_dis = observation["target"][0]
        self.goal = False
        self.final_target = self.nsmr.target.copy()
        self.theta = 0
        self.pre_pose = self.nsmr.pose
        
        return observation

    def step(self, action):
        self.t += 1
        self.pre_pose = self.nsmr.pose
        self.pre_dis = self.nsmr.get_relative_target_position()[0]
        self.nsmr.update(action)
        observation = self.get_observation()
        self.reward = self.get_reward(observation)
        done = self.is_done(observation)
        info = {"pose": self.nsmr.pose, "target": self.nsmr.target}
        
        return observation, self.reward, done, info

    def render(self, target, mode='human'):
        self.renderer.render(self.nsmr, self.mode, target)

    def get_observation(self):
        observation = {}
        observation["lidar"] = self.nsmr.get_lidar()
        #observation["target"] = self.nsmr.get_relative_target_position()
        observation["target"] = self.nsmr.get_subgoal2()
        return observation

    def get_observation_(self):
        observation = {}
        observation["lidar"] = self.nsmr.get_lidar()
        observation["target"] = self.nsmr.get_relative_target_position()
        #observation["target"] = self.nsmr.get_subgoal()
        return observation

    def get_reward(self, observation):
        #self.theta = np.arctan2(observation["target"][2], observation["target"][3])
        relative_target = self.nsmr.get_relative_target_position()
        #ddis = self.nsmr.get_relative_target_position(self.pre_pose, self.nsmr.pose)[0]
        ddis = self.pre_dis - relative_target[0]
        theta2 = np.arctan2(relative_target[1], relative_target[2])
        #reward = 0.1 * np.sqrt((observation["target"][0])**2 + (observation["target"][1])**2 + (theta)**2)
        reward = 0

        if relative_target[0] < ROBOT_RADIUS:
            reward += 5
        else:
            reward += 0.15 * ddis + 0.01 * abs(theta2)/(2*np.pi)
        
        if(abs(ddis)<1e-6):
            reward -= 0.05
        self.pre_dis = relative_target[0]
        
        return reward

    def is_done(self, observation):
        done = False
        if self.t >= MAX_STEPS:
            done = True
        elif self.nsmr.is_collision():
            done = True
        #if self.goal:
        #    done = True
        #if self.reward < 0.5:
        #if self.nsmr.get_relative_target_position()[0] < 0.2 and np.rad2deg(np.abs(self.theta)) < 5:
        elif self.nsmr.get_relative_target_position()[0] < ROBOT_RADIUS:
            done = True
        elif self.nsmr.get_relative_target_position(target=self.final_target)[0] < ROBOT_RADIUS:
            done = True
        return done

    def close(self):
        self.renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
