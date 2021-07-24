from nes_py.wrappers import JoypadSpace #we import it for discrete action space
import gym_super_mario_bros #used to have mario environment
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT #we have 1 type of action
from gym import Wrapper #base class for all wrappers
import numpy as np #used for stacks 
import cv2 #used for detecting the frames 

#here we are dealing with frames we need to process. we convert each frame into gray color frame and resize the frames
class ProcessCustomFrames(Wrapper):
    def __init__(self, env, size=84, skip=4):
        super(ProcessCustomFrames, self).__init__(env)
        self.skip = skip
        self.size = size
        self.running = False

    def CustomFrame(self, f):
        f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        f = cv2.resize(f, (self.size, self.size)) / 255.0
        f = np.expand_dims(f, axis=0)
        return f

    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward

            if done:
                break
        state = self.CustomFrame(state)
        return state, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        state = self.CustomFrame(state)
        return state


def MarioAgentEnvironment(world, stage):
    env = gym_super_mario_bros.make(
        f'SuperMarioBros-{world}-{stage}-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = ProcessCustomFrames(env)
    return env


class CustomEnvironment:
    def __init__(self, N, world, stage):
        self.world = world
        self.stage = stage
        self.envs = [MarioAgentEnvironment(self.world, self.stage)
                     for _ in range(N)]

    def reset(self):
        obs = []
        for env in self.envs:
            ob = env.reset()
            obs.append(ob)
        return np.stack(obs)

    def step(self, actions):
        total_obs, total_rewards, total_done, total_infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            if done:
                ob = env.reset()
            total_obs.append(ob)
            total_rewards.append(reward)
            total_done.append(done)
            total_infos.append(info)
        return np.stack(total_obs), np.stack(total_rewards), np.stack(total_done)

    def render(self):
        for env in self.envs:
            env.render()
