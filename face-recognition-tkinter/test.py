import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)

total_reward = 0
done = True
for step in range(100000):
    env.render()
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(info)
    total_reward += reward
env.close()
