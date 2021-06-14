import torch
import os
import numpy as np

from agent_new import Ornstein_Uhlenbeck, Agent
import gym
from numpy import save
from tqdm import tqdm

environment = gym.make('MountainCarContinuous-v0')

device = 'cuda' if torch.cuda.is_available() else 'cpu:0'
tau = 0.001
gamma = 0.99
memory_size = 50000
num_episodes = 1000
num_steps = 500
batch_size = 64
hidden_size_1_actor = 20
hidden_size_2_actor = 20
hidden_size_1_critic = 40
hidden_size_2_critic = 30

learning_rate_actor = 0.001
learning_rate_critic = 0.002



model_parameters = {
    'lr_actor': learning_rate_actor,
    'lr_critic': learning_rate_critic,
    'hidden_size_1_actor': hidden_size_1_actor,
    'hidden_size_2_actor': hidden_size_2_actor,
    'hidden_size_1_critic': hidden_size_1_critic,
    'hidden_size_2_critic': hidden_size_2_critic,

}

agent_parameters = {
    'memory_size': memory_size,
    'state_shape': environment.observation_space.shape,
    'action_shape': environment.action_space.shape,
    'gamma': gamma,
    'tau': tau
}

ou_noise = Ornstein_Uhlenbeck(environment)
agent = Agent(model_parameters, agent_parameters, device)

rewards = list()
avg_rewards = list()
counter = 0
positive_rewards = 0
negative_rewards = 0
positive_rewards_all = 0
negative_rewards_all = 0
with_done = True
data_folder = 'Data'
data_path_avg = os.path.join(data_folder, 'avg_rewards.npy')
data_path_row = os.path.join(data_folder, 'rewards.npy')
exploration_rate = 3
for k in range(num_episodes):
    state = environment.reset()
    ou_noise.reset()
    done = False
    episode_reward = 0

    step = 0

    while not done:
        step += 1
        if (k + 1) % 25 == 0 and k != 0:
            environment.render()
        action = agent.get_action(state)
        action = ou_noise.get_action(action, step)
        # action = np.clip(np.random.normal(action, exploration_rate), -2, 2)
        new_state, reward, done, _ = environment.step(action)
        experience = {
            'state': state,
            'action': action,
            'reward': np.array(reward),
            'new_state': new_state,
            'terminal': np.array(done)
        }
        agent.memory.push(experience)

        if agent.memory.counter > batch_size:
            # exploration_rate *= 0.995
            agent.update(batch_size)

        state = new_state

        episode_reward += reward
        counter = step
        if done:
            break
    if episode_reward >= 0:
        positive_rewards += 1
        positive_rewards_all += 1
    else:
        negative_rewards += 1
        negative_rewards_all += 1
    character = '\t'
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-100:]))


    print(f'Episode: {k + 1}, '
          f'episode reward: {episode_reward: .2f} in '
          f'{counter +1} steps, '
          f'average reward through 100 episodes: {np.mean(rewards[-100:]): .2f}')
    if (k + 1) % 25 == 0 and k != 0:
        print(80*'*')
        print(f'Summary of recent 25 episodes: '
              f'positive rewards: {positive_rewards} and '
              f'negative rewards: {negative_rewards}')
        print(f'Summary of recent {k+1} episodes: '
              f'positive rewards: {positive_rewards_all} and '
              f'negative rewards: {negative_rewards_all}')
        save(data_path_avg, np.array(avg_rewards))
        print(f'Average rewards are saved for last {k+1} episodes ... ')
        save(data_path_row, np.array(rewards))
        print(f'Rewards are saved for last {k+1} episodes ... ')

        positive_rewards = 0
        negative_rewards = 0
        agent.save_models()

