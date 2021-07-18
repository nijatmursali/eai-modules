import numpy as np
import torch
import torch.nn as nn
from models import Actor, Critic
from torch.optim import Adam



class ReplayBuffer:
    def __init__(self, memory_size, state_shape, action_shape, device):
        self.counter = 0
        self.memory_size = memory_size
        self.state_memory = np.zeros((memory_size, *state_shape))
        self.action_memory = np.zeros((memory_size, *action_shape))
        self.rewards_memory = np.zeros(memory_size)
        self.new_state_memory = np.zeros((memory_size, *state_shape))
        self.terminals_memory = np.zeros(memory_size)
        self.device = device

    def push(self, experience):
        """

        :param experience: dictionary of experiences
        :return:
        """
        index = self.counter % self.memory_size
        self.state_memory[index] = experience['state']
        self.action_memory[index] = experience['action']
        self.rewards_memory[index] = experience['reward']
        self.new_state_memory[index] = experience['new_state']
        self.terminals_memory[index] = experience['terminal']
        self.counter += 1

    def sample(self, batch_size):
        chosen_data = np.random.choice(self.memory_size, size=batch_size)

        batch_state = torch.FloatTensor(self.state_memory[chosen_data]).to(self.device)
        batch_action = torch.FloatTensor(self.action_memory[chosen_data]).to(self.device)
        batch_reward = torch.FloatTensor(self.rewards_memory[chosen_data]).to(self.device)
        batch_new_state = torch.FloatTensor(self.new_state_memory[chosen_data]).to(self.device)
        batch_terminal = torch.Tensor(self.terminals_memory[chosen_data]).to(self.device)

        return batch_state, batch_action, batch_reward, batch_new_state, batch_terminal


class Ornstein_Uhlenbeck:
    def __init__(self, environment, mean=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mean = mean
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dimension = environment.action_space.shape[0]
        self.action_low = environment.action_space.low
        self.action_high = environment.action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mean

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mean - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t/self.decay_period)
        return np.clip(action + ou_state, self.action_low, self.action_high)



class Agent:
    def __init__(self, model_parameters, agent_parameters, device):
        lr_actor = model_parameters['lr_actor']
        lr_critic = model_parameters['lr_critic']
        hs1_actor = model_parameters['hidden_size_1_actor']
        hs2_actor = model_parameters['hidden_size_2_actor']
        hs1_critic = model_parameters['hidden_size_1_critic']
        hs2_critic = model_parameters['hidden_size_2_critic']
        temporal_difference = nn.MSELoss()

        self.mem_size = agent_parameters['memory_size']     #int
        self.state_shape = agent_parameters['state_shape']  # tuple
        self.action_shape = agent_parameters['action_shape'] # tuple
        num_states = self.state_shape[0]
        num_actions = self.action_shape[0]
        self.gamma = agent_parameters['gamma'] # float
        self.tau = agent_parameters['tau'] # float
        self.device = device

        self.actor = Actor(num_states, hs1_actor, hs2_actor, num_actions).to(device)
        self.actor_target = Actor(num_states, hs1_actor, hs2_actor, num_actions).to(device)
        # self.critic = Critic_after(num_states, num_actions, hs1_critic, num_actions).to(device)
        # self.critic_target = Critic_after(num_states, num_actions, hs1_critic, num_actions).to(device)

        self.critic = Critic(num_states + num_actions, hs1_critic, hs2_critic, num_actions).to(device)
        self.critic_target = Critic(num_states + num_actions, hs1_critic, hs2_critic, num_actions).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.temporal_difference = nn.MSELoss()

        self.memory = ReplayBuffer(self.mem_size, self.state_shape, self.action_shape, device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)


    def get_action(self, state):
        # print(type(state))
        state = torch.FloatTensor(state).to(self.device)
        # state = torch.unsqueeze(state, 0)
        action = self.actor(state)
        # action.to('cpu')
        action = action.cpu().detach().numpy()[0]
        return action # np.array

    def update(self, batch_size):
        states, actions, rewards, new_states, terminals = self.memory.sample(batch_size)

        # compuations are needed to compute critic loss (i.e., temporal difference loss)
        q_values = self.critic(states, actions)
        next_actions = self.actor_target(new_states)
        nextQ = self.critic_target(new_states, next_actions.detach())
        rewards = torch.unsqueeze(rewards, 1)
        terminals = torch.unsqueeze(terminals, 1)


        q_values_target = rewards + (1-terminals) * self.gamma * nextQ

        td_loss = self.temporal_difference(q_values, q_values_target)

        # computation of policy loss
        policy_loss = -self.critic(states, self.actor(states)).mean()

        # update actor network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        td_loss.backward()
        self.critic_optimizer.step()
        # update actor target and critic target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # for x in self.actor_target.state_dict().keys():
        #     eval('self.actor_target.' + x + '.data.mul_((1-self.tau))')
        #     eval('self.actor_target.' + x + '.data.add_(self.tau*self.actor.' + x + '.data)')
        # for x in self.critic_target.state_dict().keys():
        #     eval('self.critic_target.' + x + '.data.mul_((1-self.tau))')
        #     eval('self.critic_target.' + x + '.data.add_(self.tau*self.critic.' + x + '.data)')

    def save_models(self):
        self.actor.save_checkpoint('actor_model')
        self.critic.save_checkpoint('critic_model')
        self.actor_target.save_checkpoint('actor_target_model')
        self.critic_target.save_checkpoint('critic_target_model')
        print(80 * '*')

    def load_models(self):
        self.actor.load_checkpoint('actor_model')
        self.critic.load_checkpoint('critic_model')
        self.actor_target.load_checkpoint('actor_target_model')
        self.critic_target.load_checkpoint('critic_target_model')