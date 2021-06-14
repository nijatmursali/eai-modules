import numpy as np
import torch
import random



class ReplayBuffer:
    def __init__(self, memory_size, state_shape, action_shape):
        self.counter = 0
        self.memory_size = memory_size
        self.buffer = {
            'state':     np.zeros((memory_size, *state_shape)),
            'action':    np.zeros((memory_size, *action_shape)),
            'reward':    np.zeros(memory_size),
            'new_state': np.zeros((memory_size, *state_shape)),
            'terminal':  np.zeros(memory_size)
        }

    def push(self, experience):
        """

        :param experience: dictionary of experiences
        :return:
        """
        index = self.counter % self.memory_size
        for each_key in self.buffer.keys():
            self.buffer[each_key][index] = experience[each_key]
        self.counter += 1

    def sample(self, key, batch_size):

        key_batch = random.sample(list(self.buffer[key]), batch_size)

        return key_batch


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
        self.actor = model_parameters['actor']
        self.critic = model_parameters['critic']
        self.actor_target = model_parameters['actor']
        self.critic_target = model_parameters['critic']
        self.temporal_difference = model_parameters['loss']

        self.mem_size = agent_parameters['memory_size']     #int
        self.state_shape = agent_parameters['state_shape']  # tuple
        self.action_shape = agent_parameters['action_shape'] # tuple
        self.gamma = agent_parameters['gamma'] # float
        self.tau = agent_parameters['tau'] # float
        self.device = device

        self.memory = ReplayBuffer(self.mem_size, self.state_shape, self.action_shape)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = model_parameters['actor_optimizer']
        self.critic_optimizer = model_parameters['critic_optimizer']

    def get_action(self, state):
        # print(type(state))
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        # action.to('cpu')
        action = action.cpu().detach().numpy()[0]
        return action # np.array

    def update(self, batch_size):
        states = torch.FloatTensor(self.memory.sample('state', batch_size)).to(self.device)
        actions = torch.FloatTensor(self.memory.sample('action', batch_size)).to(self.device)
        rewards = torch.FloatTensor(self.memory.sample('reward', batch_size)).to(self.device)
        new_states = torch.FloatTensor(self.memory.sample('new_state', batch_size)).to(self.device)
        terminals = torch.Tensor(self.memory.sample('terminal', batch_size)).to(self.device)

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
        self.actor.save_checkpoint('actor_model', True)
        self.critic.save_checkpoint('critic_model')
        self.actor_target.save_checkpoint('actor_target_model')
        self.critic_target.save_checkpoint('critic_target_model')

    def load_models(self):
        self.actor.load_checkpoint('actor_model')
        self.critic.load_checkpoint('critic_model')
        self.actor_target.load_checkpoint('actor_target_model')
        self.critic_target.load_checkpoint('critic_target_model')