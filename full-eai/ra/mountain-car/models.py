import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np



class Actor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)

        # self.linear = nn.Linear(input_size, hidden_size2)
        self.out = nn.Linear(hidden_size2, output_size)
        self.checkpoint_folder = 'models'

    def forward(self, state):
        out1 = F.relu(self.linear1(state))
        out2 = F.relu(self.linear2(out1))
        output = torch.tanh(self.out(out2))

        # out = F.relu(self.linear(state))
        # output = torch.tanh(self.out(out))

        return output

    def save_checkpoint(self, model_name, check_first=False):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        character = ''
        if check_first:
            character = '\n'
        print(f'{character}... saving checkpoint ...')
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, model_name):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, output_size)
        self.checkpoint_folder = 'models'
        # self.linear = nn.Linear(input_size, hidden_size2)
        # self.out = nn.Linear(hidden_size2, output_size)


    def forward(self, state, action):
        input_data = torch.cat([state, action], 1)
        out1 = F.relu(self.linear1(input_data))
        out2 = F.relu(self.linear2(out1))
        output = self.out(out2)

        # out = F.relu(self.linear(input_data))
        # output = torch.tanh(self.out(out))

        return output

    def save_checkpoint(self, model_name, check_first=False):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        character = ''
        if check_first:
            character = '\n'
        print(f'{character}... saving checkpoint ...')
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, model_name):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(checkpoint_file))




class Critic_after(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, output_size):
        super(Critic_after, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(action_dim, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.checkpoint_folder = 'models'
        # self.linear = nn.Linear(input_size, hidden_size2)
        # self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, state, action):
        # input_data = torch.cat([state, action], 1)
        out1 = self.linear1(state)
        out2 = self.linear2(action)
        output = self.out(F.relu(out1 + out2))

        # out = F.relu(self.linear(input_data))
        # output = torch.tanh(self.out(out))

        return output

    def save_checkpoint(self, model_name, check_first=False):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        character = ''
        if check_first:
            character = '\n'
        print(f'{character}... saving checkpoint ...')
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, model_name):
        checkpoint_file = os.path.join(self.checkpoint_folder, model_name)
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(checkpoint_file))