import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
import numpy as np
from statistics import mean, mode

from library.environment import CustomEnvironment
from library.agent import ActorCriticNetworkAgent

# 
# HERE WE SET THE INITIAL SETTINGS FOR THE TRAINING PROCESS
#
max_train_step = 3000000
epsilon = 0.1
gamma = 0.99
lambd = 0.95
lr = 3e-5

t_horizon = 32
k_epochs = 10
n_workers = 8

world = 1
stage = 1

model_path = './saved-models/SuperMarioBros_PPO_LSTM_{}-{}.model'.format(
    world, stage)
history_path = './history/history_PPO_LSTM_{}-{}'.format(
    world, stage)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)


def train(agent, optimizer, states, actions, rewards, dones, old_probs, final_state, start_h, start_c):
    states = torch.FloatTensor(states).to(device) 
    actions = torch.LongTensor(actions).view(-1, 1).to(device) 
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device) 
    dones = torch.FloatTensor(dones).to(device) 
    old_probs = torch.FloatTensor(old_probs).view(-1, 1).to(device)
    final_state = torch.FloatTensor(final_state).to(device)

    for _ in range(k_epochs):
        # Calculate Probs, values
        probs = []
        values = []
        h = start_h
        c = start_c

        for state, done in zip(states, dones):
            prob, value, (h, c) = agent(state, (h, c))
            probs.append(prob)
            values.append(value)
            for i, d in enumerate(done):
                if d.item() == 0:
                    h, c = reset_hidden(i, h, c)

        _, final_value, _ = agent(final_state, (h, c))
        next_values = values[1:]
        next_values.append(final_value)

        probs = torch.cat(probs)
        values = torch.cat(values)
        next_values = torch.cat(next_values)

        td_targets = rewards + gamma * next_values * \
            dones.view(-1, 1)
        deltas = td_targets - values 

        # calculate GAE
        deltas = deltas.view(t_horizon, n_workers,
                             1).cpu().detach().numpy()  # (T, N, 1)
        masks = dones.view(t_horizon, n_workers, 1).cpu().numpy()
        advantages = []
        advantage = 0
        for delta, mask in zip(deltas[::-1], masks[::-1]):
            advantage = gamma * lambd * advantage * mask + delta
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device)

        probs_a = probs.gather(1, actions)
        m = Categorical(probs)
        entropys = m.entropy()

        ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = F.smooth_l1_loss(values, td_targets.detach())
        entropy_loss = torch.mean(entropys)

        loss = actor_loss + critic_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def reset_hidden(i, h, c):
    filter_tensor = torch.ones_like(h)
    filter_tensor[i] = torch.zeros(512)
    h = h * filter_tensor
    c = c * filter_tensor
    return h, c

def main():
    env = CustomEnvironment(n_workers, world, stage)
    agent = ActorCriticNetworkAgent(7).to(device)
    #agent.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(agent.parameters(), lr)

    scores = [0.0 for _ in range(n_workers)]
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()

    step = len(train_history) * 1000

    state = env.reset()
    h = torch.zeros(n_workers, 512).to(device)
    c = torch.zeros(n_workers, 512).to(device)

    print("Training process has been started!")
    while step <= max_train_step:
        start_h = h
        start_c = c
        total_states, total_actions, total_rewards, total_dones, total_old_probs = list(), list(), list(), list(), list()
        for _ in range(t_horizon):
            prob, _, (next_h, next_c) = agent(
                torch.FloatTensor(state).to(device), (h, c))
            m = Categorical(prob)

            action = m.sample()
            old_prob = prob.gather(1, action.unsqueeze(1))

            action = action.cpu().detach().numpy()
            old_prob = old_prob.cpu().detach().numpy()

            next_state, reward, done = env.step(
                action)

            # save transition
            total_states.append(state) 
            total_actions.append(action) 
            total_rewards.append(reward/100.0) 
            total_dones.append(1-done)
            total_old_probs.append(old_prob) 

            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                scores[i] += r

                if d == True:
                    score_history.append(scores[i])
                    scores[i] = 0.0
                    if len(score_history) > 100:
                        del score_history[0]
                    next_h, next_c = reset_hidden(
                        i, next_h, next_c)  # if done, reset hidden

            state = next_state
            h = next_h.detach()
            c = next_c.detach()

            step += 1
            #print(score_history)
            #print(train_history)

            if step % 1000 == 0:
                train_history.append(mean(score_history))
                torch.save(agent.state_dict(), 'test.model')
                np.save(history_path, np.array(train_history))
                print("step : {}, world {}-{}, Average score of last 100 episode: {:.1f}".format(
                    step, world, stage, mean(score_history)))

        train(agent, optimizer, total_states, total_actions, total_rewards,
              total_dones, total_old_probs, state, start_h, start_c)

    torch.save(agent.state_dict(), 'test.model')
    np.save(history_path, np.array(train_history))
    print("Train end, avg_score of last 100 episode : {}".format(mean(score_history)))


if __name__ == "__main__":
    main()
