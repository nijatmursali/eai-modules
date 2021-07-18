import time
import matplotlib.pyplot as plt
import torch

from library.Environment import CreateEnv
from library.Model import ActorCriticNet_MA

world = 1
stage = 1

model_path = './Models/SuperMarioBros_MA_1-1_COMPLEX_WITHENTROPY.model'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def convert_action(action):
    # Priority: right = left > A > B
    # action : [right, left, A, B]
    if action[0]:
        if action[1]:
            if action[2]:
                return 5  # A
            else:
                return 0  # NOOP
        else:
            if action[2]:
                if action[3]:
                    return 4  # right A B
                else:
                    return 2  # right A
            else:
                if action[3]:
                    return 3  # right B
                else:
                    return 1  # right
    else:
        if action[1]:
            if action[2]:
                if action[3]:
                    return 9  # left A B
                else:
                    return 7  # left A
            else:
                if action[3]:
                    return 8  # left B
                else:
                    return 6  # left
        else:
            if action[2]:
                return 5  # A
            else:
                return 0  # NOOP


def Test():
    env = CreateEnv(world, stage, 'COMPLEX')
    Net = ActorCriticNet_MA().to(device)
    Net.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))

    score = 0
    score_arr = []
    iter_arr = []
    i = 0
    state = env.reset()
    h = torch.zeros(1, 512).to(device)
    c = torch.zeros(1, 512).to(device)
    done = False

    while not done:
        i += 1
        prob, _, (next_h, next_c) = Net(
            torch.FloatTensor([state]).to(device), (h, c))
        action = prob > 0.5
        action = convert_action(action.squeeze().cpu().detach().numpy())

        next_state, reward, done, info = env.step(action)
        env.render()

        score += reward
        score_arr.append(score)
        iter_arr.append(i)

        state = next_state
        h = next_h.detach()
        c = next_c.detach()
        time.sleep(0.03)

    time.sleep(2)
    plt.plot(iter_arr, score_arr)
    plt.show()

    print('score : {}'.format(score))


if __name__ == "__main__":
    Test()
