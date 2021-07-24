import time
import torch
import sys
from library.environment import MarioAgentEnvironment
from library.agent import ActorCriticNetworkAgent
import os 
world = sys.argv[1] 
stage = sys.argv[2] 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path =  os.path.join(BASE_DIR, f'mario/models/SuperMarioBros_PPO_LSTM_{world}-{stage}.model') 
#model_path = sys.argv[3]
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_device = torch.device('cpu')


def main():
    mario_env = MarioAgentEnvironment(world, stage)
    state = mario_env.reset()
    agent = ActorCriticNetworkAgent(7).to(torch_device)
    agent.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))

    result = 0
    h = torch.zeros(1, 512).to(torch_device)
    c = torch.zeros(1, 512).to(torch_device)
    done = False

    while not done:
        prob, _, (next_h, next_c) = agent(
            torch.FloatTensor([state]).to(torch_device), (h, c))
        action = torch.argmax(prob).item()
        next_state, reward, done, info = mario_env.step(action)
        mario_env.render()
        print(info)
        print(reward)
        result += reward
        state = next_state
        h = next_h.detach()
        c = next_c.detach()
        time.sleep(0.03)

    time.sleep(2)

    print('Final result is: {}'.format(result))


if __name__ == "__main__":
    main()
