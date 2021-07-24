import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetworkAgent(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticNetworkAgent, self).__init__()
        # We have 4 convolution layers. in each one we set the in and output channels, kernel size, stride and padding  
        self.convLayer1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.convLayer2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.convLayer3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.convLayer4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.output_size = self.get_output_size()

        self.myLSTM = nn.LSTMCell(self.output_size * self.output_size * 64, 512)

        self.PI = nn.Linear(512, num_actions)
        self.V = nn.Linear(512, 1)

        self.weight_init()

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def get_output_size(self):
        input_tensor = torch.FloatTensor(1, 1, 84, 84)
        output_tensor = self.convLayer4(self.convLayer3(
            self.convLayer2(self.convLayer1(input_tensor))))
        output_size = output_tensor.size()[-1]
        return output_size

    def forward(self, x, hidden_layer):
        x = F.relu(self.convLayer1(x))
        x = F.relu(self.convLayer2(x))
        x = F.relu(self.convLayer3(x))
        x = F.relu(self.convLayer4(x))

        x = x.view(-1, self.output_size * self.output_size * 64)

        h, c = self.myLSTM(x, hidden_layer)

        prob = self.PI(h)
        prob = F.softmax(prob, dim=-1)
        value = self.V(h)
        return prob, value, (h, c)