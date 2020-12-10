import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class test_Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.fc1 = nn.Linear(8, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 2)

    # you have to change options according to train activation function (it's a hassle)
    def forward(self, x):
        # # relu
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # sigmoid
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)

        # # softmax
        # x = F.softmax(self.fc1(x), dim=0)
        # x = F.softmax(self.fc2(x), dim=0)
        # x = self.fc3(x)

        return x


def test():
    env = gym.make('CartPole-v0')
    render = True

    net = test_Qnet()
    net.load_state_dict(torch.load('./result/model/model_CartPole-v0_double_sigmoid_256_20201208-18_36_24.pth'))
    net.eval()

    s_cur = env.reset()
    s_prev = s_cur
    score = 0

    while True:
        if render:
            env.render()
        x = torch.from_numpy(np.concatenate((s_cur, s_cur - s_prev))).float()
        q = net(x.view(1, -1)).squeeze()
        qmax, a = torch.max(q, 0)
        a = a.item()
        s_prev = s_cur
        s_cur, r, done, _ = env.step(a)
        score += r

        if done:
            break

    env.close()
    print("Test score: {}".format(score))


if __name__ == "__main__":
    test()
