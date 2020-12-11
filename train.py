# >> tensorboard --logdir=./tensorboard
import json
import gym
import datetime
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Json_Parser:
    def __init__(self, file_name):
        with open(file_name) as json_file:
            self.json_data = json.load(json_file)

    def load_parser(self):
        return self.json_data


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.parser = Json_Parser("config.json")
        h = self.parser.load_parser()['agent']['hidden_unit']
        self.fc1 = nn.Linear(8, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, 2)

    def forward(self, x):
        acti = self.parser.load_parser()['agent']['activation']
        if acti == "relu":
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        elif acti == "sigmoid":
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = self.fc3(x)
        elif acti == "softmax":
            x = F.softmax(self.fc1(x), dim=0)
            x = F.softmax(self.fc2(x), dim=0)
            x = self.fc3(x)
        return x


class ReplayMemory:
    def __init__(self, memory_size, keys):
        self.memory = {}
        for key in keys:
            self.memory[key] = collections.deque(maxlen=memory_size)
        self.memory_size = memory_size

    def save(self, observations):
        for i, key in enumerate(self.memory.keys()):
            self.memory[key].append(observations[i])

    def __len__(self):
        return len(self.memory['x'])

    def sample(self, idx):
        sub_memory = {}
        for key in self.memory.keys():
            sub_memory[key] = [self.memory[key][i] for i in idx]

        ss, actions, rs, ss_next, dones = sub_memory.values()
        ss = torch.stack(ss)
        ss_next = torch.stack(ss_next)
        rs = np.array(rs)
        rs = torch.from_numpy(rs).float()

        return (ss, actions, rs, ss_next, dones)


class DQNAgent:
    def __init__(self):
        super().__init__()

        date_time = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
        self.parser = Json_Parser("config.json")
        self.parm = self.parser.load_parser()
        self.method = self.parm['method']
        self.max_step = self.parm['max_step']
        self.discount_factor = self.parm['agent']['discount_factor']
        self.lr = self.parm['optimizer']['learning_rate']
        self.eps = self.parm['optimizer']['eps']
        self.eps_max = self.eps
        self.eps_min = self.parm['optimizer']['eps_min']
        self.eps_mid = self.parm['optimizer']['eps_mid']
        self.eps_anneal = self.parm['optimizer']['eps_anneal']
        self.episode_size = self.parm['episode_size']
        self.minibatch_size = self.parm['minibatch_size']
        self.net_update_period = self.parm['net_update_period']

        self.env = gym.make('{}'.format(self.parm['env_name']))
        self.net = Qnet()
        self.target_net = Qnet()
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.replay_memory = ReplayMemory(self.parm['memory_size'], keys=[
                                          'x', 'a', 'r', 'x_next', 'done'])
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=0)
        self.loss = nn.SmoothL1Loss()

        save_name = self.parm['env_name'] + '_' + self.method + '_' + self.parm['agent']['activation'] + '_' + \
            str(self.parm['agent']['hidden_unit']) + '_' + date_time
        self.writer = SummaryWriter('./result/tensorboard/' + save_name)
        self.net_save_path = './result/model/model_{}.pth'.format(save_name)
        self.writer.add_text('config', json.dumps(self.parm))

    def get_action(self, x):
        if np.random.rand() < self.eps:
            action = np.random.randint(2)
        else:
            self.net.eval()
            q = self.net(x.view(1, -1))
            action = np.argmax(q.detach().numpy())
        return action

    def epsilon_decaying(self):
        if self.eps > self.eps_mid:
            self.eps -= (self.eps_max-self.eps_mid)/self.eps_anneal
        if self.eps < self.eps_mid and self.eps > self.eps_min:
            self.eps -= (self.eps_mid-self.eps_min)/self.eps_anneal

    def train(self, running_loss):
        self.epsilon_decaying()

        self.net.train()
        minibatch_idx = np.random.choice(self.replay_memory.__len__(), self.minibatch_size)
        ss, actions, rs, ss_next, dones = self.replay_memory.sample(minibatch_idx)
        final_state_idx = np.nonzero(dones)

        if self.method == "double":
            with torch.no_grad():
                self.net.eval()
                q_next = self.net(ss_next)
                q_next_ = self.target_net(ss_next)

            self.net.train()
            self.optimizer.zero_grad()
            q = self.net(ss)
            q_next_max, q_next_argmax = torch.max(q_next, 1)
            v_next = torch.gather(q_next_, 1, q_next_argmax.view(-1, 1)).squeeze()

        if self.method == "vanilla":
            with torch.no_grad():
                q_next = self.target_net(ss_next)

            self.optimizer.zero_grad()
            q = self.net(ss)
            q_next_max, q_next_argmax = torch.max(q_next, 1)
            v_next = q_next_max

        v_next[final_state_idx] = 0
        q_target = rs + self.discount_factor*v_next
        actions = torch.tensor(actions).view(-1, 1)
        q_relevant = torch.gather(q, 1, actions).squeeze()

        loss = self.loss(q_relevant, q_target)
        loss.backward()
        self.optimizer.step()

        running_loss = loss.item() if running_loss == 0 else 0.99 * \
            running_loss + 0.01*loss.item()

        return running_loss

    def run(self):
        backprops_total = 0
        running_loss = 0
        latest_scores = collections.deque(maxlen=100)
        pass_score = self.max_step - 4

        s_now = self.env.reset()
        s_prev = s_now
        score = 0

        for episode in range(self.episode_size):
            episode += 1
            for step in range(self.max_step):
                x = torch.from_numpy(np.concatenate(
                    (s_now, s_now-s_prev))).float()
                a = self.get_action(x)

                s_next, r, done, _ = self.env.step(a)
                score += 1

                x_next = torch.from_numpy(
                    np.concatenate((s_next, s_next-s_now))).float()
                self.replay_memory.save((x, a, r, x_next, done))

                if done:
                    latest_scores.append(score)
                    score = 0
                    s_now = self.env.reset()
                    s_prev = s_now
                else:
                    s_prev = s_now
                    s_now = s_next

                if self.replay_memory.__len__() > self.minibatch_size:
                    running_loss = self.train(running_loss)
                    backprops_total += 1

                self.writer.add_scalar('memory_size', self.replay_memory.__len__(), episode)
                self.writer.add_scalar('epsilon', self.eps, episode)
                self.writer.add_scalar('running_loss', running_loss, episode)
                self.writer.add_scalar('avg_score', np.mean(latest_scores), episode)

                if backprops_total % self.net_update_period == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                if done and episode % 100 == 0:
                    print("episode: {} | memory_size: {:5d} | eps: {:.3f} | running_loss: {:.3f} | last 100 avg score: {:3.1f}".
                          format(episode, self.replay_memory.__len__(), self.eps, running_loss, np.mean(latest_scores)))
                    torch.save(self.net.state_dict(), self.net_save_path)

                    if np.mean(latest_scores) > pass_score:
                        print('Latest 100 average score: {}, pass score: {}, test is passed'.format(
                            np.mean(latest_scores), pass_score))
                        exit(0)

                if done:
                    break

        self.env.close()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
