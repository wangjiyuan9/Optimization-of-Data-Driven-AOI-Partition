import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, rewards):
        return -torch.sum(torch.log(probs) * rewards)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        input_dim, output_dim, hidden_dim = config.h * config.w, 2 + 5, config.hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)  # 映射到0-1的区间
        return x


class AOIAgent:
    def __init__(self, config):
        # 载入初始参数
        self.gamma = config.gamma
        self.device = config.device
        self.net = MLP(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        self.loss_fun = RLLoss()
        self.max_iter = config.max_iter

        self.h = config.h
        self.w = config.w

    def choose_action(self, state):
        state = torch.from_numpy(state).to(self.device, dtype=torch.float32)
        action_net = self.net(state)
        probs = action_net[2:]
        action = torch.zeros([1, 3], dtype=torch.int)
        action_xy = action_net[:2]
        # prob, action= torch.max(probs) # 示例代码里不是这样，是按照概率选取
        action_policy = torch.multinomial(probs, num_samples=1, replacement=False)
        prob = probs[action_policy]
        action[0, :2] = action_xy
        action[0, -1] = action_policy
        # logp = prob.log()
        return action, prob

    def update(self, prob_pool, reward_pool):  # ,flag_pool
        # 计算回报
        reward_current = 0.
        for i in reversed(range(len(reward_pool))):
            '''if flag_pool[i]==True: #回合结束标志
                reward_current = reward_pool[i]'''
            reward_current = reward_current * self.gamma + reward_pool[i]
            reward_pool[i] = reward_current
            if i % self.max_iter == 0:
                reward_current = 0.

        # 标准化回报
        prob_pool = torch.tensor(prob_pool, dtype=torch.float32, device=self.device)
        reward_pool = torch.tensor(reward_pool, dtype=torch.float32, device=self.device)
        reward_pool = (reward_pool - torch.mean(reward_pool)) / torch.std(reward_pool)

        # 计算loss
        '''for i in range(reward_pool.size[0]):
            loss -= prob_pool[i]*reward_pool[i] #为啥是减'''
        '''loss=0.
        prob_pool = torch.tensor(prob_pool,dtype=torch.float32)
        loss = -sum(np.log(prob_pool)*reward_pool)'''
        loss = self.loss_fun(prob_pool, reward_pool)

        # 反向传播
        self.optimizer.zero_grad()
        loss.requires_grad = True  # https://blog.csdn.net/wu_xin1/article/details/116502378
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.net.state_dict(), path)
