import os
import torch
import numpy as np
from itertools import count
from matplotlib import pyplot as plt

from aoi_venv import AOIVirtualEnv
from aoi_agent import AOIAgent


class Config:
    def __init__(self):
        # 运行
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 10
        self.save_param = 'AOI_divide1_1.pth'
        self.init_param = None
        self.name = 'AOI_divide1'
        self.train = True
        self.test = True

        # 训练
        self.batch_size = 8
        self.tra_eps = 80
        self.max_iter = 400
        # 智能体
        self.gamma = 0.8
        self.lr = 0.1
        self.hidden_dim = 64
        # 环境
        self.h = 100
        self.w = 100
        self.threshold = 5


class PolicyGradientRL:
    def __init__(self):
        self.signal = None
        self.config = Config()
        self.agent = AOIAgent(self.config)
        self.env = AOIVirtualEnv(self.config)
        self.config.w = self.env.w
        self.config.h = self.env.h

        torch.manual_seed(self.config.seed)
        if self.config.device == 'cuda':
            torch.cuda.manual_seed(self.config.seed)

        # 创建保存模型文件夹
        if not os.path.exists('./save_model'):
            os.makedirs('./save_model')
            os.makedirs(os.path.join('./save_model/', self.config.name))
            self.config.save_model = os.path.join('./save_model/', self.config.name)

        # 更改载入模型、储存模型路径
        if self.config.init_param:
            self.config.init_param = os.path.join('./save_model/', self.config.name, self.config.init_param)
        if self.config.save_param:
            self.config.save_param = os.path.join('./save_model/', self.config.name, self.config.save_param)

    def execute(self, signal=None):
        """
        执行起点，由线程调用
        相当于之前的 main
        """
        self.signal = signal
        print('using ', self.config.device)
        if self.config.train:
            rewards, ma_rewards = self.train()
            self.plot_rewards(rewards, ma_rewards)

        if self.config.test:
            self.test()

    def train(self):
        state_pool, action_pool, prob_pool, reward_pool = [], [], [], []
        rewards, ma_rewards = [], []
        # 循环epoch个回合
        for epoch in range(self.config.tra_eps):
            # 开始训练
            state = self.env.reset()
            ep_reward, best_reward = 0, 0
            for iter in range(self.config.max_iter):
                state = np.reshape(state, (self.config.h * self.config.w))
                if self.signal and iter % 50 == 0:
                    self.signal.emit(state.reshape([100, 100]))
                action, prob = self.agent.choose_action(state)
                nxt_state, reward = self.env.step(action)
                prob_pool.append(prob)
                reward_pool.append(reward)  # state_pool.append(state); action_pool.append(action);
                state = nxt_state
                ep_reward += reward
                best_reward = max(best_reward, reward)

            print('epoch{},reward:{},best_reward:{}'.format(epoch, ep_reward, best_reward))
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)

            if epoch % self.config.batch_size == 0:  # 更新
                self.env.render()
                self.agent.update(prob_pool, reward_pool)
                prob_pool, reward_pool = [], []  # state_pool.clear(); action_pool.clear();

        print("训练完毕！")

        # 保存训练参数
        if self.config.save_param:
            self.agent.save(self.config.save_param)

        return rewards, ma_rewards

    def test(self):
        # 输出测试结果
        state = self.env.reset()
        best_state, best_reward = state, 0
        for iter in range(self.config.max_iter):
            action, prob = self.agent.choose_action(state)
            nxt_state, reward = self.env.step(action)
            state = nxt_state

            if reward > best_reward:
                best_reward = reward
                best_state = nxt_state
        self.env.state = best_state
        # env.render(os.path.join(config.save_model),'best_divide.png')
        # print('best reward is ',best_reward)

    def save(self):
        self.agent.save(self.config.save_param)

    def load(self):
        self.agent.load(self.config.init_param)

    def plot_rewards(rewards, ma_rewards, path):
        fig = plt.figure(1, figsize=[16, 10])
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title('reward')
        ax1.plot(rewards)

        ax2 = fig.add_subplot(2, 1, 2)  # 第2行第1列
        ax2.set_title('ma_reward')
        ax2.plot(ma_rewards)

        plt.show()
        plt.savefig(os.path.join(path, 'reward.png'))
