from PySide6.QtCore import *
import numpy as np
import os
import torch
import numpy as np
from itertools import count
from matplotlib import pyplot as plt

from AOI_divide import area1
from pg_learning import PolicyGradient


class config():
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


def train(env, agent, config, signal=None):
    state_pool, action_pool, prob_pool, reward_pool = [], [], [], []
    rewards, ma_rewards = [], []
    # 循环epoch个回合
    for epoch in range(config.tra_eps):
        # 开始训练
        state = env.reset()
        ep_reward, best_reward = 0, 0
        for iter in range(config.max_iter):
            state = np.reshape(state, (config.h * config.w))
            if signal and iter % 50 == 0:
                signal.emit(state.reshape([100, 100]))
            action, prob = agent.choose_action(state)
            nxt_state, reward = env.step(action)
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

        if epoch % config.batch_size == 0:  # 更新
            env.render()
            agent.update(prob_pool, reward_pool)
            prob_pool, reward_pool = [], []  # state_pool.clear(); action_pool.clear();

    print("训练完毕！")

    # 保存训练参数
    if config.save_param:
        agent.save(config.save_param)

    return rewards, ma_rewards


def test(env, agent, config):
    # 输出测试结果
    state = env.reset()
    best_state, best_reward = state, 0
    for iter in range(config.max_iter):
        action, prob = agent.choose_action(state)
        nxt_state, reward = env.step(action)
        state = nxt_state

        if reward > best_reward:
            best_reward = reward
            best_state = nxt_state
    env.state = best_state
    # env.render(os.path.join(config.save_model),'best_divide.png')
    # print('best reward is ',best_reward)


def run(config, signal=None):
    # 创建环境
    env = area1(config)
    config.w = env.w
    config.h = env.h

    # 创建智能体
    agent = PolicyGradient(config)

    if config.train:
        rewards, ma_rewards = train(env, agent, config, signal)
        plot_rewards(rewards, ma_rewards, config.save_model)

    if config.test:
        test(env, agent, config)


class PGThread(QThread):
    update_signal = Signal(np.ndarray)

    def __init__(self, aoi_slot):
        super().__init__()
        self.update_signal.connect(aoi_slot)

    def run(self):
        cfg = config()

        torch.manual_seed(cfg.seed)
        if cfg.device == 'cuda':
            torch.cuda.manual_seed(cfg.seed)

        print('using ', cfg.device)

        # 创建保存模型文件夹
        if not os.path.exists('./save_model'):
            os.makedirs('./save_model')
            os.makedirs(os.path.join('./save_model/', cfg.name))
            cfg.save_model = os.path.join('./save_model/', cfg.name)

        # 更改载入模型、储存模型路径
        if cfg.init_param:
            cfg.init_param = os.path.join('./save_model/', cfg.name, cfg.init_param)
        if cfg.save_param:
            cfg.save_param = os.path.join('./save_model/', cfg.name, cfg.save_param)

        run(cfg, signal=self.update_signal)
