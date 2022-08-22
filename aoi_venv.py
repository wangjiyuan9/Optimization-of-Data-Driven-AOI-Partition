from inspect import trace
import numpy as np
from matplotlib import pyplot as plt
import os

current_path = os.path.dirname(__file__)


class AOIVirtualEnv:
    def __init__(self, config):
        self.config = config
        self.w, self.h = config.w, config.h
        self.state = np.arange(self.w * self.h).reshape([self.h, self.w])
        # self.matrix = np.load('matrix.npy')
        self.trace = np.load('./data/trace/trace_1.npy', allow_pickle=True)
        self.tracetmp = np.load('./data/trace/trace_2.npy', allow_pickle=True)
        self.trace = np.append(self.trace, self.tracetmp)
        self.tracetmp = np.load('./data/trace/trace_3.npy', allow_pickle=True)
        self.trace = np.append(self.trace, self.tracetmp)
        self.tracetmp = np.load('./data/trace/trace_4.npy', allow_pickle=True)
        self.trace = np.append(self.trace, self.tracetmp)
        self.tracetmp = np.load('./data/trace/trace_5.npy', allow_pickle=True)
        self.trace = np.append(self.trace, self.tracetmp)
        # matrix_1 = self.matrix.copy().T
        # self.matrix = self.matrix+matrix_1
        self.create_map(self.trace)  # !!

    def reset(self):
        self.state = np.arange(self.w * self.h).reshape([self.h, self.w])
        return self.state

    def step(self, action):
        dir_h, dir_w = [-1, 1, 0, 0, 0], [0, 0, -1, 1, 0]
        if action.shape[0] == 1:  # 变化一个点
            # action 应当为一个要变化的点的坐标， 上下左右中，中代表不变化，上下左右代表合并到对应方向的格子
            point_h, point_w, act = action[0, 0], action[0, 1], action[0, 2]
            if 0 <= point_h + dir_h[act] < self.h and point_w + dir_w[act] >= 0 and point_w + \
                    dir_w[act] < self.w:
                self.state[point_h][point_w] = self.state[point_h + dir_h[act]][point_w + dir_w[act]]
        else:  # 变化全部点
            state_temp = self.state.copy()
            for i in range(self.h):
                for j in range(self.w):
                    act = action[i, j]
                    if 0 <= i + dir_h[act] < self.h and j + dir_w[act] >= 0 and j + dir_w[
                        act] < self.w:
                        state_temp[i][j] = self.state[i + dir_h[act]][j + dir_w[act]]
            self.state = state_temp
        # print("完成一步\n")
        reward = self.cal_reward()
        print(reward)
        return self.state, reward

    def create_map(self, trace):
        self.map = np.zeros([self.h, self.w, 2])
        for i in range(trace.shape[0]):
            for j in range(len(trace[i]) - 1):
                # print(j)
                if trace[i][j][0] - trace[i][j + 1][0] != 0:
                    self.map[min(trace[i][j][0], trace[i][j + 1][0])][trace[i][j][1]][0] += 1
                else:
                    self.map[trace[i][j][0]][min(trace[i][j][1], trace[i][j + 1][1])][1] += 1

    def cal_reward(self):
        # 轨迹跨过的AOI尽可能少——大于阈值且同AOI的轨迹奖励+1, 小于阈值且同AOI的轨迹奖励-1，大于阈值且异AOI的轨迹奖励-1
        threshold = self.config.threshold
        '''check_1 = np.argwhere(self.matrix>threshold)
        check_2 = np.argwhere(self.matrix<=threshold & self.matrix>0)
        situation_1 = np.sum( self.state[check_1[0] // self.w][check_1[0]%self.w] == self.state[check_1[1] // self.w][check_1[1]%self.w] )
        situation_2 = np.sum( self.state[check_2[0] // self.w][check_2[0]%self.w] == self.state[check_2[1] // self.w][check_2[1]%self.w] )
        situation_3 = np.sum( self.state[check_1[0] // self.w][check_1[0]%self.w] != self.state[check_1[1] // self.w][check_1[1]%self.w] )
        reward = situation_1-situation_2-situation_3'''

        s1, s2, s3 = 0, 0, 0
        dir_w, dir_h = [1, 0], [0, 1]
        for i in range(self.h):
            for j in range(self.w):
                for k in range(2):
                    i2, j2 = i + dir_h[k], j + dir_w[k]
                    if i2 >= 0 and i2 < self.h and j2 >= 0 and j2 < self.w:
                        if self.state[i, j] == self.state[i2, j2]:
                            if self.map[i, j, k] >= threshold:
                                s1 += 1
                            else:
                                s2 -= 1
                        else:
                            if self.map[i, j, k] >= threshold:
                                s3 -= 1
        reward = s1 + s2 + s3

        return reward

    def render(self, save=None):
        # plt.figure(figsize=(10,10))

        plt.show()
        if save:
            plt.savefig(save)
