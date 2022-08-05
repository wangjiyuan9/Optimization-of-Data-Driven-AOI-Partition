import numpy as np

class artificial_env():
    def __init__(self):
        self.h,self.w = 100,100 #h*w 网格
        num_AOI = 10
        self.parcel_num = 10

        #AOI 区域生成/导入 100*100 对应所属AOI
        self.grid = np.ones(self.h,self.w)
        #快递员服务AOI集合生成/导入 二维列表，n个快递员，每个快递员负责一些AOI
        self.couriers = [[1,2,3],[2,4,5],[4,6,7],[7,8,10],[9]]
        self.courier_grid_lst = []
        for courier in self.couriers:
            self.courier_grid_lst = np.argwhere(self.grid==courier)

    def step(self,action):
        pass

    def update(self,action_pool,reward_pool,done_pool):
        pass

    def init_trace(self):
        # 计算网格距边界的距离
        border_dis = np.zeros_like(self.grid)
        for i in range(self.h):
            for j in range(self.w):
                border_dis[i,j]=self.find_border_dis(border_dis,i,j)
        self.border_dis = border_dis

        for i in range(len(self.couriers)):
            # 每个快递员循环
            courier_grid_xy = self.courier_grid_lst[i]
            # 在其服务区域内随机生成k个快递（注意边界区域生成的包裹概率大一点）
            parcels = self.init_parcel(num=self.parcel_num,courier_grid_xy=courier_grid_xy)

            # 规划轨迹
            trace = self.get_trace(parcels)


    def get_trace(self,parcels):
        # 先对服务区域进行排序
        # 再对每个区域内部包裹进行排序
        # 得到轨迹数据集

        # 或者状压dp也行
        pass

    def render(self):
        # 绘图
        pass

    def AOI_crossed(self,x1,y1,x2,y2):
        num = 0
        trace1,trace2 = [],[]
        delta_x = 1 if x1<x2 else -1
        delta_y = 1 if y1<y2 else -1
        # 先横后竖
        AOIs = set()
        for x in range(x1,x2+1,delta_x):
            AOIs.add(self.grid[x,y1])
            trace1.append([x,y1])
        for y in range(y1,y2+1,delta_y):
            AOIs.add(self.grid[x2,y])
            trace1.append([x2,y])
        num = len(AOIs)

        # 先竖后横
        AOIs.clear()
        for y in range(y1,y2+1,delta_y):
            AOIs.add(self.grid[x1,y])
            trace2.append([x1,y])
        for x in range(x1,x2+1,delta_x):
            AOIs.add(self.grid[x,y2])
            trace2.append([x,y2])

        if num<=len(AOIs):
            trace = trace1
        else:
            trace = trace2
            num=len(AOIs)
        return num,trace

    def find_border_dis(self,border_dis,i,j):
        dirs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        l = 0
        while not border_dis[i, j] and l<max(self.h,self.w):
            for dir in dirs:
                if i + l*dir[0] > 0 and i + l*dir[0] < self.h and j + l*dir[1] > 0 and j + l*dir[1] < self.w:
                    if self.grid[i,j]!=self.grid[i+l*dir[0],j+l*dir[1]]:
                        return l
            l+=1
        return l

    def init_parcel(self,num,courier_grid_xy): # courier_grid_xy:n*2 (xy)
        courier_grid = courier_grid_xy[:,0]*self.w + courier_grid_xy[:,1]
        courier_grid_p = self.border_dis[courier_grid_xy[:,0],courier_grid_xy[:,1]] #n*1
        courier_grid_p = np.max(courier_grid_p) - courier_grid_p + 1 # 设置边界和内部权重，再加个偏移量
        courier_grid_p = courier_grid_p/np.sum(courier_grid_p)
        parcels = np.random.choice(courier_grid, size=num, replace=False, p=courier_grid_p)
        return parcels


