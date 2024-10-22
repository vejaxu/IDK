import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

# 近似最近邻算法，对高维数据进行特征变换与表示
class iNN_IK:
    data = None # input data
    centroid = [] # 每轮迭代中的随机子集索引
    def __init__(self, subset_size, t):
        self.subset_size = subset_size # 每轮迭代选择的子集大小
        self.t = t # 迭代次数

    def fit_transform(self, data):
        self.data = data # 存储输入数据
        # 用于存储每次迭代选取的子集以及对应的半径
        self.centroid = [] # initial
        self.centroids_radius = [] # initial

        sn = self.data.shape[0]
        n, d = self.data.shape

        IDX = np.array([])  # 用来存储稀疏矩阵的列索引
        V = [] # 用来存储稀疏矩阵中的值（即二元特征值 1 或 0）

        for i in range(self.t):
            subIndex = sample(range(sn), self.subset_size) # 随机选取子集
            self.centroid.append(subIndex) # 随机子集索引
            tdata = self.data[subIndex, :] # 选取对应数据
            tt_dis = cdist(tdata, tdata) # 返回距离矩阵

            radius = [] # restore centroids' radius

            for r_idx in range(self.subset_size):
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx)
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
            nt_dis = cdist(tdata, self.data)
            centerIdx = np.argmin(nt_dis, axis=0)
            for j in range(n):
                V.append(int(nt_dis[centerIdx[j],j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.subset_size), axis=0)
        IDR = np.tile(range(n), self.t) #row index
        #V = np.ones(self.t * n) #value
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.subset_size))
        return ndata

    # training function
    # 根据输入数据 计算和存储 中心点以及对应的半径
    def fit(self, data):
        self.data = data
        self.centroid = []
        self.centroids_radius = []
        sn = self.data.shape[0]

        for i in range(self.t):
            subIndex = sample(range(sn), self.subset_size)
            self.centroid.append(subIndex)
            tdata = self.data[subIndex, :] # 当前迭代随机选择的子集数据
            tt_dis = cdist(tdata, tdata) # 计算距离
            radius = [] #restore centroids' radius

            for r_idx in range(self.subset_size): # 对每个子集点
                r = tt_dis[r_idx]
                r[r<0] = 0
                r = np.delete(r,r_idx) # 获取该点与其他子集点的距离
                radius.append(np.min(r)) # 选择最小距离作为半径
            self.centroids_radius.append(radius)

    # 将新数据转换为基于已经拟合的中心点和半径的特征表示
    def transform(self, newdata):
        assert self.centroid != None, "invoke fit() first!"

        n, d = newdata.shape

        IDX = np.array([])
        V = []
        
        for i in range(self.t):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            dis = cdist(tdata, newdata)
            centerIdx = np.argmin(dis, axis=0)
            for j in range(n):
                V.append(int(dis[centerIdx[j], j] <= radius[centerIdx[j]]))
            IDX = np.concatenate((IDX, centerIdx + i * self.subset_size), axis=0)
        IDR = np.tile(range(n), self.t)
        ndata = csr_matrix((V, (IDR, IDX)), shape=(n, self.t * self.subset_size))
        return ndata

