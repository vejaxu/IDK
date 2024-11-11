import numpy as np
import sys
# from iNNE_IK import *
from iNNE_IK_notation import *


# 计算多个分布基于近似最近邻的核矩阵
def idk_kernel_map(list_of_distributions, psi, t=100):
    """
    :param list_of_distributions:
    :param psi: subsample size
    :param t: iteration times
    :return: idk kernel matrix of shape (n_distributions, n_distributions)
    """

    # 每个分布在alldata中的起始，终止索引
    D_idx = [0]  # index of each distributions

    # 将所有分布数据整合到一个大的numpy数组中
    alldata = []
    n = len(list_of_distributions)

    for i in range(1, n + 1):
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        alldata += list_of_distributions[i - 1]
    alldata = np.array(alldata)

    inne_ik = iNN_IK(psi, t)
    all_ikmap = inne_ik.fit_transform(alldata).toarray()

    idkmap = []
    for i in range(n):
        idkmap.append(np.sum(all_ikmap[D_idx[i]:D_idx[i + 1]], axis=0) / (D_idx[i + 1] - D_idx[i]))
    idkmap = np.array(idkmap)

    return idkmap

# group anamoly detector
def idk_square(list_of_distributions, psi1,  psi2, t1=100, t2=100):
    idk_map1 = idk_kernel_map(list_of_distributions, psi1, t1)
    #np.save(idkmapsavepath + "/idkmap1_psi1_"+str(psi1)+".npy", idk_map1)
    inne_ik = iNN_IK(psi2, t2)
    idk_map2 = inne_ik.fit_transform(idk_map1).toarray()
    #np.save(idkmapsavepath + "/idkmap2_psi1_"+str(psi1)+"_psi2_" + str(psi2) + ".npy", idk_map2)
    idkm2_mean = np.average(idk_map2, axis=0) / t1
    idk_score = np.dot(idk_map2, idkm2_mean.T)
    return idk_score

# point anomaly detector
def idk_anomalyDetector(data, psi, t=100):
    inne_ik = iNN_IK(psi, t)
    ik_feature_map = inne_ik.fit_transform(data).toarray()
    idkm_mean = np.average(ik_feature_map, axis=0) / t
    idk_score = np.dot(ik_feature_map, idkm_mean.T)

    return ik_feature_map, idk_score