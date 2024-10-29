from scipy.io import loadmat
from IDK2 import * 
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    """ name = "PenDigits"
    data = loadmat(f'dataset/data_{name}.mat')
    X = data['fea']
    y = data['gt']
    idk_score = idk_anomalyDetector(X, psi=500, t=100) """

    X = np.array([[0, 1],
                  [1, 2], 
                  [2, 3], 
                  [3, 4], 
                  [1000, 1000]])
    
    idk_score = idk_anomalyDetector(X, psi=3, t=10)

    print("idk_score: ")
    print(idk_score)

    # idk_score 越高越是异常点还是越低越是异常点 ? 