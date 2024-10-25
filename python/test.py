from scipy.io import loadmat
from IDK2 import * 
from sklearn.metrics import roc_auc_score


if __name__ == '__main__':
    X = np.array([[1, 2],
                  [2, 3],
                  [1111111, 11111111]])
    
    idkscore = idk_anomalyDetector(X, 2, 3)

    print(idkscore)