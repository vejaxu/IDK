from scipy.io import loadmat
from IDK2 import * 
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    X = np.array([[0, 1],
                  [1, 2], 
                  [2, 3], 
                  [3, 4], 
                  [1000, 1000]])
    
    ik_feature_map, idk_score = idk_anomalyDetector(X, psi=2, t=5)

    mu_r = np.mean(idk_score)
    sigma_r = np.var(idk_score)

    print("ik_feature_map: ")
    print(ik_feature_map)

    print("idk_score: ")
    print(idk_score)

    print(f"mu_r: {mu_r}")
    print(f"sigma_r: {sigma_r}")