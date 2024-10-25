from scipy.io import loadmat
from IDK2 import * 
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    # Load data
    name = "PenDigits"
    data = loadmat(f'dataset/data_{name}.mat')
    X = data['fea']
    y = data['gt']
    idk_score = idk_anomalyDetector(X, psi=500, t=100)


    idk_score_array = np.array(idk_score)
    top_20_indices = np.argsort(idk_score_array)[-20:][::-1]
    top_20_y_values = np.array(y)[top_20_indices]


    print(f'IDK score: {idk_score}')
    print(f"top 20 indices: {top_20_y_values}")