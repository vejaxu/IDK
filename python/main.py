from scipy.io import loadmat
from IDK2 import * 


if __name__ == '__main__':
    # Load data
    name = "PenDigits"
    data = loadmat(f'dataset/data_{name}.mat')
    X = data['fea']
    y = data['gt']


    idk_score = idk_anomalyDetector(X, psi=500, t=100)
    print(f'IDK score: {idk_score}')