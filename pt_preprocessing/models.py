import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.05, max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        self.log = []
        self._init_weights(X)
        n = X.shape[0]
        for epoch in range(self.max_iter):
            y_pred = X @ self.weights + self.bias
            err = y_pred - y
            try:
                self.weights -= self.alpha * (err.T @ X).T / n
            except ValueError as e:
                print("err.T @ X shape:", (err.T @ X).shape)
                print("weights shape:", self.weights.shape)
                print(e)
                break
            self.bias -= self.alpha * np.mean(err)
                
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def _init_weights(self, X):
        self.weights = np.random.randn(X.shape[1], 1) / 100
        if len(self.weights.shape) == 1:
            self.weights = self.weights.reshape(1, -1)
        self.bias = 1

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.train_X = None
        self.train_y = None
    
    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
    
    def predict(self, X):
        if X.shape[0] == 0:
            return X.squeeze()
        d = self._euclid_dist(self.train_X, X)
        knl = self._k_nearest_labels(d, self.train_y)
        if self.k == 1:
            return knl.squeeze()[:, np.newaxis]
        else:
            return np.array([np.argmax(np.bincount(y.squeeze().astype(np.int64))) for y in knl])
    
    def _euclid_dist(self, X_known, X_unknown):
        sqrt = np.sqrt
        sm = np.sum
        return np.array([sqrt(sm((x - X_unknown) ** 2, axis=1)) for x in X_known]).T
    
    def _k_nearest_labels(self, dists, y_known):
        num_pred = dists.shape[0]
        n_nearest = []
        closest_y = None
        for j in range(num_pred):
            dst = dists[j]
            closest_y = y_known[np.argsort(dst)][:self.k]
            n_nearest.append(closest_y)
        return np.asarray(n_nearest)