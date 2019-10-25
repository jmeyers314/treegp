import numpy as np

class poly_interp2d(object):

    def __init__(x, y, z, cov=None, order=3):

        self.x = x
        self.y = y
        self.z = z
        self.order = order

        if cov is None:
            self.w = np.eye(len(self.z))
        else:
            self.w = np.linalg.inv(cov)

        self.theta = None

    def _build_polynomial_series(self, x, y):

        X = np.ones((self.order + 2, len(x)))
        for k in range(self.order+1):
            for i in range(self.order-k):
                X[k] *= x
            for j in range(k):
                X[k] *= y
        return X

    def solve(self):
        X = self._build_polynomial_series(self.x, self.y)
        T = np.dot(X.T, self.w.dot(X))
        T_inv = np.linalg.inv(T)
        B = np.dot(X.T, self.w.dot(np.matrix(self.z).T))
        self.theta = (np.dot(T_inv, np.matrix(B).T)).T
 
    def predict(self, x, y):

        X = self._build_polynomial_series(self.x, self.y)
        z = np.dot(self.theta, X)
        return z
