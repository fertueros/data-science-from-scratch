import numpy as np

class LinearModel:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        raise NotImplementedError("Este metodo deber ser sobreescrito por la subclase")

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            raise Exception("El modelo no ha sido ajustado aun")

        return X.dot(self.coef_) + self.intercept_

class LinearRegression(LinearModel):
    def fit(self, X, y):
        X_with_intercept = np.hstack([np.ones((X.shape[0],1)), X])
        theta = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
