from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from ..base.baseclass import SYSIDBase
import numpy as np


class DimensionMismatchException(Exception):
    
    pass


class SYSIDRegressor(SYSIDBase, RegressorMixin):
    '''
    Classe para identificacao de sistemas em conjunto com o sklearn.
    Herda dos principais objetos do sklearn
    '''

    def __init__(self, nX, ny, preprocessor=None, estimator=LinearRegression(fit_intercept=False)):

        self.estimator = estimator
        super().__init__(nX, ny, preprocessor=preprocessor)
        

    def fit(self, X, y):
        X, y = self.fit_transform(X, y)
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y, steps_ahead=None):

        if steps_ahead == 1:
            yhat = self.OSA_predict(X, y)
        elif steps_ahead is None:
            yhat = self.FS_predict(X, y)
        else:
            yhat = self.nS_predict(X, y, steps_ahead)
        return yhat


    def fit_predict(self, X, y, steps_ahead=None):
        return self.fit(X, y).predict(X, y, steps_ahead)


    def OSA_predict(self, X, y):

        nx, ny, kx, ky, N, Nx, p = self.estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        X, y = self.transform(X, y)
        yhat[p-1:,:] = self.estimator.predict(X).reshape(-1,ky)
        return yhat

    def nS_predict(self, X, y, steps_ahead):

        nx, ny, kx, ky, N, Nx, p = self.estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        Phi, _ = self.transform(X, y)
        Phi_i = Phi[0,:].reshape(1,-1)
        for i in range(p-1,N):
            yhat[i,:] = self.estimator.predict(Phi_i).reshape(-1)
            if (1+i-p) % steps_ahead != 0:
                Phi, _ = self.transform(X[:i+1,:], yhat[:i+1,:])
            else:
                Phi, _ = self.transform(X[:i+1,:], y[:i+1,:])
            Phi_i = Phi[-1,:].reshape(1,-1)
        return yhat

    def FS_predict(self, X, y):

        nx, ny, kx, ky, N, Nx, p = self.estimate_parameters(X, y)
        y = y.reshape(-1,ky)
        yhat = np.zeros(y.shape)
        yhat[:p-1,:] = y[:p-1,:]
        Phi, y = self.transform(X, y)
        Phi_i = Phi[0,:].reshape(1,-1)
        for i in range(p-1,N):
            yhat[i,:] = self.estimator.predict(Phi_i).reshape(-1)
            Phi, _ = self.transform(X[:i+1,:], yhat[:i+1,:])
            Phi_i = Phi[-1,:].reshape(1,-1)
        return yhat






