from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.dimension_operations import to_1D
import numpy as np


class DimensionMismatchException(Exception):
    
    pass

class SYSIDBase(BaseEstimator, TransformerMixin):
    '''
    Classe para identificacao de sistemas em conjunto com o sklearn.
    Herda dos principais objetos do sklearn
    '''

    def __init__(self, nX, ny, preprocessor=None, add_static=False):

        self.nX = nX
        self.ny = ny
        self.add_static = add_static
        self.preprocessor=preprocessor

    def transform(self, X, y):
        if not self.preprocessor is None and not X is None:
            X = self.preprocessor.transform(X)
        y, X = self.matReg(X, y)
        return X, y


    def fit_transform(self, X, y):
        if not self.preprocessor is None and not X is None:
            X = self.preprocessor.fit_transform(X)
        y, X = self.matReg(X, y)
        return X, y

    def estimate_parameters(self, X, y):

        ny = self.ny
        if len(y.shape) == 1: y = y.reshape(-1,1)
        (N, ky) = y.shape
        if isinstance(ny, int):
            ny = [ny]*ky
        
        if X is None:
            nx = []
            Nx = N
            kx = 0
        else:
            nx = self.nX
            if len(X.shape) == 1: X = X.reshape(-1,1)
            (Nx,kx) = X.shape
            if isinstance(nx, int):
                nx = [nx]*kx

        p = np.max(ny+nx) + 1

        if N != Nx:
            raise DimensionMismatchException(f'Dimensions of input and output are mismatched. Dimensions of X are {X.shape} and y are {y.shape}')

        return nx, ny, kx, ky, N, Nx, p

    def matReg(self,X,Y):
        
        nx, ny, kx, ky, N, Nx, p = self.estimate_parameters(X, Y)
            
        # create target vector
        target = Y[p-1:,:]

        # create regression matrix
        Phi = np.zeros((N-p+1,np.sum(ny+nx)))
        counter = 0
        for i in range(ky):
            ny_i = ny[i]
            y = Y[:,i]
            for j in range(ny_i):
                Phi[:,counter] = y[p-j-2: N-j-1].reshape(-1)
                counter +=1
                #print(Phi.shape, counter)

        for i in range(kx):
            nx_i = nx[i]
            x = X[:,i]
            for j in range(nx_i):
                Phi[:,counter] = x[p-j-2: N-j-1].reshape(-1)
                counter +=1
                #print(Phi.shape, counter)
        if self.add_static:
            Phi = np.hstack((Phi, X[p-1:,:]))
        target = to_1D(target)
        return (target, Phi)
    

    






