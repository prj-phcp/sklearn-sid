import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.dimension_operations import to_2D
from statsmodels.tsa.seasonal import seasonal_decompose, STL


class SeasonalDecomposeExpansor(BaseEstimator, TransformerMixin):


    def __init__(self,
                 period, 
                 model='additive', 
                 filt=None, 
                 two_sided=False, 
                 extrapolate_trend='freq',
                 keep_residuals=True):
        
        self.model = model
        self.filt = filt
        self.period = period
        self.two_sided = two_sided
        self.extrapolate_trend = extrapolate_trend

        self.keep_residuals = keep_residuals

    def fit(self, X):

        return self
    
    def transform(self, X):

        X = to_2D(X)
        decomp = seasonal_decompose(X,
                                    self.model,
                                    self.filt,
                                    self.period,
                                    self.two_sided,
                                    self.extrapolate_trend)
        
        X_transformed = []
        X_transformed.append(to_2D(decomp.trend))
        X_transformed.append(to_2D(decomp.seasonal))
        if self.keep_residuals:
            X_transformed.append(to_2D(decomp.resid))
        
        X_transformed = np.hstack(X_transformed)
        return X_transformed