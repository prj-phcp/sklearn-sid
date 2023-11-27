import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..utils.dimension_operations import to_2D
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import stft, istft



class FFTEnergyBaseExpansor(BaseEstimator, TransformerMixin):

    def __init__(self, energy_levels=2, drop=1.0, fftkwargs={}):

        self.energy_levels = energy_levels
        self.drop = drop
        self.treshold_frequencies = None
        self.n_variables = None
        self.fftkwargs = fftkwargs

    def fit(self, X):

        X = to_2D(X)
        self.treshold_frequencies = []
        _, self.n_variables = X.shape
        for i in range(self.n_variables):
            x = X[:,i]
            fftx, freq = self.fftransform(x)
            frequency_tresholds = self.get_frequency_tresholds(fftx, freq)
            self.treshold_frequencies.append(frequency_tresholds)
        
        return self

    def transform(self, X):

        X = to_2D(X)
        n_steps, _ = X.shape
        X_transformed = []
        for i in range(self.n_variables):
            x = X[:,i]
            fftx, freq = self.fftransform(x)
            frequency_tresholds = self.treshold_frequencies[i]
            fftx_split = self.split_signal(fftx, freq, frequency_tresholds)
            for fftxi in fftx_split:
                x_i = self.reverse_fftransform(fftxi, n_steps)
                X_transformed.append(to_2D(x_i))
        X_transformed = np.hstack(X_transformed)
        return X_transformed

    def get_energy_tresholds(self):

        energy_tresholds = np.linspace(0, 1.0, self.energy_levels+1, endpoint=True)
        energy_tresholds[-1] = self.drop
        return energy_tresholds

    def get_frequency_tresholds(self, fftx, freq):

        energy_tresholds = self.get_energy_tresholds()
        fftx = to_2D(fftx)
        cum_fftx = np.cumsum(np.abs(fftx)**2, axis=0)
        cum_fftx = cum_fftx / cum_fftx[-1,:].reshape(1,-1)
        avg_cum_fftx = np.average(cum_fftx, axis=1).ravel()
        frequency_tresholds = np.zeros(self.energy_levels+1)-1.0
        for i in range(1, len(energy_tresholds)):
            energy_treshold_min = energy_tresholds[i-1]
            energy_treshold_max = energy_tresholds[i]
            try:
                frequency_tresholds[i] = np.max(freq[(avg_cum_fftx>energy_treshold_min)&(avg_cum_fftx<=energy_treshold_max)])
            except ValueError:
                pass
        frequency_tresholds = np.unique(frequency_tresholds)
        return frequency_tresholds
    
    def split_signal(self, fftx, freq, tresholds):

        fftx = to_2D(fftx)
        fftx_split = []
        for i in range(1, len(tresholds)):
            treshold_min = tresholds[i-1]
            treshold_max = tresholds[i]
            fftxi = fftx.copy()
            fftxi[(freq<=treshold_min)|(freq>treshold_max),:] = 0.0
            fftx_split.append(fftxi)
        return fftx_split

    def fftransform(self, x):

        pass

    def reverse_fftransform(self, fftx):

        pass


class RFFTEnergyExpansor(FFTEnergyBaseExpansor):

    pass


class SFFTEnergyExpansor(FFTEnergyBaseExpansor):

    def fftransform(self, x):

        freq, t, fftx = stft(x, **self.fftkwargs)
        return fftx, freq

    def reverse_fftransform(self, fftx, n_steps):

        t, x = istft(fftx, **self.fftkwargs)
        x = x.ravel()
        if len(x) >= n_steps:
            x = x[:n_steps]
        return x

############################################################################################################

class FFTFrequencyBaseExpansor(BaseEstimator, TransformerMixin):

    def __init__(self, frequency=0.1, drop=1.0, fftkwargs={}):

        self.frequency = frequency
        self.drop = drop
        self.treshold_frequencies = None
        self.n_variables = None
        self.fftkwargs = fftkwargs

    def fit(self, X):

        X = to_2D(X)
        self.treshold_frequencies = []
        _, self.n_variables = X.shape
        for i in range(self.n_variables):
            x = X[:,i]
            fftx, freq = self.fftransform(x)
            frequency_tresholds = self.get_frequency_tresholds(fftx, freq)
            self.treshold_frequencies.append(frequency_tresholds)
        
        return self

    def transform(self, X):

        X = to_2D(X)
        n_steps, _ = X.shape
        X_transformed = []
        for i in range(self.n_variables):
            x = X[:,i]
            fftx, freq = self.fftransform(x)
            frequency_tresholds = self.treshold_frequencies[i]
            fftx_split = self.split_signal(fftx, freq, frequency_tresholds)
            for fftxi in fftx_split:
                x_i = self.reverse_fftransform(fftxi, n_steps)
                X_transformed.append(to_2D(x_i))
        X_transformed = np.hstack(X_transformed)
        return X_transformed

    def get_frequency_tresholds(self, fftx, freq):

        if not hasattr(self.frequency, '__iter__'):
            frequencies = [self.frequency]
        else:
            frequencies = self.frequency
        frequency_tresholds = np.zeros(len(frequencies)+2)-1.0
        frequency_tresholds[-1] = np.min([np.max(freq), self.drop])
        i = 1
        for frequency in frequencies:
            frequency_tresholds[i] = frequency
            i += 1
        frequency_tresholds = np.unique(frequency_tresholds)
        frequency_tresholds = np.sort(frequency_tresholds)
        return frequency_tresholds
    
    def split_signal(self, fftx, freq, tresholds):

        fftx = to_2D(fftx)
        fftx_split = []
        for i in range(1, len(tresholds)):
            treshold_min = tresholds[i-1]
            treshold_max = tresholds[i]
            fftxi = fftx.copy()
            fftxi[(freq<=treshold_min)|(freq>treshold_max),:] = 0.0
            fftx_split.append(fftxi)
        return fftx_split

    def fftransform(self, x):

        pass

    def reverse_fftransform(self, fftx):

        pass

class SFFTFrequencyExpansor(FFTFrequencyBaseExpansor):

    def fftransform(self, x):

        freq, t, fftx = stft(x, **self.fftkwargs)
        return fftx, freq

    def reverse_fftransform(self, fftx, n_steps):

        t, x = istft(fftx, **self.fftkwargs)
        x = x.ravel()
        if len(x) >= n_steps:
            x = x[:n_steps]
        return x

