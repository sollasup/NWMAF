import LoadRawData
import Config
import FeatureConstruction
import h5py
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import Regression
import math
from scipy.signal import butter, lfilter, freqz
from scipy import signal

from sklearn import preprocessing
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
#TODO RELATIVER RMSE
#TODO Instanzen erweitern
#TODO Erweitern 3+ 1-
#TODO FUSION REGREssors
#TODO METRIKEN wie bewerte ich am besten



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
m = cm.ScalarMappable(norm=norm, cmap=cmap)

configuration,dictionaire = Config.loadconfig()

featuredata,featurelabel,featurelabelcamera = FeatureConstruction.Stepfiles(configuration)
if False:

    for n in range(0,2):
        plt.plot(featurelabel[n][:,1])
    plt.show()








#Data,Label=FeatureConstruction.to_dataframe(featuredata,featurelabel,configuration,dictionaire)

featurematrix = FeatureConstruction.Featurefiles(configuration)
featurematrixcam = FeatureConstruction.Featurefilescamera(configuration)

Regression.linearRegression(featurematrix,featurelabel,configuration)



























def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
if False:
    plt.subplot(311)
    plt.plot(featurematrix[0][:,3:9])
    plt.subplot(312)
    plt.plot(moving_average(featuredata[0][:,72]))
    plt.subplot(313)
    plt.plot(featurelabel[0][:, 2])
    plt.show()
def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n