import LoadRawData
import Config
import FeatureConstruction
import h5py
import matplotlib.pyplot as plt
import Regression
import math
from sklearn import preprocessing
import numpy as np

configuration,dictionaire = Config.loadconfig()

featuredata,featurelabel,featurelabelcamera = FeatureConstruction.Stepfiles(configuration)
featurematrix = FeatureConstruction.Featurefiles(configuration)
featurematrixcam = FeatureConstruction.Featurefilescamera(configuration)

Regression.linearRegression(featurematrix,featurelabel,configuration)


Data,Label=FeatureConstruction.to_dataframe(featuredata,featurelabel,configuration,dictionaire)

x=Data["Participant1"]["Session1"]["RUF"]["AccX"]
accdata = [value for value in x if not math.isnan(value)]
minmax_scale = preprocessing.MinMaxScaler().fit(accdata)
plt.subplot(211)
plt.plot(minmax_scale.transform(accdata))
plt.subplot(212)
plt.plot(accdata)
plt.show()
























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