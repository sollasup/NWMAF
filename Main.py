import LoadRawData
import Config
import FeatureConstruction
import h5py
import matplotlib.pyplot as plt
import Regression
import numpy as np

configuration,dictionaire = Config.loadconfig()

featuredata,featurelabel = FeatureConstruction.Stepfiles(configuration)
featurematrix = FeatureConstruction.Featurefiles(configuration)
Regression.linearRegression(featurematrix,featurelabel)
FeatureConstruction.to_dataframe(featuredata,configuration,dictionaire)
print featuredata[0]
print len(featuredata[0])
print len(featuredata[0][1,:])
plt.plot(featuredata[0][:,3])
plt.show()
























def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
if True:
    plt.subplot(311)
    plt.plot(featurematrix[0][:,3:9])
    plt.subplot(312)
    plt.plot(moving_average(featuredata[0][:,72]))
    plt.subplot(313)
    plt.plot(featurelabel[0][:, 2])
    plt.show()
correctlist =[]
print np.concatenate(featurematrix).shape
for matrix in featurematrix:

    print matrix.shape


for feature in featurelabel:
    print feature
f = h5py.File("mytestfile.hdf5", "a")

print f.keys()

print f.get("participant5").keys()
print f.get("participant5").get("session3").keys()
print f.get("participant5").get("session3").get("sensor134").keys()
#print f.get("participant2").get("session3").get("labelstep").get("0").keys()

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n