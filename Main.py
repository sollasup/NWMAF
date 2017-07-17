import LoadRawData
import Config
import FeatureConstruction
import h5py

configuration = Config.loadconfig()
featuredata,featurelabel = FeatureConstruction.Stepfiles(configuration)
featurematrix = FeatureConstruction.Featurefiles(configuration)
correctlist =[]
for matrix in featurematrix:

    print matrix.shape


for feature in featurelabel:
    print feature
f = h5py.File("mytestfile.hdf5", "a")

print f.keys()

print f.get("participant5").keys()
print f.get("participant5").get("session3").get("labelstep")
#print f.get("participant2").get("session3").get("labelstep").get("0").keys()