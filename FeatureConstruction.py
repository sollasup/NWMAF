import LoadRawData
import numpy as np
import Config
import h5py
import Stepdetection
import pandas as pd


def Featureconstruction(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    RawSensorData, ElanGroundTruth = LoadRawData.loaddata(configuration)
    for i in range(len(RawSensorData)):
        matrix,label,passmatrix,passlabel =Stepdetection.elantosteps(RawSensorData[i].as_matrix()[:,:],np.array(ElanGroundTruth[i]))
        feature, featurelabel = Stepdetection.getstepvalues(matrix.as_matrix(),label,[202,203,204])
        print feature
    dataposition = 0
    for participant in configuration.get("Participants"):
        for session in configuration.get("Sessions"):
            print participant
            print session

            f.create_dataset(("participant" + str(participant) + "/" + "session" + str(session)),
                             data=np.array(RawSensorData[dataposition]))
            dataposition += 1