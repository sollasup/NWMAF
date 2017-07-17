import LoadRawData
import numpy as np
import Config
import matplotlib.pyplot as plt
import h5py
import Stepdetection
import pandas as pd



def Featurefiles(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    featuresforsession = []
    checksum=0
    for participant in configuration.get("Participants"):
        for session in configuration.get("Sessions"):
            featuresforsensors = []
            for sensor in configuration.get("Sensors"):
                for feature in configuration.get("Features"):
                    e = True
                    try:
                        f["participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "sensor"+str(sensor)+ "/" + "feature"+str(feature)]
                    except KeyError:
                        e = False

                    if e == False:
                        contentlist = list(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"])
                        dataset = []
                        for i in range(len(contentlist)):
                            dataset.append(f[("participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep/" + str(i))])
                        featureset=[]
                        if sensor !=0:
                            Stepfeatures=[]
                            for steps in dataset:
                                FeatureList=[]
                                for sensorposition in range(sensor,sensor+24):
                                    if feature==1:
                                        FeatureList.append(np.mean(steps[:, sensorposition]))

                                    if feature == 2:
                                        FeatureList.append(np.max(steps[:, sensorposition]))
                                    if feature == 3:
                                        FeatureList.append(np.min(steps[:, sensorposition]))
                                Stepfeatures.append(FeatureList)
                            Stepfeatures= np.array(Stepfeatures)
                            f.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "sensor"+str(sensor)+ "/" + "feature"+str(feature)),
                                             data=Stepfeatures)
                    if sensor!=0:

                        featuresforsensors.append(f["participant" + str(participant) + "/" + "session" + str(session) + "/" + "sensor"+str(sensor)+"/feature"+str(feature)])

            featuresforsession.append(np.array(featuresforsensors).reshape((np.array(featuresforsensors).shape[1], -1)))

            print checksum
            checksum+=1
    return featuresforsession








def Stepfiles(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    RawSensorData, ElanGroundTruth = LoadRawData.loaddata(configuration)
    stepsensordata=[]
    steplabel=[]
    for i in range(len(RawSensorData)):
        matrix,label,passmatrix,passlabel =Stepdetection.elantosteps(RawSensorData[i].as_matrix()[:,:],np.array(ElanGroundTruth[i]))
        stepsensordata.append(matrix)
        steplabel.append(label)

    dataposition = 0
    featuredata=[]
    featurelabel=[]
    #create hdf5 file entry for created step data sets
    for participant in configuration.get("Participants"):
        for session in configuration.get("Sessions"):

            if len(stepsensordata)>0:


                e=True
                try:
                    f["participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "rawstep"]
                except KeyError:
                    e = False
                if e==False:
                    for i in range(len(stepsensordata[dataposition])):

                        f.create_dataset(("participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"+"/" +str(i)),
                                             data=stepsensordata[dataposition][i])



                    f.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                        session) + "/" + "labelstep" ),
                                     data=np.array(steplabel[dataposition], dtype=np.float))
                    dataposition += 1

            contentlist = list(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"])
            dataset = []
            for i in range(len(contentlist)):
                featuredata.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"+"/" +str(i)])
            featurelabel.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/"+"labelstep"])

    return featuredata,featurelabel



