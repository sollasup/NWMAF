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
            print "----"
            print "Participant"+str(participant)
            print "Session"+str(session)

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
                                for sensorposition in range(sensor,sensor+15):

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
            print np.array(featuresforsensors).shape
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
                dataset.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"+"/" +str(i)])
            featuredata.append(np.concatenate(dataset))
            featurelabel.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/"+"labelstep"])

    return featuredata,featurelabel


def to_dataframe(featuredata,configuration,dictionary):
    dataframes=[]
    for i in range(len(featuredata)):
        dataframes.append(pd.DataFrame(featuredata[i][:,2:]))

    Frame=pd.concat(dataframes,axis=1)
    print (Frame)
    Frame = np.array(Frame)
    print Frame


    participantshead=[]
    sessionshead=[]
    sensorhead=[]
    for participant in configuration.get("Participants"):
        participantshead.append(dictionary[0].get(participant))
    for session in configuration.get("Sessions"):
        sessionshead.append(dictionary[1].get(session))
    for sensor in configuration.get("Sensors"):
        sensorhead.append(dictionary[2].get(sensor))

    valueshead=["AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ","QuatA","QuatB","QuatC","QuatD","V1X","V1Y","V1Z","V2X","V2Y","V2Z","V3X","V3Y","V3Z"]



    header = pd.MultiIndex.from_product([participantshead,
                                         sessionshead,
                                         sensorhead,
                                         valueshead],
                                        names=["Participant", 'Session', "Sensor", "Sensoraxis"])
    print len(header.labels)
    print len(header.labels[0])
    print len(header.labels[1])
    print len(header.labels[2])
    print len(header.labels[3])

    SensorData = pd.DataFrame(Frame,columns=header)
    print SensorData