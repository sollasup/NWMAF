import LoadRawData
import numpy as np
import Config
import matplotlib.pyplot as plt
import h5py
from sklearn.preprocessing import normalize
from sklearn import preprocessing
import Stepdetection
from scipy import stats
from scipy.signal import butter, lfilter, freqz

import pandas as pd
from sklearn.preprocessing import Imputer
from scipy.signal import medfilt



def Featurefiles(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    d = h5py.File("feature.hdf5", "a")
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
                        d["participant" + str(participant) + "/" + "session" + str(
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
                                imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                                imp.fit(steps)
                                steps = imp.transform(steps)
                                FeatureList=[]


                                for sensorposition in range(sensor,sensor+22):

                                    if feature==1:
                                        FeatureList.append(np.mean(steps[:, sensorposition]))

                                    if feature == 2:
                                        FeatureList.append(np.max(steps[:, sensorposition]))
                                    if feature == 3:
                                        FeatureList.append(np.min(steps[:, sensorposition]))
                                    if feature == 5:
                                        FeatureList.append(np.percentile(steps[:, sensorposition],95))
                                    if feature == 6:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 5))
                                    if feature == 7:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 20))
                                    if feature == 8:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 50))
                                    if feature == 4:
                                        FeatureList.append(np.sum(np.abs(steps[:, sensorposition])))
                                    if feature ==9:
                                        FeatureList.append(stats.skew(steps[:, sensorposition]))
                                    if feature ==10:
                                        FeatureList.append(np.argmax(steps[:,sensorposition]))
                                    if feature == 11:
                                        FeatureList.append(np.argmax(steps[:, sensorposition]))
                                    if feature == 12:
                                        FeatureList.append(np.mean(normalize(steps[:, sensorposition])))
                                    if feature == 13:
                                        FeatureList.append(np.mean(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 14:
                                        FeatureList.append(np.max(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 15:
                                        FeatureList.append(np.min(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 16:
                                        FeatureList.append(np.percentile(preprocessing.scale(steps[:, sensorposition]),5))
                                    if feature == 17:
                                        FeatureList.append(np.mean(medfilt(steps[:, sensorposition],3)))

                                    if feature == 18:
                                        FeatureList.append(np.max(medfilt(steps[:, sensorposition],3)))
                                    if feature == 19:
                                        FeatureList.append(np.min(medfilt(steps[:, sensorposition],3)))
                                    if feature == 20:
                                        FeatureList.append(np.argmax(medfilt(steps[:, sensorposition], 3)))
                                    if feature == 21:
                                        FeatureList.append(np.argmax(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 22:
                                        FeatureList.append(
                                            np.argmin(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 23:
                                        FeatureList.append(
                                            np.max(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 24:
                                        FeatureList.append(
                                            np.min(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 25:
                                        FeatureList.append(
                                            stats.skew(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 26:
                                        FeatureList.append(
                                            np.percentile(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6),5))
                                    if feature == 27:
                                        FeatureList.append(
                                            np.argmax(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 28:
                                        FeatureList.append(
                                            np.argmin(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 29:
                                        FeatureList.append(
                                            np.max(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 30:
                                        FeatureList.append(
                                            np.min(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 31:
                                        FeatureList.append(
                                            stats.skew(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 32:
                                        FeatureList.append(
                                            np.percentile(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6), 5))

                                Stepfeatures.append(FeatureList)
                            Stepfeatures= np.array(Stepfeatures)
                            d.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "sensor"+str(sensor)+ "/" + "feature"+str(feature)),
                                             data=Stepfeatures)
                    if sensor!=0:

                        featuresforsensors.append(d["participant" + str(participant) + "/" + "session" + str(session) + "/" + "sensor"+str(sensor)+"/feature"+str(feature)][:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]])
            print np.array(featuresforsensors).shape


            reshaped=np.hstack(featuresforsensors)

            featuresforsession.append(reshaped)

            checksum+=1
    return featuresforsession


def Featurefilescamera(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    c= h5py.File("featurecamera.hdf5", "a")
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
                        c["participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "sensor"+str(sensor)+ "/" + "featurecam"+str(feature)]
                    except KeyError:
                        e = False

                    if e == False:
                        contentlist = list(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstepcamera"])
                        dataset = []
                        for i in range(len(contentlist)):
                            dataset.append(f[("participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstepcamera/" + str(i))])
                        featureset=[]
                        if sensor !=0:
                            Stepfeatures=[]
                            for steps in dataset:
                                imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                                imp.fit(steps)
                                steps = imp.transform(steps)
                                FeatureList=[]

                                for sensorposition in range(sensor,sensor+22):

                                    if feature==1:
                                        FeatureList.append(np.mean(steps[:, sensorposition]))

                                    if feature == 2:
                                        FeatureList.append(np.max(steps[:, sensorposition]))
                                    if feature == 3:
                                        FeatureList.append(np.min(steps[:, sensorposition]))
                                    if feature == 5:
                                        FeatureList.append(np.percentile(steps[:, sensorposition],95))
                                    if feature == 6:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 5))
                                    if feature == 7:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 20))
                                    if feature == 8:
                                        FeatureList.append(np.percentile(steps[:, sensorposition], 50))
                                    if feature == 4:
                                        FeatureList.append(np.sum(np.abs(steps[:, sensorposition])))
                                    if feature ==9:
                                        FeatureList.append(stats.skew(steps[:, sensorposition]))
                                    if feature ==10:
                                        FeatureList.append(np.argmax(steps[:,sensorposition]))
                                    if feature == 11:
                                        FeatureList.append(np.argmax(steps[:, sensorposition]))
                                    if feature == 12:
                                        FeatureList.append(np.mean(normalize(steps[:, sensorposition])))
                                    if feature == 13:
                                        FeatureList.append(np.mean(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 14:
                                        FeatureList.append(np.max(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 15:
                                        FeatureList.append(np.min(preprocessing.scale(steps[:, sensorposition])))
                                    if feature == 16:
                                        FeatureList.append(np.percentile(preprocessing.scale(steps[:, sensorposition]),5))
                                    if feature == 17:
                                        FeatureList.append(np.mean(medfilt(steps[:, sensorposition], 3)))

                                    if feature == 18:
                                        FeatureList.append(np.max(medfilt(steps[:, sensorposition], 3)))
                                    if feature == 19:
                                        FeatureList.append(np.min(medfilt(steps[:, sensorposition], 3)))
                                    if feature == 20:
                                        FeatureList.append(np.argmax(medfilt(steps[:, sensorposition], 3)))
                                    if feature == 21:
                                        FeatureList.append(
                                            np.argmax(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 22:
                                        FeatureList.append(
                                            np.argmin(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 23:
                                        FeatureList.append(
                                            np.max(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 24:
                                        FeatureList.append(
                                            np.min(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 25:
                                        FeatureList.append(
                                            stats.skew(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6)))
                                    if feature == 26:
                                        FeatureList.append(
                                            np.percentile(butter_lowpass_filter(steps[:, sensorposition], 3, 50, 6), 5))
                                    if feature == 27:
                                        FeatureList.append(
                                            np.argmax(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 28:
                                        FeatureList.append(
                                            np.argmin(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 29:
                                        FeatureList.append(
                                            np.max(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 30:
                                        FeatureList.append(
                                            np.min(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 31:
                                        FeatureList.append(
                                            stats.skew(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6)))
                                    if feature == 32:
                                        FeatureList.append(
                                            np.percentile(butter_lowpass_filter(steps[:, sensorposition], 10, 50, 6),
                                                          5))

                                Stepfeatures.append(FeatureList)
                            Stepfeatures= np.array(Stepfeatures)
                            c.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "sensor"+str(sensor)+ "/" + "featurecam"+str(feature)),
                                             data=Stepfeatures)
                    if sensor!=0:

                        featuresforsensors.append(c["participant" + str(participant) + "/" + "session" + str(session) + "/" + "sensor"+str(sensor)+"/featurecam"+str(feature)][:,[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15]])
            print np.array(featuresforsensors).shape


            reshaped=np.hstack(featuresforsensors)

            featuresforsession.append(reshaped)

            checksum+=1
    return featuresforsession





def Stepfiles(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    RawSensorData, ElanGroundTruth = LoadRawData.loaddata(configuration)
    stepsensordata=[]
    stepsensordatacamera=[]
    steplabel=[]
    steplabelcamera=[]
    for i in range(len(RawSensorData)):
        matrix,label,passmatrix,passlabel =Stepdetection.elantosteps(RawSensorData[i].as_matrix()[:,:],np.array(ElanGroundTruth[i]))
        #calculate distance between steps to detect turns

        allcameraframes=[]
        cameralabel=[]
        cameraframe = []
        for e in range(0,len(matrix)-1):

            if (np.mean(matrix[e+1][:,0])-np.mean(matrix[e][:,0]))<150:
                cameraframe.append(matrix[e])
            else:
                cameraframe.append(matrix[e])
                cameralabel.append(label[e])
                allcameraframes.append(cameraframe)
                cameraframe=[]

        cameraframesstacked=[]
        for frames in allcameraframes:
            cameraframesstacked.append(np.vstack(frames))

        stepsensordatacamera.append(cameraframesstacked)
        steplabelcamera.append(cameralabel)
        stepsensordata.append(matrix)
        steplabel.append(label)

    dataposition = 0
    featuredata=[]
    featurelabel=[]
    featurelabelcamera=[]
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
                        print participant
                        print session
                        f.create_dataset(("participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"+"/" +str(i)),
                                             data=stepsensordata[dataposition][i])

                    for i in range(len(stepsensordatacamera[dataposition])):
                        print participant
                        print session
                        f.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                            session) + "/" + "rawstepcamera" + "/" + str(i)),
                                         data=stepsensordatacamera[dataposition][i])



                    f.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                        session) + "/" + "labelstep" ),
                                     data=np.array(steplabel[dataposition], dtype=np.float))
                    f.create_dataset(("participant" + str(participant) + "/" + "session" + str(
                        session) + "/" + "labelstepcamera"),
                                     data=np.array(steplabelcamera[dataposition], dtype=np.float))
                    dataposition += 1

            contentlist = list(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"])
            dataset = []
            for i in range(len(contentlist)):
                dataset.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/" +"rawstep"+"/" +str(i)])

            featuredata.append((dataset))
            featurelabel.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/"+"labelstep"])
            featurelabelcamera.append(f["participant" + str(participant) + "/" + "session" + str(session)+"/"+"labelstepcamera"])
    return featuredata,featurelabel,featurelabelcamera


def to_dataframe(featuredata,featurelabel,configuration,dictionary):
    dataframes=[]
    labelframes=[]
    for i in range(len(featuredata)):
        dataframes.append(pd.DataFrame(np.concatenate(featuredata[i])[:,2:]))
        labelframes.append(pd.DataFrame(featurelabel[i][:,:]))
    labelFrame=pd.concat(labelframes,axis=1)
    Frame=pd.concat(dataframes,axis=1)
    labelFrame=np.array(labelFrame)
    Frame = np.array(Frame)


    participantshead=[]
    sessionshead=[]
    sensorhead=[]
    print configuration.get("Sensors")
    for participant in configuration.get("Participants"):
        participantshead.append(dictionary[0].get(participant))
    for session in configuration.get("Sessions"):
        sessionshead.append(dictionary[1].get(session))

    sensorhead=["STE","LUA","LLA","LNS","RUA","RLA","RNS","CEN","LUL","LLL","LUF","RUL","RLL","RUF"]

    valueshead=["AccX","AccY","AccZ","GyrX","GyrY","GyrZ","MagX","MagY","MagZ","QuatA","QuatB","QuatC","QuatD","V1X","V1Y","V1Z","V2X","V2Y","V2Z","V3X","V3Y","V3Z"]
    labelshead=["Pass","Stickusage","Armswing","Step","Push","Grip","Pose","Feet","Timing","Foreswing","Eyeposition"]

    labelheader= pd.MultiIndex.from_product([participantshead,
                                         sessionshead,
                                         labelshead],
                                        names=["Participant", 'Session', "Label"])
    header = pd.MultiIndex.from_product([participantshead,
                                         sessionshead,
                                         sensorhead,
                                         valueshead],
                                        names=["Participant", 'Session', "Sensor", "Sensoraxis"])

    LabelData = pd.DataFrame(labelFrame,columns=labelheader)
    SensorData = pd.DataFrame(Frame,columns=header)

    return SensorData,LabelData


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y