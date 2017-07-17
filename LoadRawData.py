import pandas as pd
import FeatureConstruction
import h5py

def loaddata(configuration):
    f = h5py.File("mytestfile.hdf5", "a")
    RawSensorData=[]
    ElanGroundTruth=[]
    for participant in configuration.get("Participants"):
        for session in configuration.get("Sessions"):
            print participant
            print session
            e=True
            try:
                f["participant" + str(participant) + "/" + "session" + str(
                    session) + "/" + "rawstep"]
            except KeyError:
                e = False
            if e==False:
                RawSensorData.append(pd.read_csv(r'C:\Users\Sebastian\Documents\Recordings/'+"ID00"+str(participant)
                            +"/"+str(session)+"/"+"mergedID00"+str(participant)+"cut.csv",sep="\t"))
                elan=pd.read_csv(r'C:\Users\Sebastian\Documents\Recordings/' + "ID00" + str(participant)
                            + "/" + str(session) + "/" + "elan" + ".csv", sep = "\t",error_bad_lines=False)
                elan = pd.DataFrame(elan.as_matrix())
                elan = elancreate(elan)
                ElanGroundTruth.append(elan)


    return RawSensorData,ElanGroundTruth

def elancreate(elan):

    #elan = pd.concat([elan.iloc[:,[0,2,3,4,5]]])
    elan = pd.DataFrame(elan.iloc[:,[0,2,3,4,5]])
    elan = elan.as_matrix()


    return  elan