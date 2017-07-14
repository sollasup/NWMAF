import pandas as pd


def loaddata(configuration):
    RawSensorData=[]
    ElanGroundTruth=[]
    for participant in configuration.get("Participants"):
        for session in configuration.get("Sessions"):
            print participant
            print session
            RawSensorData.append(pd.read_csv(r'C:\Users\Sebastian\Documents\Recordings/'+"ID00"+str(participant)
                        +"/"+str(session)+"/"+"mergedID00"+str(participant)+"cut.csv",sep="\t"))
            ElanGroundTruth.append(pd.read_csv(r'C:\Users\Sebastian\Documents\Recordings/' + "ID00" + str(participant)
                        + "/" + str(session) + "/" + "elan" + ".csv", sep = "\t",error_bad_lines=False))


