import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FourierTransformation
from scipy.signal import argrelmax
from scipy import signal

def elantosteps(dataMatrix,elan):
    matrix, step = videosteps(dataMatrix[:,:],elan)

    elan = pd.DataFrame(elan[:,1:],index=elan[:,0])
    elanpass = np.array(elan.ix['1Passgang'])[:,3]
    elanstock = np.array(elan.ix['1Stockeinsatz'])[:,3]
    elanarm = np.array(elan.ix['1Armeinsatz'])[:,3]
    elanschritt = np.array(elan.ix['2Schrittlaenge'])[:,3]
    elanschub = np.array(elan.ix['2Schub'])[:,3]
    elanverkrampfung = np.array(elan.ix['2Verkrampfung'])[:,3]
    elanober = np.array(elan.ix['2Oberkoerper'])[:,3]
    elanfuss = np.array(elan.ix['2Fussaufsatz'])[:,3]
    elantiming = np.array(elan.ix['2Timing'])[:,3]
    elanshwingen = np.array(elan.ix['3Vorschwingen'])[:,3]
    elanblick = np.array(elan.ix['3Blick'])[:,3]
    new = np.c_[step,elanpass]
    new = np.c_[new,elanstock]
    new = np.c_[new,elanarm]
    new = np.c_[new,elanschritt]
    new = np.c_[new,elanschub]
    new = np.c_[new,elanfuss]
    new = np.c_[new,elanverkrampfung]
    new = np.c_[new,elanober]
    new = np.c_[new,elantiming]
    new = np.c_[new,elanshwingen]
    new = np.c_[new,elanblick]


    for k in range(4,len(new[1,:])):
        for t in range(0,len(new)):
            if new[t,k]==3 or new[t,k]==2 or new[t,k]==1:
                temp = new[t,k]
            else:
                new[t,k]= temp


    for i in range(0,len(new)):
        temp = new[i,4]
        if temp==3:

            new[i,4:]=3
    labeldata = []
    for i in range(0,len(new)):
            if new[i,4]==3 or new[i,4]==2 or new[i,4]==1:
                print new[i,4]
                labeldata.append(new[i,:])
    labeldata = np.array(labeldata)
    print labeldata
    matrixsteps = pd.DataFrame(matrix[labeldata[0,0]:labeldata[0,1],:])
    matrix = pd.DataFrame(matrix)
    for i in range(1,len(labeldata)):
        matrixsteps = pd.concat([matrixsteps,matrix.iloc[labeldata[i,0]:labeldata[i,1],:]])

    label = []
    for i in range(0,len(labeldata)):
        for k in range(labeldata[i,0],labeldata[i,1]):
            labels = np.array(labeldata[i,4:])
            #labels = map(int,labels)
            label.append(labels)





    matrixstepspass = np.array(matrixsteps)
    labelpass = np.array(label)


    label =[]
    for i in range(0,len(labeldata)):
        temp = labeldata[i,4]
        if temp != 3:
            label.append(labeldata[i,:])

    label = np.array(label)
    print label
    matrix=np.array(matrix)
    matrixsteps = pd.DataFrame(matrix[label[0,0]:label[0,1],:])
    matrix = pd.DataFrame(matrix)
    for i in range(1,len(label)):
        matrixsteps = pd.concat([matrixsteps,matrix.iloc[label[i,0]:label[i,1],:]])
    label = label[:,:]


    labelnew = []
    for i in range(0,len(label)):
        for k in range(label[i,0],label[i,1]):
            labels = np.array(label[i,4:])
            #labels = map(int,labels)
            labelnew.append(labels)

    return matrixsteps,labelnew,matrixstepspass,labelpass[:,0]


def videosteps(dataMatrix, elan):
    newMatrix= dataMatrix
    newlan =elan
    elan = pd.DataFrame(elan[:,1:],index=elan[:,0])
    elan = np.array(elan.ix['1Passgang'])

    elan[:,:2] = elan[:,:2]/10
    elanhalbe = elan[:,:2]/2
    stepright, noneed = stepDetectionright(newMatrix)
    stepleft, noneed = stepDetectionleft(newMatrix)
    steps = np.concatenate((stepright,stepleft))
    realsteps = []
    for i in range(0,len(elan)):
        nearest =  find_nearest(steps[:,0],elanhalbe[i,0])
        nearest2 = steps[nearest,0]
        nearest = steps[nearest,1]
        realsteps.append([int(nearest2),int(nearest),int(elan[i,0])*20,int(elan[i,1])*20])
    realsteps = np.array(realsteps)

    plt.plot(newMatrix[:,202],label = "Beschleunigung senkrecht")

    plt.axvline(3,color ='r',label="Bergsteigeralgorithmus Schritt")
    plt.axvline(3,color ='g',label="Video Schritt")
    plt.legend(fontsize = 18)
    plt.title("Schritterkennung",fontsize=18)
    plt.xlabel("Datenpunkte [s/50]",fontsize = 18)
    plt.ylabel("Beschleunigung [m/s^2]", fontsize = 18)
    plt.xticks(fontsize  =16)
    plt.yticks(fontsize = 16)
    distance=[]
    distance2=[]

    for  xs in elanhalbe[:,0]:
        plt.axvline(x=xs,color = 'r')
    for  xs in elanhalbe[:,1]:
        plt.axvline(x=xs,color = 'r')
    for  xs in realsteps[:,0]:
        plt.axvline(x=xs,color = 'g')
    for  xs in realsteps[:,1]:
        plt.axvline(x=xs,color = 'g')
    for  xs in stepleft[:,0]:
       plt.axvline(x=xs,color = 'g')
    for  xs in stepleft[:,1]:
       plt.axvline(x=xs,color = 'g')
    plt.show()
    for k in range(len(elanhalbe[:, 0])):
        distance.append(elanhalbe[k, 0] - realsteps[k, 0])
        distance2.append(elanhalbe[k, 1] - realsteps[k, 1])


    print np.array(distance).var()
    print np.array(distance).mean()
    print np.array(distance2).var()
    print np.array(distance2).mean()
    plt.hist(distance)
    plt.show()
    return newMatrix, realsteps



def stepDetectionright(dataMatrix):
    minimas = getmaximas(dataMatrix,Sensor=[268])
    maxima = minimas[0]

    newmatrix = dataMatrix[:]

    Steps = []
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    Steps.append([0,maxima[0]])

    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]
        Steps.append([maxima[j],maxima[j+1]])

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    Steps.append([maxima[len(maxima)-1],len(dataMatrix[:,0])])
    extractedstep = np.array(Steps)
    return extractedstep, np.c_[dataMatrix,  newmatrix[:,0]]


def stepDetectionleft(dataMatrix):
    minimas = getmaximas(dataMatrix,Sensor=[202])
    maxima = minimas[0]

    newmatrix = dataMatrix
    Steps = []
    for i in range(0,maxima[0]):
        newmatrix[i,0]= maxima[0]
    Steps.append([0,maxima[0]])

    for j in range(0,len(maxima)-1):

        for k in range(maxima[j],maxima[j+1]):
            newmatrix[k,0]= maxima[j]
        Steps.append([maxima[j],maxima[j+1]])

    for z in range(maxima[len(maxima)-1],len(dataMatrix[:,0])):
        newmatrix[z,0] = maxima[len(maxima)-1]
    Steps.append([maxima[len(maxima)-1],len(dataMatrix[:,0])])
    extractedstep = np.array(Steps)
    return extractedstep, np.c_[dataMatrix,  newmatrix[:,0]]


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def getmaximas(dataMatrix, Sensor=[290]):
    signal = dataMatrix[:,Sensor[0]]
    maxAbsValue, maxAbsFreq = FourierTransformation.maxAbsFreq(signal[:])
    Filtered = filter(dataMatrix,Sensor,maxAbsFreq)

    return argrelmax(Filtered[:,Sensor],order=25)


def filter(dataMatrix,Sensors, highcut=0, lowcut= 0, Ordnung =2, filtertype = "lowpass"):
    fs = 50
    nyquist = 0.5*50
    cutoffhigh= highcut/nyquist
    cutofflow = lowcut/nyquist
    if filtertype == "lowpass":
        bandbreite = cutoffhigh
    elif filtertype == "highpass":
        bandbreite = cutofflow
    elif filtertype == "bandpass":
        bandbreite = [cutofflow,cutoffhigh]

    y= np.array(dataMatrix)
    k = np.array(y, dtype=float)
    y2 =np.array(y)
    b, a = signal.butter(Ordnung, bandbreite, btype = filtertype)

    for Sensor in Sensors:
        y2[:,Sensor] = signal.lfilter(b, a, k[:,Sensor])  # standard filter

    y2 = np.array(y2)
    return np.c_[dataMatrix[:, :2],  y2[:,2:]]



def getstepvalues(matrix,label,Sensors):

    Sensorchooice = Sensors
    testmatrix = matrix
    testlabel = label
    a,b = stepDetectionright(testmatrix)

    testmatrix= (pd.DataFrame(testmatrix))
    zeros = np.zeros(len(testmatrix))
    for xs in a[:, 0]:
        zeros[xs]=1
        plt.axvline(x=xs, color='r')
    for xs in a[:, 1]:
        plt.axvline(x=xs, color='r')
    for xs in a[:, 0]:
        plt.axvline(x=xs, color='g')
    for xs in a[:, 1]:
        plt.axvline(x=xs, color='g')
    plt.plot(testmatrix.iloc[:,202])
    plt.show()
    matrix=testmatrix.iloc[:,[68,70,69,134,136,135]]

    new = pd.concat([matrix,pd.DataFrame(zeros)],axis=1)
    plt.plot(new)
    plt.show()
    labels =[]
    for t in a:
        labels1 =[]
        for k in range(len(testlabel[1,:])):
            x = np.array(testlabel[t[0]:t[1],k])
            x= x.astype(int)

            counts = np.bincount(x)
            labels1.append(np.argmax(counts))
        labels.append(labels1)
    labels = pd.DataFrame(labels)
    labels = labels.as_matrix()
    datas =[]

    for t in a:
        columns =[]
        for i in Sensorchooice:
            columns.append(np.mean(np.array(testmatrix[t[0]:t[1],i])))

        datas.append(columns)

    datas = pd.DataFrame(datas)
    datas = datas.as_matrix()

    return datas,labels