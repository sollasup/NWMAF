from __future__ import division
__author__ = 'Sebastian'


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
from scipy.signal import medfilt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.feature_selection import SelectKBest

#calculate moving average
def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#Regression method
def linearRegression(DataList, Labellist,configuration):
    Dataparticipants=[]
    Labelparticipants=[]

    #Formatting the list to array to fit scikit format
    for i in range(0,len(configuration.get("Participants"))):
        ListtoStack=[]
        Labeltostack=[]
        for k in range(0,len(configuration.get("Sessions"))):
            Labeltostack.append(Labellist[i*len(configuration.get("Sessions"))+k])
            ListtoStack.append((DataList[i*len(configuration.get("Sessions"))+k]))
        Dataparticipants.append(np.vstack(ListtoStack))
        Labelparticipants.append(np.vstack(Labeltostack))
    print len(Dataparticipants[0][1,:])

    #Calculate relative sensor measurements,i.e Acc foot vs Acc stick
    for i in range(0, len(Dataparticipants)):
        alladdedcolumns=[]
        for t in [0,int(len(Dataparticipants[i][1,:])/4),int(len(Dataparticipants[i][1,:])/2),int(len(Dataparticipants[i][1,:])/4*3)]:
            for p in [0,int(len(Dataparticipants[i][1,:])/4),int(len(Dataparticipants[i][1,:])/2),int(len(Dataparticipants[i][1,:])/4*3)]:
                for z in range(0,int(len(Dataparticipants[i][1,:])/4)):
                    if p!=t:
                        if t<p:
                            columns=(Dataparticipants[i][:,t+z]-Dataparticipants[i][:,p+z])
                            alladdedcolumns.append(columns)
        Dataparticipants[i] = np.append(Dataparticipants[i], np.transpose(alladdedcolumns), axis=1)


    for i in range(0, len(Dataparticipants)):
        for t in range(0,len(Dataparticipants[i][1,:])):
            size=11
            #Smooth data using the average
            if True:
                average = (moving_average(Dataparticipants[i][:, t], size))
                Dataparticipants[i][:len(average),t]=average
                Dataparticipants[i][len(average):, t]= average[-(size-1):]

            #Smooth data using the median
            if False:
                Dataparticipants[i][:,t] = (medfilt(Dataparticipants[i][:, t], size))

        #Impute missing values and scale the data
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(Dataparticipants[i])
        Dataparticipants[i]= imp.transform(Dataparticipants[i])

        std_scale =preprocessing.StandardScaler().fit(Dataparticipants[i])

        Dataparticipants[i]=std_scale.transform(Dataparticipants[i])


    print range(0, len(Dataparticipants) - 1)
    components=5
    print configuration
    print "PCA components=  "+ str(components)

    #Iterate iver all labels
    for labels in range(0,3):
        RMSEvalues = []
        #Iterate over all Participants
        for i in range(0, len(Dataparticipants)):
            Indices = range(0,len(Dataparticipants))
            Indices.remove(i)

            X_train= np.vstack([Dataparticipants[p] for p in Indices])
            y_train = np.vstack([Labelparticipants[p] for p in Indices])[:,labels]
            X_test = Dataparticipants[i]
            y_test = Labelparticipants[i][:, labels]

            #Use gradient Decent to select features
            if True:
                c=(boosting(X_train,y_train,X_test,y_test,configuration))
                X_train=X_train[:,c]
                X_test=X_test[:,c]


            y_calculated=[]

            #Use PCA for feature construction
            if True:  # PCA
                pca = PCA(n_components=components).fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
                regr = regressor()
                regr.fit(X_train, y_train)
                #X_train, y_train = downsamplingthrougherror(X_train, y_train, regr.predict(X_train))
                #X_train,y_train=getextremes(X_train, y_train, regr.predict(X_train))
                X_train,y_train=instancecreator(X_train,y_train)

            if True:
                regr = regressor()
                regr.fit(X_train, y_train)
                y_calculated.append(regr.predict(X_test))

            if False:  # outliers
                X_train2, y_train2 = outlierdetection(X_train, y_train, 1)
                regr = regressor()
                regr.fit(X_train2, y_train2)
                y_calculated.append(regr.predict(X_test))
            if False: #outliers
                X_train2,y_train2=outlierdetection(X_train,y_train,2)
                regr = regressor()
                regr.fit(X_train2, y_train2)
                y_calculated.append(regr.predict(X_test))
            if False:  # outliers
                X_train2, y_train2 = outlierdetection(X_train, y_train, 3)
                regr = regressor()
                regr.fit(X_train2, y_train2)
                y_calculated.append(regr.predict(X_test))
            if False:  # outliers
                X_train3, y_train3 = outlierdetection(X_train, y_train, 0)
                regr = regressor()
                regr.fit(X_train3, y_train3)
                y_calculated.append(regr.predict(X_test))





            if False:

                plotscatter(pd.DataFrame(X_train),y_train)
            RMSEvalues.append(np.sqrt(metrics.mean_squared_error(y_test,medfilt(np.array(y_calculated).mean(axis=0),3))))



            if True:
                plt.plot(y_test,"+",label="Ground truth")

                #plt.plot(regr.predict((X_test)),label="Regression Prediction")

                plt.plot(medfilt(np.array(y_calculated).mean(axis=0),1))
                plt.legend()
                plt.savefig("points"+"Mistake_"+str(labels)+"part_"+str(i)+".png")
                plt.close()
                #plt.show()
        print np.mean(np.array(RMSEvalues))




    # Train the model using the training sets


def plotscatter(Dataframe,label):
        Dataframe["label"]=label
        sns.pairplot(Dataframe,hue="label",diag_kind="kde")

        plt.show()


#Remove outliers from the data set
def outlierdetection(X,y,skip):
    data1= np.array(X[np.where(y==1),:])[0]
    data2 = np.array(X[np.where(y == 2), :])[0]
    data3 = np.array(X[np.where(y == 3), :])[0]
    label1=np.array(y[np.where(y==1)])
    label2=np.array(y[np.where(y == 2)])
    label3=np.array(y[np.where(y == 3)])
    if label1.size:
        data1,label1=returnmedian(data1,label1)
    if label2.size:
        data2, label2 = returnmedian(data2, label2)
    if label3.size:
        data3, label3 = returnmedian(data3, label3)
    if False:
        plt.subplot(411)
        plt.plot(data1)
        plt.subplot(412)
        plt.plot(data2)
        plt.subplot(413)
        plt.plot(data3)
        plt.subplot(414)
        plt.plot(X)
        plt.show()

    if skip==1:
        return np.concatenate([data2,data3]),np.concatenate([label2,label3])
    if skip == 2:
        return np.concatenate([data1,  data3]), np.concatenate([label1,  label3])
    if skip == 3:
        return np.concatenate([data1, data2]), np.concatenate([label1, label2])
    return np.concatenate([data1,data2,data3]),np.concatenate([label1,label2,label3])


#return the data surrounding the median
def returnmedian(data,label):
    outlierfree=[]
    for i in range(len(data[1,:])):
        median= np.median(data[:,i])
        var = np.std(data[:,i])
        outlierfree.append((data[:,i] >= median-var) & (data[:,i] <= median + var))
    for i in range(len(outlierfree)-1):
        outlierfree[0]=[a and b for a, b in zip(outlierfree[0], outlierfree[i+1])]
    c= np.where(outlierfree[0])[0]
    datareduced =data[c,:]
    if False:
        plt.subplot(211)
        plt.plot(datareduced)
        plt.subplot(212)
        plt.axhline(median-var)
        plt.plot(data)
        plt.show()
    return datareduced, label[c]

#Gradient Decent Regressor which is used for feature extraction
def boosting(X_train,y_train,X_test,y_test,configuration):
    print len(X_train)
    print len(X_train[1,:])
    featlist=[]
    for sensor in configuration.get("Sensors"):
        for feature in configuration.get("Features"):
            for axis in ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ"]:#,"MagX","MagY","MagZ","QuatA","QuatB","QuatC","QuatD","V1X","V1Y","V1Z","V2X","V2Y","V2Z","V3X","V3Y","V3Z"]:
                featlist.append(str(sensor)+str(feature)+axis)

    params = {'n_estimators': 20, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}


    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    if True:

        print("MSE: %.4f" % mse)
        # compute test set deviance
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

        for i, y_pred in enumerate(clf.staged_predict(X_test)):
            test_score[i] = clf.loss_(y_test, y_pred)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                 label='Training Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                 label='Test Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')

        feature_importance = clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        print feature_importance
        print np.where(feature_importance>4)[0]

    return np.where(feature_importance>4)[0]


def regressor():
    #return SVR()
    return RandomForestRegressor()
    #return linear_model.LinearRegression()
    #return linear_model.LinearRegression()
    #return linear_model.ElasticNet(alpha = 0.1, l1_ratio = 0.7)


#Rescale the data for more than 3 labels
def getextremes(X,y,prediction):


    data1 = np.array(X[np.where(y == 1), :])[0]
    data2 = np.array(X[np.where(y == 2), :])[0]
    data3 = np.array(X[np.where(y == 3), :])[0]
    label1 = np.array(y[np.where(y == 1)])
    label2 = np.array(y[np.where(y == 2)])
    label3 = np.array(y[np.where(y == 3)])
    pred1 = np.array(prediction[np.where(y == 1)])
    pred2 = np.array(prediction[np.where(y == 2)])
    pred3 = np.array(prediction[np.where(y == 3)])

    median = np.median(pred1)
    std =np.std(pred1)
    belowscale=(pred1 >= median +std)
    c = np.where(belowscale)[0]
    label1[c]=1.9
    belowscale = (pred1 <= median - std)
    c = np.where(belowscale)[0]
    label1[c] = 1.8

    median = np.median(pred3)
    std=np.std(pred3)
    belowscale = (pred3 <= median-std)
    c = np.where(belowscale)[0]
    label3[c] = 2.1
    belowscale = (pred3 >= median + std)
    c = np.where(belowscale)[0]
    label3[c] = 2.2

    if False:
        plt.subplot(311)
        plt.plot(y)
        plt.plot(prediction)
        plt.subplot(312)
        plt.plot(label1)
        plt.plot(pred1)
        plt.subplot(313)
        plt.plot(label2)
        plt.plot(pred2)
        plt.show()

    return np.concatenate([data1, data2, data3]), np.concatenate([label1, label2, label3])

#The prediciton of the Train data set is used to rescale the data
def downsamplingthrougherror(X,y,prediction):
    data1 = np.array(X[np.where(y == 1), :])[0]
    data2 = np.array(X[np.where(y == 2), :])[0]
    data3 = np.array(X[np.where(y == 3), :])[0]
    label1 = np.array(y[np.where(y == 1)])
    label2 = np.array(y[np.where(y == 2)])
    label3 = np.array(y[np.where(y == 3)])
    pred1 = np.array(prediction[np.where(y == 1)])
    pred2 = np.array(prediction[np.where(y == 2)])
    pred3 = np.array(prediction[np.where(y == 3)])

    median = np.median(pred1)

    belowscale = (pred1 <= median)
    c = np.where(belowscale)[0]
    label1[c] = 1.2
    label1= label1[c]
    data1 = data1[c, :]

    median = np.median(pred3)

    belowscale = (pred3 >= median)
    c = np.where(belowscale)[0]
    label3[c] = 2.8
    label3 = label3[c]
    data3=data3[c,:]

    if False:
        plt.subplot(311)
        plt.plot(y)
        plt.plot(prediction)
        plt.subplot(312)
        plt.plot(label1)
        plt.plot(pred1)
        plt.subplot(313)
        plt.plot(label3)
        plt.plot(data3)
        plt.show()

    return np.concatenate([data1, data2, data3]), np.concatenate([label1, label2, label3])

#Create new bad instances
def instancecreator(X,y):
    data1 = np.array(X[np.where(y == 1), :])[0]
    data2 = np.array(X[np.where(y == 2), :])[0]
    data3 = np.array(X[np.where(y == 3), :])[0]
    label1 = np.array(y[np.where(y == 1)])
    label2 = np.array(y[np.where(y == 2)])
    label3 = np.array(y[np.where(y == 3)])

    newdata=[]
    if len(data3)!=0:
        for i in range(len(data3)):
            newdata.append(data3[i,:]*10)
            newdata.append(data3[i,:]*-10)
            label3=np.append(label3,3)
            label3 = np.append(label3, 3)

        data3= np.append(data3, np.array(newdata), axis=0)



    return np.concatenate([data1, data2, data3]), np.concatenate([label1, label2, label3])

