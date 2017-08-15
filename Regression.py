from __future__ import division
__author__ = 'Sebastian'

from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import argrelmin,argrelmax
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from pymc3 import Model, Normal, HalfNormal
import pymc3 as pm
from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from pandas.tools.plotting import  radviz,andrews_curves
from sklearn import neighbors
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import time
from scipy.signal import medfilt
from sklearn.linear_model import Lasso
from sklearn import svm,datasets
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import numpy as np
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import math
import seaborn as sns
from sklearn.preprocessing import Imputer
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def linearRegression(DataList, Labellist,configuration):
    Dataparticipants=[]
    Labelparticipants=[]
    for i in range(0,len(configuration.get("Participants"))):
        ListtoStack=[]
        Labeltostack=[]
        for k in range(0,len(configuration.get("Sessions"))):
            Labeltostack.append(Labellist[i*len(configuration.get("Sessions"))+k])
            ListtoStack.append((DataList[i*len(configuration.get("Sessions"))+k]))
        Dataparticipants.append(np.vstack(ListtoStack))
        Labelparticipants.append(np.vstack(Labeltostack))
    print len(Dataparticipants[0][1,:])


    for i in range(0, len(Dataparticipants)):
        for t in range(0,len(Dataparticipants[i][1,:])):
            size=5

            if True:
                average = (moving_average(Dataparticipants[i][:, t], size))
                Dataparticipants[i][:len(average),t]=average
                Dataparticipants[i][len(average):, t]= average[-(size-1):]

            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imp.fit(Dataparticipants[i])
            Dataparticipants[i]= imp.transform(Dataparticipants[i])

            std_scale =preprocessing.StandardScaler().fit(Dataparticipants[i])

            Dataparticipants[i]=std_scale.transform(Dataparticipants[i])



    print range(0, len(Dataparticipants) - 1)
    components=5
    print configuration
    print "PCA components=  "+ str(components)


    for labels in range(0,11):
        RMSEvalues = []
        for i in range(0, len(Dataparticipants)):
            Indices = range(0,len(Dataparticipants))
            Indices.remove(i)

            X_train= np.vstack([Dataparticipants[p] for p in Indices])
            y_train = np.vstack([Labelparticipants[p] for p in Indices])[:,labels]
            X_test = Dataparticipants[i]
            y_test = Labelparticipants[i][:, labels]
            if False:
                c=(boosting(X_train,y_train,X_test,y_test,configuration))

            y_calculated=[]
            if False: #
                Kbest = SelectKBest(chi2, k=30).fit(np.abs(X_train), y_train)
                X_train = Kbest.transform(X_train)
                X_test = Kbest.transform(X_test)
            if True:  # PCA
                pca = PCA(n_components=components).fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
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

                plotscatter(pd.DataFrame(X_train3),y_train3)
                plotscatter(pd.DataFrame(X_test),y_test)
            #print "part: " +str(mean_squared_error(y_test,np.array(y_calculated).mean(axis=0)))
            RMSEvalues.append(mean_squared_error(y_test,medfilt(np.array(y_calculated).mean(axis=0),5)))



            if True:
                plt.plot(y_test,label="Ground truth")

                #plt.plot(regr.predict((X_test)),label="Regression Prediction")

                plt.plot(medfilt(np.array(y_calculated).mean(axis=0),5))
                plt.legend()
                plt.savefig("allkategorietlier"+str(i)+".png")
                plt.close()
                #plt.show()
        print np.mean(np.array(RMSEvalues))




    # Train the model using the training sets


def plotscatter(Dataframe,label):
        Dataframe["label"]=label
        sns.pairplot(Dataframe,hue="label",diag_kind="kde")

        plt.show()

def reject_outliers(data,m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def outlierdetection(X,y,skip):
    data1= np.array(X[np.where(y==1),:])[0]
    data2 = np.array(X[np.where(y == 2), :])[0]
    data3 = np.array(X[np.where(y == 3), :])[0]
    label1=np.array(y[np.where(y==1)])
    label2=np.array(y[np.where(y == 2)])
    label3=np.array(y[np.where(y == 3)])
    data1,label1=returnmedian(data1,label1)
    data2, label2 = returnmedian(data2, label2)
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
    return datareduced, label[0:len(datareduced)]

def boosting(X_train,y_train,X_test,y_test,configuration):
    featlist=[]
    for sensor in configuration.get("Sensors"):
        for feature in configuration.get("Features"):
            for axis in ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ"]:#,"MagX","MagY","MagZ","QuatA","QuatB","QuatC","QuatD","V1X","V1Y","V1Z","V2X","V2Y","V2Z","V3X","V3Y","V3Z"]:
                featlist.append(str(sensor)+str(feature)+axis)

    params = {'n_estimators': 30, 'max_depth': 4, 'min_samples_split': 2,
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
        print np.where(feature_importance>20)[0]

        print [featlist[x] for x in np.where(feature_importance>10)[0]]
        print [feature_importance[x] for x in np.where(feature_importance > 10)[0]]
        print [featlist[x] for x in np.where(feature_importance > 50)[0]]

        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.yticks(sorted_idx)
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.show()
    return mse


def regressor():
    #return SVR()
    #return linear_model.Lasso()
    return linear_model.ElasticNet(alpha = 1.0, l1_ratio = 0.7)



