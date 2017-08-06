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
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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
import time
from sklearn import svm,datasets
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import math
from sklearn.preprocessing import Imputer
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn import linear_model
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
            average =(moving_average(Dataparticipants[i][:,t],11))
            Dataparticipants[i][:len(average),t]=average
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imp.fit(Dataparticipants[i])
            Dataparticipants[i]= imp.transform(Dataparticipants[i])
            std_scale =preprocessing.StandardScaler().fit(Dataparticipants[i])
            Dataparticipants[i]=std_scale.transform(Dataparticipants[i])
    if False:  # Look at each participants features
        plt.subplot(611)
        plt.plot(Dataparticipants[0][:, 0:3])
        plt.plot(Labelparticipants[0][:, 1])
        plt.subplot(613)
        plt.plot(Dataparticipants[2][:, 0:3])
        plt.plot(Labelparticipants[2][:, 1])
        plt.subplot(614)
        plt.plot(Dataparticipants[3][:, 0:3])
        plt.plot(Labelparticipants[3][:, 1])
        plt.subplot(615)
        plt.plot(Dataparticipants[4][:, 0:3])
        plt.plot(Labelparticipants[4][:, 1])
        plt.subplot(616)
        plt.plot(Dataparticipants[5][:, 0:3])
        plt.plot(Labelparticipants[5][:, 1])
        plt.subplot(612)
        plt.plot(Dataparticipants[1][:, 0:3])
        plt.plot(Labelparticipants[1][:, 1])
        plt.show()
    print range(0, len(Dataparticipants) - 1)
    for i in range(0, len(Dataparticipants)):
        Indices = range(0,len(Dataparticipants))
        Indices.remove(i)
        print Indices
        X_train= np.vstack([Dataparticipants[p] for p in Indices])
        y_train = np.vstack([Labelparticipants[p] for p in Indices])[:,1]
        X_test = Dataparticipants[i]
        y_test = Labelparticipants[i][:, 1]
        pca = PCA(n_components=2).fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        regr = linear_model.LinearRegression()
        regr.fit(X_train,y_train)

        print mean_squared_error(y_test,regr.predict(X_test))
        clf = RandomForestClassifier()

        if True:
            plt.plot(X_train)
            plt.plot(y_train)

            plt.show()
        clf.fit(X_train,y_train)


        if True:
            plt.plot(y_test,label="Ground truth")

            plt.plot(regr.predict((X_test)),label="Regression Prediction")
            #plt.plot(clf.predict(X_test),label="Classification Prediction")
            plt.legend()
            plt.show()
        print clf.score(X_test,y_test)



    # Train the model using the training sets

