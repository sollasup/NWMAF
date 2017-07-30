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
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns


def linearRegression(DataList, Labellist):
    print range(0, len(DataList) - 1)
    X_train= np.concatenate([DataList[i] for i in range(0,len(DataList)-1)])
    y_train = np.concatenate([DataList[i] for i in range(0,len(DataList)-1)])[:,1]
    X_test = DataList[len(DataList)-1]
    y_test = Labellist[len(Labellist)- 1][:, 1]

    regr = linear_model.LinearRegression()

    # Train the model using the training sets

