import Config
import FeatureConstruction
import matplotlib.pyplot as plt
import Regression
import numpy as np
#TODO Confidenz intervall Randomforestregressor



configuration,dictionaire = Config.loadconfig()

featuredata,featurelabel,featurelabelcamera = FeatureConstruction.Stepfiles(configuration)


#Data,Label=FeatureConstruction.to_dataframe(featuredata,featurelabel,configuration,dictionaire)

featurematrix = FeatureConstruction.Featurefiles(configuration)
#featurematrixcam = FeatureConstruction.Featurefilescamera(configuration)

Regression.linearRegression(featurematrix,featurelabel,configuration)


























