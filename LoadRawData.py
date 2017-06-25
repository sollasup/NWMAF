import LoadRawData.py
import numpy as np

#Famework Settings
#--------Data Setup-------------
#Check one if the framework was used once and the Feature file thereby have already been created
Featurefilesaved = 0

#--------Select Data Set------------
#Xsens setup
UseXsensData = 1
if(UseXsensData==1):
    Session1 = 1
    Session2 = 1
    Session3 = 1
    Session4 = 1
    Session5 = 1
    Unlabeleddata = 1

#Shimmer setup
UseShimmerData = 1
if(UseShimmerData==1):
    Unlabeleddata = 1

#---------Select Data Segmentation-----------
#What amount of data is used to calculate features for each classification
#Evaluate for every camera walktrought- for the Xsens data around 3-5, for the Shimmer data around 20 strides
CameraSegment = 1
StrideSegment = 1

#--------Sensor Selection-------------
