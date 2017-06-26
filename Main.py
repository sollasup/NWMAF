import LoadRawData.py
import numpy as np

# Famework Settings
# --------Data Setup-------------
# Check one if the framework was used once and the Feature file thereby have already been created
Featurefilesaved = 0

# --------Select Data Set------------
# Xsens setup
UseXsensData = 1
if(UseXsensData==1):
    Session1 = 1
    Session2 = 1
    Session3 = 1
    Session4 = 1
    Session5 = 1
    Unlabeleddata = 1

# Shimmer setup
UseShimmerData = 1
if(UseShimmerData==1):
    Unlabeleddata = 1

# ---------Select Data Segmentation-----------
# What amount of data is used to calculate features for each classification
# Evaluate for every camera walktrought- for the Xsens data around 3-5, for the Shimmer data around 20 strides
CameraSegment = 1
StrideSegment = 1

# --------Sensor Selection-------------
# Select Sensors, not all available on Shimmer setup
STE = 1     # STENUM
LUA = 1     # LEFT Upper Arm
LLA = 1     # LEFT Lower Arm
LNS = 1     # LEFT Nordic Walking Stick - Shimmer available
RUA = 1     # RIGHT Upper Arm
RLA = 1     # RIGHT Lower Arm
RNS = 1     # RIGHT Nordic Walking Stick -Shimmer available
Cen = 1     # Center Back
LUL = 1     # LEFT Upper Leg
LLL = 1     # LEFT Lower Leg
LUF = 1     # LEFT Upper Foot -Shimmer available
RUL = 1     # Right Upper Leg
RLL = 1     # Right Lower Leg
RUF = 1     # Right Upper Foot -Shimmer available

# -------Feature Selection------------

#TODO Feature list

# -------Choose Classifier------------
# Select Classifier from the Scikit learn library

#--------Validation------------
LOPO = 1
CrossValidation = 0