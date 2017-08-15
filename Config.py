import numpy as np



def loadconfig():
    # Famework Settings
    # --------Data Setup-------------
    # Check one if the framework was used once and the Feature file thereby have already been created
    Featurefilesaved = 0

    # --------Select Data Set------------
    # Xsens setup
    UseXsensData = 1
    if(UseXsensData==1):
        Participant1 = 1
        Participant2 = 1
        Participant3 = 1
        Participant4 = 1
        Participant5 = 1
        Participant6 = 1
        Participant7 = 1
        Participant8 = 1
        Participant9 = 0
        Participant10 = 0
        Session1 = 1
        Session2 = 1
        Session3 = 1
        Session4 = 1
        Session5 = 0
        Unlabeleddata = 1

    # Shimmer setup
    UseShimmerData = 1
    if(UseShimmerData==1):
        Unlabeleddata = 1

    # ---------Select Data Segmentation-----------
    # What amount of data is used to calculate features for each classification
    # Evaluate for every camera walktrought- for the Xsens data around 3-5, for the Shimmer data around 20 strides
    CameraSegment = 0
    StrideSegment = 1

    # --------Sensor Selection-------------
    # Select Sensors, not all available on Shimmer setup
    STE = 0     # STENUM
    LUA = 0     # LEFT Upper Arm
    LLA = 0     # LEFT Lower Arm
    LNS = 1     # LEFT Nordic Walking Stick - Shimmer available
    RUA = 0     # RIGHT Upper Arm
    RLA = 0     # RIGHT Lower Arm
    RNS = 1     # RIGHT Nordic Walking Stick -Shimmer available
    Cen = 0     # Center Back
    LUL = 0     # LEFT Upper Leg
    LLL = 0     # LEFT Lower Leg
    LUF = 0     # LEFT Upper Foot -Shimmer available
    RUL = 0     # Right Upper Leg
    RLL = 0     # Right Lower Leg
    RUF = 0     # Right Upper Foot -Shimmer available

    #Todo Standartisation, Normalisation

    # -------Feature Selection------------

    average = 0
    maximum = 0
    minimum =0
    energy=0
    percentile=0
    percentile5=1
    percentile20=0
    percentile50=0
    skew=0
    maxposition=0
    test=0
    normalizemean=0#collision with nan
    scaleaverage=0
    scalemax=0
    scalemin=0
    scalepercentile5=0
    medianaverage=0
    medianmax=0
    medianmin=0
    maxpositionmed=0
    maxpositionfilter=1
    minpositionfilter=1
    maxfilter=1
    minfilter=1
    filterskew=0
    filterpercentile=1
    maxpositionfilter2 = 1
    minpositionfilter2 = 1
    maxfilter2 = 1
    minfilter2 = 1
    filterskew2 = 1
    filterpercentile2 = 1

    # -------Choose Classifier------------
    # Select Classifier from the Scikit learn library

    #--------Validation------------
    LOPO = 1
    CrossValidation = 0
    #End of configuration part#











    dictionary = [{1:"Participant1",2:"Participant2",3:"Participant3",4:"Participant4",5:"Participant5",6:"Participant6",7:"Participant7",8:"Participant8",9:"Participant9"},
                  {1:"Session1",2:"Session2",3:"Session3",4:"Session4",5:"Session5"},
                  {STE*2:"STE",LUA*24:"LUA",LLA*46:"LLA",LNS*68:"LNS",RUA*90:"RUA",RLA*112:"RLA",RNS*134:"RNS",Cen*156:"CEN",LUL*178:"LUL",LLL*200:"LLL",LUF*222:"LUF",RUL*244:"RUL",RLL*266:"RLL",RUF*288:"RUF"}]



    #Creating a configuration dictionary for further use
    Participant = np.array([Participant1, Participant2, Participant3, Participant4, Participant5,
     Participant6, Participant7, Participant8, Participant9, Participant10])

    Session = np.array([Session1,Session2,Session3,Session4,Session5])
    Sensorlist= [STE*2,LUA*24,LLA*46,LNS*68,RUA*90,RLA*112,RNS*134,Cen*156,LUL*178,LLL*200,LUF*222,RUL*244,RLL*266,RUF*288]
    Sensorlist=[x for x in Sensorlist if x != 0]
    print Sensorlist
    Features = np.array([average,maximum,minimum,energy,percentile,percentile5,percentile20,percentile50,skew,maxposition,test,normalizemean,scaleaverage,scalemax,scalemin,scalepercentile5,
                         medianaverage,medianmax,medianmin,maxpositionmed,maxpositionfilter,minpositionfilter,maxfilter,minfilter,filterskew,filterpercentile,maxpositionfilter2,minpositionfilter2,maxfilter2,minfilter2,filterskew2,filterpercentile2])

    configuration= {"Participants":np.where(Participant==1)[0]+1,
                    "Sessions":np.where(Session==1)[0]+1,
                    "Sensors":Sensorlist,
                    "Features":np.where(Features==1)[0]+1}
    return configuration, dictionary