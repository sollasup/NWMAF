import LoadRawData
import numpy as np
import Config
#TODO RENAME TO NWMAF

configuration = Config.loadconfig()
RawSensorData,ElanGroundTruth = LoadRawData.loaddata(configuration)
