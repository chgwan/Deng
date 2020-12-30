# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
import pandas as pd 
import os
import fnmatch
import pymongo

def readDirectory(path:str, col:pymongo.collection.Collection):
    period = 0.02
    sample_step = 0.001
    if path[-2:].isdigit():
        label = [path[-1], path[-2]]
    elif not path[-1].isdigit():
        label = [0]
    else:
        label = [path[-1]]
    # set data target label
    dataDict = {"label":label}
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, "*.mat"):
            matDict = scio.loadmat(os.path.join(path, file))
            for k, v in matDict.items():
                if type(v) == np.ndarray:
                    time = v[:,0]
                    data = v[:,1]
                    maxTime = max(time)
                    # 0.04 is two period of sample period.
                    minTime = maxTime - 2*period
                    timeAxis = np.arange(minTime, maxTime, sample_step)
                    resampled_data = np.interp(timeAxis, time, data)
                    dataDict[k] = resampled_data.tolist()
    dataDict["time"] = timeAxis.tolist() # type: ignore
    col_sour.insert_one(dataDict)



if __name__ == "__main__":
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    basepath = os.path.dirname(os.path.dirname(scriptDir))
    dataDir = os.path.join(basepath, "data")
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db  = client["Power_Fault"]
    col_sour = db["data_sour"]
    for i in range(1,21):
        split_data = os.path.join(dataDir, str(i))
        for tempName in os.listdir(split_data): 
            tempPath = os.path.join(split_data, tempName)
            if os.path.isfile(tempPath):
                continue
            elif tempName == "1Normal":
                readDirectory(tempPath, col_sour)
            else:
                for sub_folder in os.listdir(tempPath):
                    sub_folder_path = os.path.join(tempPath, sub_folder)
                    readDirectory("/scratch/chgwang/XI/data/1/1Normal", col_sour)
        
                
