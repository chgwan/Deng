# -*- coding: utf-8 -*-
from typing import Generator
import scipy.io as scio
import numpy as np
import pandas as pd 
import os
import fnmatch
import pymongo


def readDirectory(path:str, col:pymongo.collection.Collection):
    period = 0.02
    # our sample step.
    sample_step = 0.001
    # this data can got from the header.
    source_freq = 1250000
    fault_point = 6250000
    # get the labels
    path_splited = path.split("/")
    labels = []
    if path_splited[-1][0] == 0:
        labels.append(0)
    else:
        labels.append(path_splited[-1][0])
        if path_splited[-1][1].isdigit:
            labels.append(path_splited[-1][1])
    data = np.loadtxt(path, skiprows=16, delimiter=",", usecols=range(1,4))
    data_dict = {"label":labels}
    



# def retrieve_eles(path, path_list):
#     ele_list = os.listdir(path)
#     for ele in ele_list:
#         ele_path = os.path.join(path, ele)
#         if os.path.isfile(ele_path):
#             path_list.append(ele_path)
#         else:
#             return(ele_path, path_list)

def retrieve_files(path:str) -> Generator:
    path_gen = os.walk(path)
    for root, _, files in path_gen:
        for name in files:
           yield(os.path.join(root,name))
        

if __name__ == "__main__":
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    basepath = os.path.dirname(os.path.dirname(scriptDir))
    dataDir = os.path.join(basepath, "data")
    dataDir = os.path.join(dataDir, "邓茜实验数据")
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db  = client["Power_Fault"]
    col_sour = db["data_sour_exper"]

    file_gen = retrieve_files(dataDir)
    for file in file_gen:
        print(file)
    
                
