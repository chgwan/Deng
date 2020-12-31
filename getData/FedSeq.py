# -*- coding: utf-8 -*-
import numpy as np
import pymongo
import sys
from torch.nn.utils.rnn import pad_sequence
import torch
import math
import os
import typing
import pandas as pd
import multiprocessing as mp

def getId(id, offset):
    # timeAxis = np.arange(start, end, sampleInterval)
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db = client["Power_Fault"]
    col_sour = db["data_sour"]
    id_dict = col_sour.find_one({"_id":id})
    period_points = 200
    timeAxis = id_dict["time"]
    timeAxis = timeAxis[offset:offset+period_points]
    id_data = np.empty((len(timeAxis), 0))
    nodes = ["ia", "ib", "ic"]
    for node in nodes:
        data = id_dict[node]
        data = data[offset:offset+period_points]
        data = np.array(data)[:,np.newaxis]
        id_data = np.append(id_data, data, axis=1)
    id_data = torch.tensor(id_data, dtype=torch.float32)
    id_label = np.array(id_dict["label"]).astype(int)
    id_label = torch.tensor(id_label, dtype=torch.int)
    return(id_data, id_label)

def getPerIds(queue, ids, offset):
    id_data_list = []
    id_labels = []
    for id in ids:
        id_data, id_label = getId(id, offset)
        id_data_list.append(id_data)
        id_labels.append(id_label)
    queue.put((id_data_list, id_labels))

def getBatchData(map_ids:list, num_worker:int, offset:int):
    per_id_base = len(map_ids) // num_worker
    per_id_remainder = len(map_ids) % num_worker
    perpared_batch_queue = mp.Manager().Queue(num_worker)
    pros = []
    batch_data = []
    batch_labels = []
    for i in range(num_worker):
        if i < per_id_remainder:
            per_id = per_id_base + 1
        else:
            per_id = per_id_base
        per_pro_ids = map_ids[i*per_id:(i+1)*per_id]
        getPerIds_pro = mp.Process(target=getPerIds, args=(perpared_batch_queue, per_pro_ids, offset))
        getPerIds_pro.start()
        pros.append(getPerIds_pro)
    for _ in range(num_worker):
        id_data_list, id_labels = perpared_batch_queue.get()
        batch_data.extend(id_data_list)
        batch_labels.extend(id_labels)
    batch_data = torch.stack(batch_data)
    for pro in pros:
        pro.join()
    return(batch_data, batch_labels)


def gen_seq(ids:list, batch_size:int, num_worker:int, offset=0):
    batch_amount = math.ceil(len(ids) / batch_size)
    for i in range(batch_amount):
        map_ids = ids[i*batch_size:min(len(ids), (i+1)*batch_size)]
        out_seq, label = getBatchData(map_ids, num_worker, offset)
        yield(out_seq, label)
if __name__ == "__main__":
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db  = client["Power_Fault"]
    col_sour = db["data_sour"]
    # # all ID of the data, split data by ID
    ids = col_sour.distinct("_id")

    id = ids[0]
    data, _ = getId(id, offset=0)
    print(data.size())

    ids = ids[:10]
    batch_data, _ = getBatchData(ids, 4, offset=0)
    print(batch_data.size())

    train_iter = gen_seq(ids, 2, num_worker=2)
    for out_seq, label in train_iter:
        print(out_seq.shape)
