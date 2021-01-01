import pymongo
import numpy as np
import torch
from torch import nn
import ParallelNet  # type: ignore 
import sys
sys.path.append("/scratch/chgwang/XI/Scripts/getData")
import FedSeq # type: ignore 
import matplotlib.pyplot as plt

def convert_one_hot(labels):
    label_one_hot = torch.zeros(len(labels), 6)
    for index, label_list in enumerate(labels):
        for label in label_list:
            # 编码问题
            label = label - 1
            # label = -1 认为没有任何损坏
            if label == -1:
                continue
            label_one_hot[index][label] = 1
    # the label_one_hot dtype is torch.float32
    return(label_one_hot)
def convert_one_hot_to_label(one_hot_values):
    pass

# inception Class
Inception = ParallelNet.Inception
model = ParallelNet.ParallelNet(Inception, 6)
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db  = client["Power_Fault"]
col_sour = db["data_sour"]
# # all ID of the data, split data by ID
ids = col_sour.distinct("_id")
train_iter = FedSeq.gen_seq(ids, 10, 4, offset=0)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
train_loss = []
for out_seq, label in train_iter:
    out_seq = torch.unsqueeze(out_seq, 1)
    label = convert_one_hot(label)
    optimizer.zero_grad()
    modeled_label = model(out_seq)
    l = loss(modeled_label, label)
    l.backward()
    train_loss.append(l.item())
    optimizer.step()
plt.plot(train_loss)
