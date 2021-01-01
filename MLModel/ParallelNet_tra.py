import pymongo
import numpy as np
import torch
from torch import nn
import ParallelNet  # type: ignore 
import sys, os
sys.path.append("/scratch/chgwang/XI/Scripts/getData")
import FedSeq # type: ignore 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

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

# evaluation model
def eval_model(model, data_iter, loss):
    model.eval()
    loss_list = []
    for out_seq, label in data_iter:
        out_seq = torch.unsqueeze(out_seq, 1)
        label = convert_one_hot(label)
        with torch.no_grad():
            modeled_label = model(out_seq)
            l = loss(modeled_label, label)
            loss_list.append(l.item())
    return(np.mean(loss_list))


def train_model(lr, batch_size, num_worker, epochs):
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    db  = client["Power_Fault"]
    col_sour = db["data_sour"]
    # all ID of the data, split data by ID
    ids = col_sour.distinct("_id")
    delimit = int(len(ids) * 0.8)
    train_ids = ids[:delimit]
    valid_ids = ids[delimit:]
    # inception Class
    model = ParallelNet.ParallelNet(output_size=6)
    loss = nn.BCELoss()
    offsetList = range(0, 200, 10)
    for epoch in range(epochs):
        model.train()
        for offset in offsetList:
            train_iter = FedSeq.gen_seq(train_ids, batch_size, num_worker, offset=offset)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for out_seq, label in train_iter:
                out_seq = torch.unsqueeze(out_seq, 1)
                label = convert_one_hot(label)
                optimizer.zero_grad()
                modeled_label = model(out_seq)
                l = loss(modeled_label, label)
                l.backward()
                optimizer.step()


        train_loss = []
        valid_loss = []
        for offset in offsetList:
            train_iter = FedSeq.gen_seq(train_ids, batch_size, num_worker, offset=offset)
            valid_iter = FedSeq.gen_seq(valid_ids, batch_size, num_worker, offset=offset)
            train_loss.append(eval_model(model, train_iter, loss))
            valid_loss.append(eval_model(model, valid_iter, loss))
        mean_train = np.mean(train_loss) 
        mean_valid = np.mean(valid_loss)
        scriptPath = os.path.realpath(__file__)
        basedir = os.path.dirname(os.path.dirname(scriptPath))
        datadir = os.path.join(basedir, "DataBase")
        modeldir = os.path.join(datadir, "Model")
        modelPath = os.path.join(modeldir, "PN-%d-%.3f-%.3f"%(epoch+1, mean_train, mean_valid))
        torch.save(model.state_dict(), modelPath)
        print(f"Epoch:{epoch+1} \ntraining loss: {mean_train:.3f} validation loss: {mean_valid:.3f}")
        

if __name__ == "__main__":
    model = ParallelNet.ParallelNet(output_size=6)
    lr = 1e-3
    batch_size = 64
    num_worker = 4
    train_model(lr, batch_size, num_worker, 1)