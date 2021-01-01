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
from sklearn.metrics import accuracy_score
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
# def accuracy_sco
def map_conti_discr(labels, theta):
    discrete_labels = torch.zeros_like(labels)
    ids = labels > theta
    discrete_labels[ids] = 1
    discrete_labels = discrete_labels.int()
    return(discrete_labels)

# evaluation model
def eval_model(model, data_iter, loss):
    model.eval()
    loss_list = []
    acc_list = []
    for out_seq, label in data_iter:
        out_seq = torch.unsqueeze(out_seq, 1)
        label = convert_one_hot(label)
        with torch.no_grad():
            modeled_label = model(out_seq)
            l = loss(modeled_label, label)
            loss_list.append(l.item())
            modeled_label = map_conti_discr(modeled_label, theta=0.5)
            label = label.int()
            acc = accuracy_score(label, modeled_label)
            acc_list.append(acc)
    return(np.mean(loss_list), np.mean(acc_list))
# initization model
def xavier_init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(m._parameters[param])
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

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
    # initialize the model
    model.apply(xavier_init_weights)
    loss = nn.BCELoss()
    offsetList = range(0, 200, 10)
    lambdaLR = lambda epoch: 0.95**(epoch // 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                    lr_lambda=lambdaLR)
    for epoch in range(epochs):
        model.train()
        for offset in offsetList:
            train_iter = FedSeq.gen_seq(train_ids, batch_size, num_worker, offset=offset)
            for out_seq, label in train_iter:
                out_seq = torch.unsqueeze(out_seq, 1)
                label = convert_one_hot(label)
                optimizer.zero_grad()
                modeled_label = model(out_seq)
                l = loss(modeled_label, label)
                l.backward()
                optimizer.step()

        tra_loss_list = []
        val_loss_list = []
        tra_accuracy_list = []
        val_acccuracy_list = []
        for offset in offsetList:
            train_iter = FedSeq.gen_seq(train_ids, batch_size, num_worker, offset=offset)
            valid_iter = FedSeq.gen_seq(valid_ids, batch_size, num_worker, offset=offset)
            tra_loss, tra_accuracy = eval_model(model, train_iter, loss)
            val_loss, val_accuracy = eval_model(model, valid_iter, loss)
            tra_loss_list.append(tra_loss)
            val_loss_list.append(val_loss)
            tra_accuracy_list.append(tra_accuracy)
            val_acccuracy_list.append(val_accuracy)
        scheduler.step()
        mean_tra_loss = np.mean(tra_loss_list) 
        mean_val_loss = np.mean(val_loss_list)
        mean_tra_acc = np.mean(tra_accuracy_list)
        mean_val_acc = np.mean(val_acccuracy_list)
        scriptPath = os.path.realpath(__file__)
        basedir = os.path.dirname(os.path.dirname(scriptPath))
        basedir = os.path.dirname(basedir)
        datadir = os.path.join(basedir, "DataBase")
        modeldir = os.path.join(datadir, "Model")
        modelPath = os.path.join(modeldir, "PN-%d-%.3f-%.3f"%(epoch+1, mean_tra_acc, mean_val_acc))
        torch.save(model.state_dict(), modelPath)
        print(f"Epoch:{epoch+1} \ntraining loss: {mean_tra_loss:.3f} validation loss: {mean_val_loss:.3f}")
        

if __name__ == "__main__":
    model = ParallelNet.ParallelNet(output_size=6)
    lr = 1e-3
    batch_size = 64
    num_worker = 4
    num_epochs = 100
    train_model(lr, batch_size, num_worker, num_epochs)