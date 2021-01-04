from torch import nn
from torch.nn import Module# 引入相关的模块
import numpy as np
from sklearn.model_selection import train_test_split
from urllib import request
import os
def downloadData():
    if (not(os.path.exists('wine.data'))):
        print('Downloading with urllib\n')
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        request.urlretrieve(url,'./wine.data')
    else:
        print('Wine.data exists!\n')

# 数据预处理
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr =open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float,curLine[1:]))
        dataMat.append(fltLine)
        labelLine = int(curLine[0])
        labelMat.append(labelLine)
    return np.array(dataMat),np.array(labelMat)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(13, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        # it's simliar to forward function in Pytorch
        # x = self.bn(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

downloadData()
import torch
dataMat, labelMat = loadDataSet('wine.data')
dataMat = np.expand_dims(dataMat, axis=1)
dataMat = dataMat.astype('float32')
labelMat = np.expand_dims(labelMat, axis=1)
labelMat = labelMat.astype(('float32'))
# dataMat = np.transpose(dataMat, (0, 2, 1))
labelMat = torch.tensor(labelMat)
dataMat = (dataMat - np.mean(dataMat))/np.std(dataMat)
dataMat = torch.tensor(dataMat)
labelMat -= 2

x_train, x_test, y_train, y_test = train_test_split(dataMat, labelMat, test_size=0.4,random_state=1)
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader[0])
    for batch_idx in range(len(train_loader)):
        optimizer.zero_grad()
        inputs = train_loader[0][batch_idx]
        targets = train_loader[1][batch_idx]
        outputs = model(inputs)[0]
        l1loss = torch.nn.L1Loss()
        loss = l1loss(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach())
        losses_idx.append(epoch * lens + batch_idx)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.detach()))


def val(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    for batch_idx in range(len(val_loader[0])):
        inputs = val_loader[0][batch_idx]
        targets = val_loader[1][batch_idx]
        batch_size = 1
        outputs = model(inputs)
        pred = np.round(outputs.data)[0]
        pred = pred.numpy().astype('int64')
        targets = targets.numpy().astype('int64')
        acc = np.sum(targets.data == pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
    # print(f'Test Epoch: {epoch} [{batch_idx}/{len(val_loader[0])}]\tAcc: {acc:.6f}')
    print('Test Acc =', total_acc / total_num)

learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-3
epochs = 100
losses = []
losses_idx = []

model = Model()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
import random
for epoch in range(epochs):
    index = [i for i in range(len(x_train))]
    test_index = [i for i in range(len(x_test))]
    random.shuffle(index)
    y_train = y_train[index]
    y_test = y_test[test_index]
    train_loader = (x_train, y_train)
    val_loader = (x_test, y_test)
    train(model, train_loader, optimizer, epoch, losses, losses_idx)
    val(model, val_loader, epoch)
