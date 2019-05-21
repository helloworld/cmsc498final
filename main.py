import torch
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
import wget
import urllib

# datadir = './data'
# if not os.path.exists(datadir):
#     os.makedirs(datadir)

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls' 
# filename = os.path.join(datadir, 'default of credit card clients.xls')

# if not os.path.isfile(filename):
#     wget.download(url, out=filename)

# df = pd.read_excel(filename, header=1)

# df.columns = [x.lower() for x in df.columns]
# df = df.rename(index=str, columns={"pay_0": "pay_1"})
# df = df.drop('id', axis=1)
# df.columns

device = torch.device('cpu')

X_train = pd.read_csv("./data/X_train.txt", delim_whitespace=True, header=None)
y_train = pd.read_csv("./data/y_train.txt", delim_whitespace=True, header=None)
s_train = pd.read_csv("./data/subject_train.txt", header=None)

X_test = pd.read_csv("./data/X_test.txt", delim_whitespace=True, header=None)
y_test = pd.read_csv("./data/y_test.txt", delim_whitespace=True, header=None)
s_test = pd.read_csv("./data/subject_test.txt", header=None)

# Change s index to be 0-29 instead of 1-30
s_train[0] = s_train[0] - 1
s_test[0] = s_test[0] - 1


# Change y index to be 0-5 instead of 1-6
y_train[0] = y_train[0] - 1
y_test[0] = y_test[0] - 1

# Tensor for the training data and sensitive attribute
X_tensor = torch.tensor(X_train.values, device=device).double()
s_tensor = torch.tensor(s_train.values, device=device).double()
y_tensor = torch.tensor(y_train.values, device=device).double()

# Tensor for the testing data and sensitive attribute
X_tensor_test = torch.tensor(X_test.values, device=device).double()
s_tensor_test = torch.tensor(s_test.values, device=device).double()
y_tensor_test = torch.tensor(y_test.values, device=device).double()

# Append 100D noise to the training tensor for the generator training
noise = torch.randn([X_train.shape[0], 100], device=device).double()
noised_tensor = torch.cat((X_tensor, noise), 1)


# Append 100D noise to the testing tensor for the generator training
noise_test = torch.randn([X_test.shape[0], 100], device=device).double()
noised_tensor_test = torch.cat((X_tensor_test, noise_test), 1)

model_gen = torch.nn.Sequential(
          # Layer 1 - 661 -> 512
          torch.nn.Linear(661, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(512, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 3 - 512 -> 512
          torch.nn.Linear(512, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Output
          torch.nn.Linear(512, 561),
        ).to(device)

model_adv = torch.nn.Sequential(
          # Layer 1 - 561 -> 512
          torch.nn.Linear(561, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(512, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 3 - 512 -> 256
          torch.nn.Linear(512, 256),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(256),
          # Layer 4 - 256 -> 128
          torch.nn.Linear(256, 128),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(128),
          # Output - 128 -> 30
          torch.nn.Linear(128, 30),
        ).to(device)

model_class = torch.nn.Sequential(
          # Layer 1 - 561 -> 512
          torch.nn.Linear(561, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(512, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 3 - 512 -> 256
          torch.nn.Linear(512, 256),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(256),
          # Layer 4 - 256 -> 128
          torch.nn.Linear(256, 128),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(128),
          # Output - 128 -> 6
          torch.nn.Linear(128, 6),
        ).to(device)

optim_gen = torch.optim.Adam(model_gen.parameters())
loss_gen = torch.nn.CrossEntropyLoss()

optim_adv = torch.optim.Adam(model_adv.parameters())
loss_adv = torch.nn.CrossEntropyLoss()

optim_class = torch.optim.Adam(model_class.parameters())
loss_class = torch.nn.CrossEntropyLoss()

NUM_EPOCHS_GEN = 5
NUM_EPOCHS_ADV = 1
NUM_TOTAL_ITER = 15
DISTORTION_WEIGHT = 0.1
D = 3

train_loader = torch.utils.data.DataLoader(
    torch.cat((noised_tensor, s_tensor), 1), 
    batch_size=512, 
    shuffle=True)

loss_by_epoch_g = []
loss_by_epoch_a = []

for epoch in range(NUM_TOTAL_ITER):
  print("Epoch: ", epoch)
  for j in range(NUM_EPOCHS_GEN):
    total_loss_g = 0
    total_loss_d = 0
    num = 0
    for batch in train_loader:
      x, s = batch[:, 0:-1], batch[:, -1].long()
      x_hat = model_gen(x.float())
      adv_pred = model_adv(x_hat.float())

    
      loss_g = -loss_adv(adv_pred, s)
      dist_loss = torch.dist(x_hat, x[:, 0:561].float()) * DISTORTION_WEIGHT
      if dist_loss < D:
        dist_loss = 0
      total_loss_d += dist_loss
      
      loss_g += dist_loss

      num += 1
      total_loss_g += loss_g

      optim_gen.zero_grad()
      loss_g.backward()
      optim_gen.step()
    epch_loss = (total_loss_g/num).item()
    loss_by_epoch_g.append(epch_loss)
    print("Gen loss: ", epch_loss)
    
  for j in range(NUM_EPOCHS_ADV):
    total_loss_a = 0
    num = 0
    for batch in train_loader:
      x, s = batch[:, 0:-1], batch[:, -1].long()
      
      x_hat = model_gen(x.float())
      
      s_pred = model_adv(x_hat)

      loss_a = loss_adv(s_pred, s)
      num += 1
      total_loss_a += loss_a
      
      optim_adv.zero_grad()
      loss_a.backward(retain_graph=True)
      optim_adv.step()
    epch_loss = (total_loss_a/num).item()
    loss_by_epoch_a.append(epch_loss)
    print("Adv loss: ", (total_loss_a/num).item())
  print("\n")  

ax = plt.axes()
ax.plot(range(NUM_TOTAL_ITER * NUM_EPOCHS_GEN), loss_by_epoch_g)
ax.set(xlabel="Epochs", ylabel="Generator Loss", Title="Generator Loss Curve W/ D=7")

ax = plt.axes()
ax.plot(range(NUM_TOTAL_ITER * NUM_EPOCHS_ADV), loss_by_epoch_a)
ax.set(xlabel="Epochs", ylabel="Adversary Loss", Title="Adversary Loss Curve")

out_class = model_adv(X_tensor.float())
v, i = torch.max(out_class, 1)
print((s_tensor.squeeze().int() == i.int()).nonzero().shape[0]/s_tensor.shape[0])

gen_noised = model_gen(noised_tensor.float())

out_class = model_adv(gen_noised.float())
v, i = torch.max(out_class, 1)
print((s_tensor.squeeze().int() == i.int()).nonzero().shape[0]/s_tensor.shape[0])

class_loader = torch.utils.data.DataLoader(
    torch.cat((X_tensor, y_tensor), 1), 
    batch_size=512, 
    shuffle=True)

for epoch in range(20):
  loss_avg = 0
  num = 0
  for batch in class_loader:
    x, y = batch[:, 0:-1], batch[:, -1]
    y_pred = model_class(x.float())

    loss = loss_class(y_pred, y.long())
    loss_avg += loss
    num += 1

    optim_class.zero_grad()
    loss.backward()
    optim_class.step()
  print("loss: ", (loss_avg/num).item())

  out_class = model_class(X_tensor_test.float())
v, i = torch.max(out_class, 1)
print((y_tensor_test.squeeze().int() == i.int()).nonzero().shape[0]/y_tensor_test.shape[0])

model_class = torch.nn.Sequential(
          # Layer 1 - 561 -> 512
          torch.nn.Linear(561, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(512, 512),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(512),
          # Layer 3 - 512 -> 256
          torch.nn.Linear(512, 256),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(256),
          # Layer 4 - 256 -> 128
          torch.nn.Linear(256, 128),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(128),
          # Output - 128 -> 6
          torch.nn.Linear(128, 6),
        ).to(device)

optim_class = torch.optim.Adam(model_class.parameters())
loss_class = torch.nn.CrossEntropyLoss()

gen_noised = model_gen(noised_tensor.float())

class_loader = torch.utils.data.DataLoader(
    torch.cat((gen_noised.float(), y_tensor.float()), 1), 
    batch_size=512, 
    shuffle=True)

for epoch in range(20):
  loss_avg = 0
  num = 0
  for batch in class_loader:
    x, y = batch[:, 0:-1], batch[:, -1]
    y_pred = model_class(x.float())

    loss = loss_class(y_pred, y.long())
    loss_avg += loss
    num += 1

    optim_class.zero_grad()
    loss.backward(retain_graph=True)
    optim_class.step()
  print("loss: ", (loss_avg/num).item())

  out_class = model_class(X_tensor_test.float())
v, i = torch.max(out_class, 1)
print((y_tensor_test.squeeze().int() == i.int()).nonzero().shape[0]/y_tensor_test.shape[0])