#!/usr/bin/env python
# coding: utf-8

# # GAPF for the UCI Credit Card Data Set

# In[1]:


import torch
import torch.utils.data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


device = torch.device('cpu')


# ### Import the UCI Credit Card Dataset
# 
# The goal of GAPF in this scenario is to decorrelate the original data from the sex each person is associated with. Ideally data scientists would still be able to learn models to predict y without being able to predict s.

# In[2]:


import os
import shutil

datadir = './data'
if not os.path.exists(datadir):
    os.makedirs(datadir)


# In[3]:


# Get the dataset from UCI

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls' 
filename = os.path.join(datadir, 'default of credit card clients.xls')

if not os.path.isfile(filename):
    wget.download(url, out=filename)


# **Clean up**

# In[4]:


df = pd.read_excel(filename, header=1)
df.columns = [x.lower() for x in df.columns]
df = df.rename(index=str, columns={"pay_0": "pay_1"})
df = df.drop('id', axis=1)
df['target'] = df['default payment next month'].astype('category')

df.columns


# In[5]:


X = df.drop(['default payment next month', 'target'], axis=1)


# In[6]:


encoders = {}
X_num = X.copy()

label_cols = ['sex', 'education', 'marriage', 'age', 'target']

for col in X_num.columns.tolist():
    if col in label_cols:
        encoders[col] = preprocessing.LabelEncoder().fit(X_num[col])
        X_num[col] = encoders[col].transform(X_num[col])
X_num = X_num.drop(['sex'], axis=1)

y = df['target'].copy()
s = encoders['sex'].transform(X['sex'])


# # Visualizations

# In[7]:


fig, ax = plt.subplots(1,2)
fig.set_size_inches(20,5)
fig.suptitle('Defaulting by absolute numbers, for various demographics')

df['sex'] = df['sex'].astype('category').cat.rename_categories(['M', 'F'])
df['marriage'] = df['marriage'].astype('category').cat.rename_categories(['na', 'married', 'single', 'other'])
# df['education'] = df['education'].astype('category').cat.rename_categories(['na', 'graduate school', 'university', 'high school', 'other', 'other2'])

d = df.groupby(['target', 'sex']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[0])

d = df.groupby(['target', 'marriage']).size()
p = d.unstack(level=1).plot(kind='bar', ax=ax[1])


# In[8]:


fig, ax = plt.subplots(1,2)
fig.set_size_inches(20,5)
fig.suptitle('Defaulting by relative numbers given each class, for various demographics')

d = df.groupby(['target', 'sex']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[0])

d = df.groupby(['target', 'marriage']).size().unstack(level=1)
d = d / d.sum()
p = d.plot(kind='bar', ax=ax[1])


# ## Create Tensors

# In[9]:


X_tensor = torch.tensor(X_num.values, device=device).double()
noise = torch.randn([X_tensor.shape[0], 5], device=device).double()
X_noised = torch.cat((X_tensor, noise), 1)

s_tensor = torch.tensor(s, device=device).double().unsqueeze(1)
y_tensor = torch.tensor(y.values, device=device).double().unsqueeze(1)
print("X", X_tensor.shape, "X_noised", X_noised.shape, "s", s.shape, "y", y.shape)


# ## Create Models

# The GAPF model consists of two primary adversarial models.
# 
# The adversary takes in the output of the generator and outputs the sensitive attribute.
# 
# <img src="./project_webpage/images/gapf.png"></img>

# In[10]:


SIZE_1 = 256
SIZE_2 = 128

dim_with_noise = 27
dim = 22

model_gen = torch.nn.Sequential(
          # Layer 1 - 28 -> 512
          torch.nn.Linear(dim_with_noise, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 3 - 512 -> 512
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Output
          torch.nn.Linear(SIZE_1, dim),
        ).to(device)

model_adv = torch.nn.Sequential(
          # Layer 1 - 23 -> 512
          torch.nn.Linear(dim, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 3 - 512 -> SIZE_1
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 4 - SIZE_1 -> SIZE_2
          torch.nn.Linear(SIZE_1, SIZE_2),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_2),
          # Output - SIZE_2 -> 2
          torch.nn.Linear(SIZE_2, 2),
        ).to(device)

model_class = torch.nn.Sequential(
          # Layer 1 - 23 -> 512
          torch.nn.Linear(dim, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 3 - 512 -> SIZE_1
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 4 - SIZE_1 -> SIZE_2
          torch.nn.Linear(SIZE_1, SIZE_2),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_2),
          # Output - SIZE_2 -> 2
          torch.nn.Linear(SIZE_2, 2),
        ).to(device)


# In[11]:


optim_gen = torch.optim.Adam(model_gen.parameters())
loss_gen = torch.nn.CrossEntropyLoss()

optim_adv = torch.optim.Adam(model_adv.parameters())
loss_adv = torch.nn.CrossEntropyLoss()

optim_class = torch.optim.Adam(model_class.parameters())
loss_class = torch.nn.CrossEntropyLoss()


# In[14]:


from torchviz import make_dot, make_dot_from_trace

x = torch.randn(256, 22)
make_dot(model_adv(x), params=dict(model_adv.named_parameters()))


# ## Train Adversarially
# 
# We alternate training between the generator and adversary.
# 
# The adversary loss is simply the cross entropy loss of its output.
# 
# The generators loss is the negative loss of the adversary plus a limited distortion metric to limit how much it modifies the original data.

# In[ ]:


NUM_EPOCHS_GEN = 5
NUM_EPOCHS_ADV = 1
NUM_TOTAL_ITER = 1
DISTORTION_WEIGHT = 0.05
D = 3

train_loader = torch.utils.data.DataLoader(
    torch.cat((X_noised, s_tensor), 1), 
    batch_size=50, 
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
            dist_loss = torch.dist(x_hat, x[:, 0:22].float()) * DISTORTION_WEIGHT
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


# In[ ]:


ax = plt.axes()
ax.plot(range(NUM_TOTAL_ITER * NUM_EPOCHS_GEN), loss_by_epoch_g)
ax.set(xlabel="Epochs", ylabel="Generator Loss", Title="Generator Loss Curve W/ D=7")


# In[ ]:


ax = plt.axes()
ax.plot(range(NUM_TOTAL_ITER * NUM_EPOCHS_ADV), loss_by_epoch_a)
ax.set(xlabel="Epochs", ylabel="Adversary Loss", Title="Adversary Loss Curve")


# # Testing
# ## Test Adversary before and after decorrelation
# 
# Test the adversary to check that the decorrelation works and lowers the adversary's accuracy

# In[ ]:


out_class = model_adv(X_tensor.float())
v, i = torch.max(out_class, 1)
print((s_tensor.squeeze().int() == i.int()).nonzero().shape[0]/s_tensor.shape[0])


# In[ ]:


gen_noised = model_gen(X_noised.float())

out_class = model_adv(gen_noised.float())
v, i = torch.max(out_class, 1)
print((s_tensor.squeeze().int() == i.int()).nonzero().shape[0]/s_tensor.shape[0])


# ## Test Classifier Before and After Decorrelation
# 
# Test the classifier to see if it can still train and predict accuractely with the decorrelated data.

# ### Before

# In[17]:


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


# In[18]:


out_class = model_class(X_tensor_test.float())
v, i = torch.max(out_class, 1)
print((y_tensor_test.squeeze().int() == i.int()).nonzero().shape[0]/y_tensor_test.shape[0])


# ### After

# In[ ]:


model_class = torch.nn.Sequential(
          # Layer 1 - 23 -> 512
          torch.nn.Linear(dim, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 2 - 512 -> 512
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 3 - 512 -> SIZE_1
          torch.nn.Linear(SIZE_1, SIZE_1),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_1),
          # Layer 4 - SIZE_1 -> SIZE_2
          torch.nn.Linear(SIZE_1, SIZE_2),
          torch.nn.LeakyReLU(),
          torch.nn.BatchNorm1d(SIZE_2),
          # Output - SIZE_2 -> 2
          torch.nn.Linear(SIZE_2, 2),
        ).to(device)

optim_class = torch.optim.Adam(model_class.parameters())
loss_class = torch.nn.CrossEntropyLoss()

gen_noised = model_gen(X_noised.float())

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


# In[ ]:


out_class = model_class(X_tensor_test.float())
v, i = torch.max(out_class, 1)
print((y_tensor_test.squeeze().int() == i.int()).nonzero().shape[0]/y_tensor_test.shape[0])


# In[ ]:




