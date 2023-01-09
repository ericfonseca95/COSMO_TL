# IMPORT PACKAGES
import os
import gc
import time
import pickle
import random
#import webbrowser
## DASK
import dask.dataframe as dd
import dask
import dask.array as da
from dask_ml.preprocessing import StandardScaler

import torch
from torch.nn import ModuleList
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, r2_score
from joblib import parallel_backend

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
from . import sigma_functions as sf
import pickle
    

class NN(nn.Module):
    def __init__(self, n_inputs=106, n_outputs=1, layers=3, layer_size=75):
        """
        Initialize the NN model with a given number of layers and layer size.
        
        Args:
        - num_classes (int): The number of classes the model should output.
        - layers (int): The number of fully connected layers in the model.
        - layer_size (int): The number of neurons in each fully connected layer.
        """
        super(NN, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.layer_size = layer_size
        self.layers = layers
        
        self.fc1 = nn.Linear(n_inputs, layer_size)
        self.fcs = ModuleList([nn.Linear(layer_size, layer_size) for i in range(layers)])
        self.fout = nn.Linear(layer_size, n_outputs)

    def forward(self, x):
        """
        Forward pass of the NN model.
        
        Args:
        - x (torch.Tensor): The input tensor of shape (batch_size, input_dim)
        
        Returns:
        - y (torch.Tensor): The output tensor of shape (batch_size, num_classes)
        """
        try:
            x = F.relu(self.fc1(x))
        except:
            try:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size)
            except:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size).to('cuda')
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = self.fout(x)
        return x
  
class Conv(nn.Module):
    def __init__(self, layers=3, n_outputs=1, layer_size=75, n_inputs=52, kernel_size=3, out_channels=(3, 10)):
        """
        Initialize the convolutional model with a given number of layers, output size, and layer size.
        
        Args:
        - layers (int): The number of fully connected layers in the model.
        - n_outputs (int): The number of output classes for the model.
        - layer_size (int): The number of neurons in each fully connected layer.
        - n_inputs (int): The number of input features for the model.
        - kernel_size (int): The kernel size for the convolutional layers.
        - out_channels (tuple): The number of output channels for the convolutional layers.
        """
        super(Conv, self).__init__()
        self.layers = layers
        self.layer_size = layer_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.c1 = nn.Conv1d(1, out_channels[0], 2)
        self.c2 = nn.Conv1d(out_channels[0], out_channels[1], 2)
        self.p1 = nn.AvgPool1d(2, 2)
        self.fc1 = nn.Linear(138, self.layer_size)
        self.fcs = ModuleList([nn.Linear(self.layer_size, self.layer_size) for i in range(layers-1)])
        self.fout = nn.Linear(self.layer_size, n_outputs)
        
    
    def forward(self,x):
        rows = x.shape[0]
        cols = x.shape[1]
        x = x.reshape(rows, 1, cols)
        x = F.relu(self.c1(x))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p1(x)
        x = torch.flatten(x).reshape(rows, -1)
        try:
            x = F.relu(self.fc1(x))
        except:
            try:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size)
                x = F.relu(self.fc1(x))
            except:
                self.fc1 = nn.Linear(x.shape[1], self.layer_size).to('cuda')
                x = F.relu(self.fc1(x))
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = self.fout(x)
        return x


def run_Pytorch(model, X_train, Y_train, n_epochs=100, learning_rate=1e-5, batch_size=int(1e5), device='cuda'):
    torch.cuda.empty_cache()
    losses = train_pytorch(model, 
                 X_train, 
                 Y_train,
                 n_epochs=n_epochs,
                 batch_size=batch_size, 
                 learning_rate=learning_rate)
    return losses

def run_epochs(model, X_train, Y_train, loss_func, optimizer, batches, n_epochs=100, device='cuda'):
    t1 = time.time()
    losses = []
    for epoch in range(n_epochs):
        for i in batches:
           # i = indicies[i]
            optimizer.zero_grad()   # clear gradients for next train
            x = X_train[i,:].to(device)
            y = Y_train[i,:].to(device)
            pred = model(x)
            loss = loss_func(pred, y) # must be (1. nn output, 2. target)
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        losses.append(loss)
        torch.cuda.empty_cache()
        gc.collect()
        if epoch%10 == 0:
            t2 = time.time()
            print('EPOCH : ', epoch,', dt: ',
                  t2 - t1, 'seconds, losses :', 
                  float(loss.detach().cpu())) 
            t1 = time.time()
    return losses

def train_pytorch(model, X_train, Y_train, n_epochs=1000, batch_size=int(1e3), learning_rate=1e-3, device='cuda'):
    losses = []
    batches = batch_data(X_train, batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.MSELoss()
    losses = run_epochs(model, X_train, Y_train, loss_func, optimizer, batches, n_epochs=n_epochs)
    return [i.detach().cpu() for i in losses]

def split_within_solvent(df, train_split=0.8, columns=['Compound Name2']):
    group = df.groupby(columns)
    solvent_indicies = [np.array(i) for i in group.groups.values()]
    dfs = []
    [np.random.shuffle(i) for i in solvent_indicies]
    end_indicies = [int(len(i)*train_split) for i in solvent_indicies]
    train = pd.concat([df.iloc[solvent[0:end_indicies[index]]] for index, 
                       solvent in enumerate(solvent_indicies)]).reset_index(drop=True)
    test = pd.concat([df.iloc[solvent[end_indicies[index]::]] for index, 
                      solvent in enumerate(solvent_indicies)]).reset_index(drop=True)
    return train, test

def split_across_solvents(df, train_split=0.8, columns='Compound Name2'):
    # using df lets get a list of all the unique solvents, and get a list of all the indicies corresponding to each solvent
    solvents = df[columns].unique()
    solvent_indices = [df[df[columns] == solvent].index for solvent in solvents]
    random.shuffle(solvent_indices)
    # add the index groups to form a test set and a train set. Add to the test set first until it is (1-train_split) * len(df) in size. make sure to flatten the list of lists each time
    test_indices = []
    train_indices = []
    for i in range(len(solvent_indices)):
        if len(test_indices) < (1-train_split) * len(df):
            test_indices.extend(solvent_indices[i])
        else:
            train_indices.extend(solvent_indices[i])
    return df.loc[train_indices], df.loc[test_indices]
    
def TLNN_conv(GAMMA, X, change_layers=1):
    new = Conv(layers=GAMMA.layers, layer_size=GAMMA.layer_size,
                out_channels=GAMMA.out_channels,kernel_size=GAMMA.kernel_size, 
                n_inputs=GAMMA.n_inputs, n_outputs=GAMMA.n_outputs)
    test = new(X)
    new.load_state_dict(GAMMA.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    for i in range(change_layers):
        layer = children[total_layers-i-1]
        layer_params = layer.parameters()
        for p in layer_params:
            p.requires_grad = True
    return new

def TLNN(GAMMA, X, change_layers=1):
    new = NN(layers=GAMMA.layers, layer_size=GAMMA.layer_size,
                n_inputs=GAMMA.n_inputs, n_outputs=GAMMA.n_outputs)
    test = new(X)
    new.load_state_dict(GAMMA.state_dict())
    children = [child for child in new.children()]
    for child in children:
        for param in child.parameters():
            param.requires_grad = False
    total_layers = len(children)
    for i in range(change_layers):
        layer = children[total_layers-i-1]
        layer_params = layer.parameters()
        for p in layer_params:
            p.requires_grad = True
    return new

def batch_data(Y, batch_size):
    batch_size = int(batch_size)
    n_observations = int(Y.shape[0])
    batch_index = np.arange(0, n_observations, batch_size)
    #np.random.shuffle(batch_index)
    batches = np.array([np.arange(batch_index[i], batch_index[i+1]) \
                   for i in range(len(batch_index)-1)])
    shape = batches.shape
    temp = batches.reshape(-1,1)
    np.random.shuffle(temp)
    batches = temp.reshape(shape[0], shape[1])
    np.random.shuffle(batches)
    n_batches = len(batches)
    return batches


def cross_validate_Pytorch(X, Y, cv=10, learning_rate=1e-3):
    losses = []
    models = []
    maes = []
    indicies = np.arange(X.shape[0])
    np.random.shuffle(indicies)
    splits = np.array_split(indicies, cv)
    for i in splits:
        ml_model = NN(X.shape[1], Y.shape[1])
        i = np.array(i)
        train = [j for j in indicies if j not in i]
        x_train = X[train, :]
        x_test = X[i, :]
        y_train = Y[train, :]
        y_test = Y[i, :]
        #x_train, x_test, y_train, y_test = train_test_split(X[i, :], Y[i], test_size=1/cv)
        print(x_train.shape, y_train.shape)
        losses.append(run_Pytorch(ml_model, x_train, y_train, n_epochs=600, learning_rate=learning_rate, batch_size=x_train.shape[0]-1))
        pred = ml_model(x_test.to('cuda')).detach().cpu().numpy()
        mae = mean_absolute_error(pred, y_test)
        maes.append(mae)
        models.append(ml_model)
    return losses, models, maes

def get_vars(df, dt=5):
    sig_cols1 = ['sigma_' + str(i) +'_1' for i in range(51)]
    sig_cols2 = ['sigma_' + str(i) +'_2' for i in range(51)]
    keep_cols = sig_cols1
    keep_cols.extend(['Area_1','Volume_1'])
    keep_cols.extend(sig_cols2)
    keep_cols.extend(['Area_2','Volume_2'])
    #keep_cols.extend(['Temperature, K'])
    #df = df.dropna()
    X = df[keep_cols].reset_index(drop=True).to_numpy()
    try:
        temp = df['Temperature, K'].to_numpy().reshape(-1,1)
    except:
        temp = torch.distributions.Uniform(300-dt, 300+dt).sample((1,X.shape[0])).numpy().reshape(-1,1)
        pass
    #print(temp)
    #X = np.hstack([X, temp])
    try:
        Y = np.concatenate(df['ln_gamma1'].to_numpy()).reshape(-1, len(df['ln_gamma1'].iloc[0]))
        Y = Y[:,0]
    except:
        Y = np.log(df['Activity coefficient'].to_numpy().reshape(-1,1))
    return X, Y

def cross_validate_Pytorch_TLNN(model, X, Y, change_layers=3, cv=10, learning_rate=1e-3, batch_size=None, n_epochs=100, device='cuda'):
    print('X, Y : ',X.shape, Y.shape)
    losses = []
    models = []
    maes = []
    if batch_size is None:
        batch_size = int(1/cv*X.shape[0]-1)
    indicies = np.arange(X.shape[0])
    np.random.shuffle(indicies)
    splits = np.array_split(indicies, cv)
    model = model.to(device)
    for i in splits:
        NN = TLNN_conv(model, X[0:2, :], change_layers=change_layers).to(device)
        i = np.array(i)
        train = [j for j in indicies if j not in i]
        #print(train)
        x_train = X[train, :]
        x_test = X[i, :]
        y_train = Y[train, :]
        y_test = Y[i, :]
        #print(y_test.shape)
        losses.append(run_Pytorch(NN, x_train, y_train, 
                                  n_epochs = n_epochs, 
                                  learning_rate = learning_rate, 
                                  batch_size = batch_size,
                                  device=device))
        #print(NN.to('cpu'))
        pred = NN(x_test.to(device)).detach().cpu().numpy()
        mae = mean_absolute_error(pred, y_test)
        maes.append(mae)
        models.append(NN)
    return losses, models, maes

def learning_curve(GAMMA, X, Y, cv=5, learning_rate = 1e-3, change_layers=3, batch_size=4, n_epochs=10, splits=[0.1, 0.25, 0.5, 0.75, 0.9], device='cuda'):
    indicies = np.arange(X.shape[0])
    split_ratios = splits
    cv_losses = []
    cv_models = []
    cv_maes = []
    cv_test_maes = []
    for ratio in split_ratios:
        num = int(ratio*len(indicies))
        num2 = int(split_ratios[-1]*len(indicies))
        x = torch.from_numpy(X[indicies[0:num]].astype(np.float32))
        y = torch.from_numpy(Y[indicies[0:num]].reshape(-1,1).astype(np.float32))
        loss, models, maes = cross_validate_Pytorch_TLNN(GAMMA,
                                                             x, 
                                                             y,
                                                             batch_size=batch_size,
                                                             n_epochs=n_epochs,
                                                             cv=cv,
                                                             learning_rate=learning_rate,
                                                            device=device,
                                                            change_layers=change_layers)
        cv_test_maes.append([mean_absolute_error(np.array(model(torch.from_numpy(X[indicies[num2::],:].astype(np.float32)).to(device)).detach().cpu()),
                                                Y[indicies[num2::]]) for model in models])
        cv_losses.append(loss)
        cv_models.append(models)
        cv_maes.append(maes)
    return split_ratios, cv_losses, cv_models, cv_maes

def split_vars_cv(array, cv=5):
    df = pd.DataFrame(array[:,-2]).reset_index(drop=True)
    #print(len(df))
    group = df.groupby(0)
    solvent_indicies = np.array([np.array(i) for i in group.groups.values()])
    #print(solvent_indicies)
    d = []
    for split in solvent_indicies:
        c = np.array_split(split, cv)
        np.random.shuffle(c)
        d.append(np.array(c))
    cv_list = np.arange(0, cv)
    splits = []
    for i in range(cv):
        e = np.concatenate([d[j][i] for j in range(len(d))])
        splits.append(e)
    return splits
    
def plot_parity(GAMMA, X_train, Y_train, X_valid, Y_valid):
    fig, ax = plt.subplots(dpi=300)
    torch.cuda.empty_cache()
    n = Y_valid.shape[0]
    pred = GAMMA(X_valid).detach().cpu()
    pred_train = GAMMA(X_train).detach().cpu()
    ax.set_aspect(1)
    
    mae = mean_absolute_error(Y_valid, pred)
    r2 = r2_score(pred, Y_valid)
    mae_train = mean_absolute_error(Y_train, pred_train)
    r2_train = r2_score(pred_train, Y_train)


    plt.scatter(Y_valid, pred, edgecolor='k', alpha=0.5, marker='s', label='COSMO-SAC')
    plt.plot(Y_valid, Y_valid,'--k')
    plt.xlabel('COSMO-SAC $ln(\gamma_1^\infty)$', fontsize=18)
    plt.ylabel('NN $ln(\gamma_1^\infty)$', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # testing scores go in the lower right
    text_box1 = AnchoredText('R2 = '+str(r2.round(2))+' \nMAE = '+str(mae.round(3)), frameon=True, loc=4, pad=0.5)
    plt.setp(text_box1.patch, facecolor='white', alpha=0.5)
    # training scores go in the upper left
    text_box2 = AnchoredText('R2 = '+str(r2_train.round(2))+' \nMAE = '+str(mae_train.round(3)), frameon=True, loc=2, pad=0.5)
    plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
    ax.add_artist(text_box1)
    ax.add_artist(text_box2)
    return fig, ax

