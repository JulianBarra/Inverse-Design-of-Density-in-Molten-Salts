#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

#%%

#This is the general use libraries. The first two are generally used in data analysis. The third one is used to plot data
#!jupyter nbextension enable --py widgetsnbextension
import joblib
import molmass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#This library is used To do the train/test sets split
from sklearn.model_selection import train_test_split

#Import Jarvis
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors


#%%

#Loading the dataset
data = joblib.load('CuratedDensity.pkl', )

#%%

#If there is a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
#PARSING OUT DATA AND STUFF
#All keys
all_keys = list(data.keys())

all_keys.remove('Formula')
#all_keys.remove('Density')
Desc = all_keys
El_Desc = Desc[:-2]

#Randomizer of input values
y = data['Density']
X = data[El_Desc]
#Let's try something
MW_vec = np.ones(60)
MV_vec = np.ones(60)
X_vec = np.ones(60)
A_vec = np.ones(60)
G_vec = np.ones(60)
Atom_Rad_vec = np.ones(60)
Z_vec = np.ones(60)
V_vec = np.ones(60)



#Eliminate the temperature
X = X.drop_duplicates().reset_index(drop = True)

Z = pd.DataFrame()

#
for i in range(len(El_Desc)):
    MW_vec[i] = (molmass.Formula(Desc[i]).mass)#/(get_chem_only_descriptors(Desc[i])[0][349])
    X_vec[i] = get_chem_only_descriptors(Desc[i])[0][247]
    MV_vec[i] = get_chem_only_descriptors(Desc[i])[0][349]
    A_vec[i] = get_chem_only_descriptors(Desc[i])[0][94]
    G_vec[i] = get_chem_only_descriptors(Desc[i])[0][888]
    Atom_Rad_vec[i] = get_chem_only_descriptors(Desc[i])[0][108]
    Z_vec[i] = get_chem_only_descriptors(Desc[i])[0][398]
    V_vec[i] = get_chem_only_descriptors(Desc[i])[0][152]

X_vec[46] = 1.28
MV_vec[46] = 12.29
A_vec[46] = 24.5
G_vec[46] = 65
Atom_Rad_vec[46] = 2.43
Z_vec[46] = 94
V_vec[46] = 8



for i in range(len(El_Desc)):
    Z['MW_'+El_Desc[i]] = X[El_Desc[i]]*MW_vec[i]/np.mean(MW_vec)
    Z['X_'+El_Desc[i]] = X[El_Desc[i]]*X_vec[i]/np.mean(X_vec)
    Z['MV_'+El_Desc[i]] = X[El_Desc[i]]*MV_vec[i]/np.mean(MV_vec)
    Z['A_'+El_Desc[i]] = X[El_Desc[i]]*A_vec[i]/np.mean(A_vec)
    Z['G_'+El_Desc[i]] = X[El_Desc[i]]*G_vec[i]/np.mean(G_vec)
    Z['AR_'+El_Desc[i]] = X[El_Desc[i]]*Atom_Rad_vec[i]/np.mean(Atom_Rad_vec)


#%%

#The split
X_train_recon, X_test_recon, y_train_recon, y_test_recon = train_test_split(Z, X, test_size=0.2, random_state=42)

#%%

class Deconstructor(nn.Module):
    def __init__(self):
        super(Deconstructor, self).__init__()
        self.P_hidden = nn.Linear(360, 250)
        self.P_hidden2 = nn.Linear(250, 180)
        self.P_hidden3 = nn.Linear(180, 100)
        self.P_output = nn.Linear(100, 60)
        
        self.Softplus = nn.Softplus(3)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h     = self.Softplus(self.P_hidden(x))
        h     = self.tanh(self.P_hidden2(h))
        h     = self.tanh(self.P_hidden3(h))
        pred = (self.P_output(h))
        
        return pred
    
#%%

#Converting thee dataframe into a tensor (change whether your scaling or not)
X_train_recon = torch.tensor(X_train_recon.values, dtype=torch.float32)
X_test_recon = torch.tensor(X_test_recon.values, dtype=torch.float32)
y_train_recon = torch.tensor(y_train_recon.values, dtype=torch.float32)
y_test_recon = torch.tensor(y_test_recon.values, dtype=torch.float32)

#%%
dec= Deconstructor()

#The Adam optimizer 
optimizer = optim.Adam(dec.parameters(), lr=1e-5,)

#The training part
n_epochs = 16000
batch_size = 400


#%%

#This'll be the error list
err_list = []

#This is the training itself
print("Start training VAE...")
dec.train()
for epoch in range(n_epochs):
    overall_loss = 0
    overall_perloss = 0
    for i in range(0, len(X_train_recon), batch_size):
        Xbatch = X_train_recon[i:i+batch_size, :]
        ybatch = y_train_recon[i:i+batch_size, :]
        
        x_hat = dec.forward(Xbatch)
        
        #Loss of the prediction model
        reproduction_loss = nn.functional.mse_loss(x_hat, ybatch, reduction='mean')        
        
        loss = reproduction_loss
        overall_loss += loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
    err_list.append(overall_loss)
    print(f'Finished epoch {epoch}, latest loss {overall_loss/batch_size}')
    
print("Finish!!")

#%%

#Evaluate using the testing
x_hatest = dec.forward(X_test_recon)


#Plot the error across each batch
plt.figure()
plt.plot(err_list)
plt.yscale('log')
plt.grid()
plt.show()


#%%
import joblib

#Saving the model in question
#joblib.dump(dec, 'Dec_Model.pkl')