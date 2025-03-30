#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:37:06 2024

@author: gez
"""

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
import mendeleev as mv
from matplotlib import pyplot as plt
import matplotlib

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


#This library is used To do the train/test sets split
from sklearn.model_selection import train_test_split

#PCA
from sklearn.decomposition import PCA

#Import Jarvis
from jarvis.ai.descriptors.cfid import get_chem_only_descriptors


#%%

#'Formula'
#'Density'
#'Temperature'
data = joblib.load('CuratedDensity.pkl', )

#%%

#If there is a GPU (there isn't)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
#PARSING OUT DATA AND STUFF
#All keys
all_keys = list(data.keys())

all_keys.remove('Formula')
#all_keys.remove('Density')
Desc = all_keys
El_Desc = Desc[:-1]

#Randomizer of input values
y = data['Density']
X = data[Desc]
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
#X = X[El_Desc]
#X = X[Desc]
#X = X.drop_duplicates().reset_index(drop = True)

Z = pd.DataFrame()

#
for i in range(len(El_Desc)-1):
    MW_vec[i] = (molmass.Formula(Desc[i]).mass)
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


def feat(Z, X, T):
    for i in range(len(El_Desc)-1):
        Z['MW_'+El_Desc[i]] = X[El_Desc[i]]*MW_vec[i]/np.mean(MW_vec)
        Z['X_'+El_Desc[i]] = X[El_Desc[i]]*X_vec[i]/np.mean(X_vec)
        Z['MV_'+El_Desc[i]] = X[El_Desc[i]]*MV_vec[i]/np.mean(MV_vec)
        Z['A_'+El_Desc[i]] = X[El_Desc[i]]*A_vec[i]/np.mean(A_vec)
        Z['G_'+El_Desc[i]] = X[El_Desc[i]]*G_vec[i]/np.mean(G_vec)
        Z['AR_'+El_Desc[i]] = X[El_Desc[i]]*Atom_Rad_vec[i]/np.mean(Atom_Rad_vec)
    Z['Temperature'] = T
    return Z

Z = feat(Z, X, X['Temperature'])

#%%

#The split
X_train, X_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=42)

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

class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, 90)
        self.FC_input2 = nn.Linear(90, 70)
        self.FC_input3 = nn.Linear(70, 40)
        self.FC_mean  = nn.Linear(40, latent_dim)
        self.FC_var   = nn.Linear(40, latent_dim)
        
        self.Softplus = nn.Softplus(3)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.Softplus(self.FC_input(x))
        h_       = self.Softplus(self.FC_input2(h_))
        h_       = self.Softplus(self.FC_input3(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

#%%
	
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, 40)
        self.FC_hidden2 = nn.Linear(40, 70)
        self.FC_hidden3 = nn.Linear(70, 90)
        self.FC_output = nn.Linear(90, output_dim)
        
        self.Softplus = nn.Softplus(3)
        
    def forward(self, x):
        h     = self.Softplus(self.FC_hidden(x))
        h     = self.Softplus(self.FC_hidden2(h))
        h     = self.Softplus(self.FC_hidden3(h))
        
        x_hat = (self.FC_output(h))
        return x_hat

#%%
	
class Predictor(nn.Module):
    def __init__(self, latent_dim):
        super(Predictor, self).__init__()
        self.P_hidden = nn.Linear(latent_dim, 30)
        self.P_hidden2 = nn.Linear(30, 20)
        self.P_hidden3 = nn.Linear(20, 10)
        self.P_output = nn.Linear(10, 1)
        
        self.Softplus = nn.Softplus(3)
        
    def forward(self, x):
        h     = self.Softplus(self.P_hidden(x))
        h     = self.Softplus(self.P_hidden2(h))
        h     = self.Softplus(self.P_hidden3(h))
        pred = (self.P_output(h))
        
        return pred

#%%

class Model(nn.Module):
    def __init__(self, Encoder, Decoder, Predictor):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Predictor = Predictor
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
             
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        pred = self.Predictor(z)
        
        return x_hat, mean, log_var, pred

#Converting thee dataframe into a tensor (change whether your scaling or not)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

#%%
#Encoder and decoder
encoder = Encoder(input_dim=361,  latent_dim=10)
decoder = Decoder(latent_dim=10, output_dim = 361)
predictor = Predictor(latent_dim=10)
#The model itself
model = Model(Encoder=encoder, Decoder=decoder, Predictor=predictor).to(device)

#%%
#The Adam optimizer 
optimizer = optim.Adam(model.parameters(), lr=1e-3, )#weight_decay=1e-6)

#The training part
n_epochs = 10000
batch_size = 400

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    #kl_loss = nn.KLDivLoss(reduction="batchmean")
    KLD      = - (1/batch_size) * 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    #print(KLD)
    return reproduction_loss + KLD

#%%
#This'll be the error list
err_list = []

#This is the training itself
print("Start training VAE...")
model.train()
for epoch in range(n_epochs):
    overall_loss = 0
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size, :]
        ybatch = y_train[i:i+batch_size, :]
        
        x_hat, mean, log_var, pred = model.forward(Xbatch)
        
        #HOW TO WRITE A TERM FOR A LOSS WITH NORMALIZATION? DETACH Y_PRED
        rowsum = torch.sum(x_hat[:,:60], 1)
        sum_loss = np.sum(((rowsum-1).detach().numpy())**2)
        
        #Physics constrained stuff
        prop_loss_1 = (x_hat[:,-1] - torch.sum((x_hat[:,:60] * torch.from_numpy(MW_vec)), axis=1)/torch.mean(torch.from_numpy(MW_vec)))**2


        #Loss of the prediction model
        pred_loss = nn.functional.mse_loss(pred, ybatch, reduction='sum')
        
        loss = loss_function(Xbatch, x_hat, mean, log_var) + pred_loss
        overall_loss += loss.item() #+ sum_loss
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
    err_list.append(overall_loss)
    print(f'Finished epoch {epoch}, latest loss {overall_loss/batch_size}')
    
print("Finish!!")

#%%

#Plot the error across each batch
plt.figure()
plt.plot(err_list)
plt.yscale('log')
plt.grid()
plt.show()

#%%
import matplotlib as mpl
#rowsum = torch.sum(y_pred[:,:-1], 1)
#Reduce the latent space to only two dimensions
X_new = encoder(X_train)  
Reparam = model.reparameterization(X_new[0], torch.exp(0.5*X_new[1])).detach().numpy()

pca = PCA(n_components=3).fit(Reparam)
print(pca.explained_variance_ratio_)
X_dec= pca.transform(Reparam)

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

"""
The probabilistic nature of the SVAE means that it is impossible to get the 
exact same image for the latent space every time the code is run.
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('PCA of Molten Salt Density Dataset')
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = X_dec[:,0]
    ys = X_dec[:,1]
    zs = X_dec[:,2]
    cs = y_train
    p = ax.scatter(xs, ys, zs, c=cs, alpha = 0.3, norm=mpl.colors.Normalize(0, 6),)#cmap = mpl.cm.cool,
    
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
fig.colorbar(p, label='Density [g/cm3]', extend = 'max', ticks = [0, 1, 2, 3, 4, 5, 6])
plt.show()


#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.figure()
plt.hist(y_train.detach().numpy())
plt.title('Histogram for Density values in the Training Dataset')
plt.xlabel(r'Molten Salt Density $[g/cm^{3}]$')
plt.ylabel(r'Frequency of Values in the Training Set')
plt.grid()
plt.show()

#%%
#Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

font = {'weight' : 'normal',
        'size'   : 18}
mpl.rc('font', **font)


#print(rowsum)
#matplotlib.pyplot.hist(y_train.detach().numpy())

#Plotting the parity plot (TEST)
pred_test = model(X_test)[-1].detach().numpy()
y_plot = y_test.detach().numpy()

#Plotting the parity plot (TRAIN)
pred_train = model(X_train)[-1].detach().numpy()
y_prain = y_train.detach().numpy()

plt.figure()
x=np.linspace(np.min([np.min(y_plot),np.min(pred_test)]),np.max([np.max(y_plot),np.max(pred_test)]),10)
plt.scatter(pred_test, y_plot, alpha = 0.2, label = 'Test Set')
#plt.scatter(pred_train, y_prain, alpha = 0.2)

plt.plot(x,x,'k')
#plt.title('Parity plot for predictive component of architecture')
plt.xlabel(r'Reference Density $\mathrm{[g/cm^{3}]}$')
#plt.xlabel(r'Reference density')
plt.ylabel(r'Predicted Density $\mathrm{[g/cm^{3}]}$')
#plt.ylabel(r'Predicted density []')
plt.legend()
plt.grid()
plt.show()

#Plot the metrics
print('For this training batch, the metrics are; ')
print('The r2 score is as follows; ', r2_score(pred_test, y_plot))
print('The MAE score is as follows; ', mean_absolute_error(pred_test, y_plot))
print('The MAPE score is as follows; ', 100*mean_absolute_percentage_error(pred_test, y_plot))

#%%

torch.save(model, 'Complete_Model.pkl')
torch.save(encoder, 'Encoder.pkl')
torch.save(decoder, 'Decoder.pkl')
torch.save(predictor, 'Predictor.pkl')

#Saving the model in question
#torch.save(pre, 'Pred_Model.pkl')
#torch.save(dec, 'Dec_Model.pkl')
# Latest: [0.05979645 0.03984734 0.02711511]
#model = torch.load('VAE_Model_With_Descriptors.pkl')
#torch.save(model, 'VAE_Model_With_Descriptors.pkl')

#%%
#The following line imports the "properties to molar fractions" network
#that is trained using the code "Julian Barra - Descriptors to Composition Network"
dec = joblib.load('Dec_Model.pkl' )

#%%
comp_dp = []
MF_sum = []


#%%

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

font = {'weight' : 'normal',
        'size'   : 16}
mpl.rc('font', **font)

plt.figure()
plt.hist(MF_sum, bins = 50)
plt.title('Histogram for Cumulative Molar Fractions of Generated Data Points')
plt.xlabel(r'Sum of all Molar Fractions $[-]$', )
plt.ylabel(r'Frequency', )
plt.grid()
plt.show()

#%%
#PCA
#Sampler from the latent space (High Density)
High_d_sample = pca.inverse_transform([-2, 1, 0])
High_d_pred = predictor(torch.tensor(High_d_sample, dtype=torch.float32))
print('Density predicted for original High density sample (between 5 ad 7): ', High_d_pred)
High_d_sample = decoder(torch.tensor(High_d_sample, dtype=torch.float32))
T_High = High_d_sample[-1].detach().numpy()
High_d_sample = dec(High_d_sample[:-1])
print('Molar Fractions add up to: ', torch.sum(High_d_sample))
High_d_sample = High_d_sample.detach().numpy()
print('Temperature for the HD point is: ', (T_High)*2190 + 260)

#Sampler from the latent space (Low Density)
Low_d_sample = pca.inverse_transform([0, 1, 0])
Low_d_pred = predictor(torch.tensor(Low_d_sample, dtype=torch.float32))
print('Density predicted for original Low density sample (between 0 and 1): ', Low_d_pred)
Low_d_sample = decoder(torch.tensor(Low_d_sample, dtype=torch.float32))
T_Low = Low_d_sample[-1].detach().numpy()
Low_d_sample = dec(Low_d_sample[:-1])
print('Molar Fractions add up to: ', torch.sum(Low_d_sample))
Low_d_sample = Low_d_sample.detach().numpy()
print('Temperature for the LD point is: ', (T_Low)*2190 + 260)


#Sampler from the latent space (Mid Density)
Mid_d_sample = pca.inverse_transform([2, 1, 0])
Mid_d_pred = predictor(torch.tensor(Mid_d_sample, dtype=torch.float32))
print('Density predicted for original Mid density sample (between 2 and 3): ', Mid_d_pred)
Mid_d_sample = decoder(torch.tensor(Mid_d_sample, dtype=torch.float32))
T_Mid = Mid_d_sample[-1].detach().numpy()
Mid_d_sample = dec(Mid_d_sample[:-1])
print('Molar Fractions add up to: ', torch.sum(Mid_d_sample))
Mid_d_sample = Mid_d_sample.detach().numpy()
print('Temperature for the MD point is: ', (T_Mid)*2190 + 260)




#%%

#Filtering according to said criteria
Filt_High = np.maximum(High_d_sample, 0) * np.equal(0.01, np.minimum(np.maximum(High_d_sample, 0), 0.01))
Filt_Low = np.maximum(Low_d_sample, 0) * np.equal(0.01, np.minimum(np.maximum(Low_d_sample, 0), 0.01))
Filt_Mid = np.maximum(Mid_d_sample, 0) * np.equal(0.01, np.minimum(np.maximum(Mid_d_sample, 0), 0.01))

#%%
#The probabilistic nature of the SVAE makes it impossible to generate the 
#exact same mixtures every time the model is trained and sampled. Instead,
#the sampled compositions are provided in this code block. Uncommenting them
#and passing them through the SVAE show they correspond to the values of low,
#medium and high density

"""
#FIRST HIGH DENSITY
Filt_High = np.zeros(60)
Filt_High[27] = 0.02     #K
Filt_High[35] = 0.02     #Zn
Filt_High[45] = 0.06     #Br
Filt_High[52] = 0.45    #I
Filt_High[56] = 0.45      #Tl
"""

"""
#SECOND HIGH DENSITY
Filt_High = np.zeros(60)
Filt_High[17] = 0.2      #Gd
Filt_High[14] = 0.04  #Cs
Filt_High[33] = 0.04   #Sr
Filt_High[58] = 0.72   #I
"""


"""
#FIRST LOW DENSITY
#Manual modification smol approximation Low
Filt_Low = np.zeros(60)
Filt_Low[1] = 0.03      #Be
Filt_Low[2] = 0.56      #Cl
Filt_Low[8] = 0.35      #Li
Filt_Low[15] = 0.03    #Al
Filt_Low[36] = 0.03     #Mg
"""

"""

#SECOND LOW DENSITY
Filt_Low = np.zeros(60)
Filt_Low[36] = 0.04      #Be
Filt_Low[52] = 0.1     #Cl
Filt_Low[11] = 0.48  #F
Filt_Low[56] = 0.06#Al
Filt_Low[55] = 0.32   #K
"""


"""
#FIRST MID DENSITY
#Manual modification Mid
Filt_Mid = np.zeros(60)
Filt_Mid[1] = 0.04      #Be
Filt_Mid[6] = 0.58      #F
Filt_Mid[15] = 0.05     #Al
Filt_Mid[18] = 0.31     #Na
Filt_Mid[36] = 0.02     #K
"""

"""
#SECOND MID DENSITY
Filt_Mid = np.zeros(60)
Filt_Mid[52] = 0.55      #Cl
Filt_Mid[11] = 0.16 #F
Filt_Mid[55] = 0.15   #K
Filt_Mid[18] = 0.14   #Th
"""

#%%

Filt_High = Filt_High/np.sum(Filt_High)
Filt_Low = Filt_Low/np.sum(Filt_Low)
Filt_Mid = Filt_Mid/np.sum(Filt_Mid)

print('Number of elements in the "clean" High Density point: ', np.count_nonzero(Filt_High))
print('Number of elements in the "clean" Low Density point: ', np.count_nonzero(Filt_Low))
print('Number of elements in the "clean" Mid Density point: ', np.count_nonzero(Filt_Mid))

#%%
#Reconvert into 
def feat2(Z, x, T):
    for i in range(len(El_Desc)-1):
        #Uncomment some day
        Z.at[0, 'MW_'+El_Desc[i]] = x[i]*MW_vec[i]/np.mean(MW_vec)
        Z.at[0, 'X_'+El_Desc[i]] = x[i]*X_vec[i]/np.mean(X_vec)
        Z.at[0, 'MV_'+El_Desc[i]] = x[i]*MV_vec[i]/np.mean(MV_vec)
        Z.at[0, 'A_'+El_Desc[i]] = x[i]*A_vec[i]/np.mean(A_vec)
        
        Z.at[0, 'G_'+El_Desc[i]] = x[i]*G_vec[i]/np.mean(G_vec)
        Z.at[0, 'AR_'+El_Desc[i]] = x[i]*Atom_Rad_vec[i]/np.mean(Atom_Rad_vec)
    Z['Temperature'] = T
    return Z

#Create the three dataframes
Hold_High = pd.DataFrame()
Hold_Low = pd.DataFrame()
Hold_Mid = pd.DataFrame()

Rec_High = torch.tensor(feat2(Hold_High, Filt_High, T_High).values, dtype=torch.float32)
Rec_Low = torch.tensor(feat2(Hold_Low, Filt_Low, T_Low).values, dtype=torch.float32)
Rec_Mid = torch.tensor(feat2(Hold_Mid, Filt_Mid, T_Mid).values, dtype=torch.float32)


#%%
#Let's evaluate the "clean" data points
print('The prediction for the "cleaned-up" High density point: ', model(Rec_High)[-1])
print('The prediction for the "cleaned-up" Low density point: ', model(Rec_Low)[-1])
print('The prediction for the "cleaned-up" Mid density point: ', model(Rec_Mid)[-1])



#%%
chrg_list = []
nmbr_chrg = []
pos_vec = Filt_High != 0

for i in range(len(El_Desc[:-1])):
    chrg_list.append(mv.element(El_Desc[i]).oxistates)
    if pos_vec[i]:
        nmbr_chrg +=mv.element(El_Desc[i]).oxistates
    else:
        pass
   
#From now on, for the different compositions    
#pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
import itertools
def ChargePicker(comp_arr):
    chrg_arr = list(np.zeros(len(El_Desc[:-1])))
    for i in range(len(El_Desc[:-1])):
        if np.equal(0.01, np.minimum(np.maximum(comp_arr, 0), 0.01))[i]:
            chrg_arr[i] = chrg_list[i]
        else:
            chrg_arr[i] = [0]
    #
    chrg_sum = []
    #
    for l in list(itertools.product(*chrg_arr)):
        chrg_sum.append(np.sum(comp_arr*l)) 
    min_chrg = np.min(np.abs(chrg_sum))
    return chrg_arr, chrg_sum, min_chrg

#
High_res = ChargePicker(Filt_High)
Low_res = ChargePicker(Filt_Low)
Mid_res = ChargePicker(Filt_Mid)
print('Residual of charge for High Density point: ', High_res[2])
print('Residual of charge for Low Density point: ', Low_res[2])
print('Residual of charge for Mid Density point: ', Mid_res[2])


#%%
#Similarity to existing systems?
data_els = data[El_Desc[:]]
data_els = data_els.drop_duplicates()
data_els = data_els.reset_index(drop=True)

diff_error = []



#%%
#from scipy.optimize import minimize
#res=minimize(Funcion_costo,[N_helio_0,n_globos_0],bounds=((0.5, None), (2, None)), method='SLSQP')
import joblib
from scipy.optimize import nnls
data_linear = joblib.load('CuratedDensity_LinearDatapoints.pkl', )
linear_generators = joblib.load('LinearCuratedDensityGenerators.pkl', )
data_linear = data_linear[El_Desc[:-1]]
data_linear = data_linear.drop_duplicates()
data_linear = data_linear.reset_index(drop=True)

#data_els = data_els.values
least_High = nnls(data_linear.T, np.transpose(Filt_High))
leleast_High = least_High[0]
print('L2 norm of residual of linear composition combination (High density): ', least_High[1])
print('Number of added rows (High density): ', np.count_nonzero(least_High[0]))
print('')

#data_els = data_els.values
least_Low = nnls(data_linear.T, np.transpose(Filt_Low))
leleast_Low = least_Low[0]
print('L2 norm of residual of linear composition combination (Low density): ', least_Low[1])
print('Number of added rows (Low density): ', np.count_nonzero(least_Low[0]))
print('')

#data_els = data_els.values
least_Mid = nnls(data_linear.T, np.transpose(Filt_Mid))
leleast_Mid = least_Mid[0]
print('L2 norm of residual of linear composition combination (Mid density): ', least_Mid[1])
print('Number of added rows (Mid density): ', np.count_nonzero(least_Mid[0]))
print('')


#%%
#AAAAAAA
#LET'S TRY TO DO EVALUATIONS AT 900 K ALL
Rec_High_const = torch.tensor(feat2(Hold_High, Filt_High, (900 - 260)/2190).values, dtype=torch.float32)
Rec_Low_const = torch.tensor(feat2(Hold_Low, Filt_Low, (900 - 260)/2190).values, dtype=torch.float32)
Rec_Mid_const = torch.tensor(feat2(Hold_Mid, Filt_Mid, (900 - 260)/2190).values, dtype=torch.float32)

#%%
#Let's evaluate the "clean" data points
print('The prediction for the "cleaned-up" High density point at 900 K: ', model(Rec_High_const)[-1])
print('The prediction for the "cleaned-up" Low density point at 900 K: ', model(Rec_Low_const)[-1])
print('The prediction for the "cleaned-up" Mid density point at 900 K: ', model(Rec_Mid_const)[-1])

#%%
#Try to do the experimental validation of density
#HIGH
Exp_dens_High = 0
for i in np.nonzero(leleast_High)[0]:
    Coef_1 = float(linear_generators.iloc[i]['Data 1'])
    Coef_2 = float(linear_generators.iloc[i]['Data 2'])
    Exp_dens_High += (leleast_High[i])*(Coef_1 - 900*np.sign(Coef_2)*Coef_2)
    #print(linear_generators.iloc[i]['Data 1'])
    #print(linear_generators.iloc[i]['Data 2'])
    pass
print('High density point as linear sum of database information at 900 K: ', np.sum(Exp_dens_High))

#LOW
Exp_dens_Low = 0
for i in np.nonzero(leleast_Low)[0]:
    Coef_1 = float(linear_generators.iloc[i]['Data 1'])
    Coef_2 = float(linear_generators.iloc[i]['Data 2'])
    Exp_dens_Low += (leleast_Low[i])*(Coef_1 - 900*np.sign(Coef_2)*Coef_2)
    #print(linear_generators.iloc[i]['Data 1'])
    #print(linear_generators.iloc[i]['Data 2'])
    pass
print('Low density point as linear sum of database information at 900 K: ', np.sum(Exp_dens_Low))

#Mid
Exp_dens_Mid = 0
for i in np.nonzero(leleast_Mid)[0]:
    Coef_1 = float(linear_generators.iloc[i]['Data 1'])
    Coef_2 = float(linear_generators.iloc[i]['Data 2'])
    Exp_dens_Mid += (leleast_Mid[i])*(Coef_1 - 900*np.sign(Coef_2)*Coef_2)
    #print(linear_generators.iloc[i]['Data 1'])
    #print(linear_generators.iloc[i]['Data 2'])
    pass
print('Mid density point as linear sum of database information at 900 K: ', np.sum(Exp_dens_Mid))

