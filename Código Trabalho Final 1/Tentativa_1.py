#%% Importação

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import rice
from scipy.stats import rayleigh
import random


#%%  Funções 

# Rayleigh (NLos)
def ph_g_rayleigh(h, Ph=1):
    g = h**2    
    Ph_Rayleigh = ((1)/(Ph))*(np.exp((-g)/(Ph)))
    return Ph_Rayleigh
#print(ph_g_rayleigh(0.3))

# Rice (Los)
def ph_g_rice(h, K):
    g = h**2
    Ph_rice = 2*(np.sqrt(g))*(K+1)*np.i0(0.)*np.sqrt(K(K+1))*np.exp(-K-(K+1)*g)
    return Ph_rice

def PLd(d, alpha):
    PLdj = C0lin*(d/d_0)**(-alpha)
    PLdl = np.sqrt(PLdj)
    return PLdj

#%% Parâmetros

N=10
M=40
Pt = 1
kappa_G=1.5
kappa_d=0.5
kappa_k=3.0
kappa_Bt1=0.5
alpha_G=2.0
alfa_d=2.8
alfa_k=2.8
kappa_Bt2=2.0
lambda_k=5      #bps/Hz
d_0=1           #m
C0dB=-30        #dBm
C0lin=10**((C0dB+30)/10)
sigma2dB=-80    #dBm
sigma2lin=10**((sigma2dB+30)/10)
dG=70          #m
dv=2           #m
dk=20          #m


#%% Variáveis Aleatórias com métodos de importação:

# var_rayleigh = rayleigh(0.0)
# var_rice = rice(0.0)

#%% Distancias

# dG = Horizontal Distance entre BS e STAR_RIS
# dv = Vertical Distance
dr = random.random() # Horizontal Distance entre UEr e STAR_RIS (Não tem valor)
dt = random.random() # Horizontal Distance entre STAR_RIS e UEt (Não tem valor)

dSt = np.sqrt(dt**2 + dv**2)                    # d Star_RIS & UEt
dSr = np.sqrt(dr**2 + dv**2)                    # d STAR_RIS & UEr
dBr = np.sqrt((dG-dr)**2 + dv**2)*10**-2        # d BS & UEr
dBt = np.sqrt((dG+dt)**2 + dv**2)*10**-2        # d BS & UEt

#print(dBr,dBt)

# print(dSt, dSr, dBr, dBt)

# %% Fading

i=10
hLoS_Br = np.zeros(i)
hLoS_Bt = np.array(i)

for i in range(i):

    dr = random.random()        # Horizontal Distance entre UEr e STAR_RIS (Não tem valor)
    dt = random.random()        # Horizontal Distance entre STAR_RIS e UEt (Não tem valor)
    dSt = np.sqrt(dt**2 + dv**2)                    # d Star_RIS & UEt
    dSr = np.sqrt(dr**2 + dv**2)                    # d STAR_RIS & UEr
    dBr = np.sqrt((dG-dr)**2 + dv**2)*10**-2        # d BS & UEr
    dBt = np.sqrt((dG+dt)**2 + dv**2)*10**-2        # d BS & UEt

    
    h1_Br= ph_g_rayleigh(dBr,1)
#np.append(hLoS_Br,(PLd(dG,alpha_G)*h1_Br))
    hLoS_Br = (PLd(dG,alpha_G)*h1_Br)

    h2_UEt = ph_g_rayleigh(dBt,1) + ph_g_rayleigh(dBt,1)
#np.append((hLoS_Bt),(PLd(dBt, alfa_d)*h2_UEt))
    hLoS_Bt_result = (PLd(dBt, alfa_d)*h2_UEt)
    print(hLoS_Bt_result)
    np.insert(hLoS_Bt,i,hLoS_Bt_result)
    

wBr = np.sqrt(Pt/N)*((hLoS_Br)/(np.conjugate(hLoS_Br)))
wBt = np.sqrt(Pt/N)*((hLoS_Bt)/(np.conjugate(hLoS_Bt)))

print([hLoS_Bt])


#%%
"""
Sistema com 2Rx (um com Rayleigh (h1) e outro com Rice (h2))

V 1) Implementar o fading (Rayleigh e Rice)
V 2) Implementar PL(dj) para jogar dentro do fading total
3) Implementar peso de velocidade de inércia (w)
4) Rodar o código com as variáveis aleatórias N vezes;
5) Implementar R_OMA e tirar a média das N iterações;


OBS: 
a) Rayleigh é a soma de duas variáveis gaussianas;
b) w é um vetor utilizado da BS até a STAR_RIS 
   (Sinal incidente (Ganho));
c) Para o canal Rice pode-se usar a eq que está no artigo (?)
d) Gerar hLoS (Gaussiana) e hNLoS (soma de 2 gaussiana);






"""


