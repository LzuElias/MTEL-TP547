#%% Importação

import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import rice
from scipy.stats import rayleigh
from scipy.special import i0
import random


#%%  Funções 

# Rayleigh (NLos)

# def ph_g_rayleigh():
#     var_gauss = np.random.randn    
#     return var_gauss

def ph_g_rayleigh(h, Ph=1):
    g = h**2    
    Ph_Rayleigh = ((1)/(Ph))*(np.exp((-g)/(Ph)))
    return Ph_Rayleigh


def ph_g_rice():
    x = np.random.rand()
    y = np.random.rand()
    var_gauss = complex(x,y)
    return var_gauss




# # Rice (Los)
# def ph_g_rice(h, K=2):
#     g = h**2
#     Ph_rice = 2*(np.sqrt(g))*(K+1)*i0(0.)*np.sqrt(K(K+1))*np.exp(-K-(K+1)*g)
#     return Ph_rice

def PLd(d, alpha):
    PLdj = C0lin*(d/d_0)**(-alpha)
    PLdl = np.sqrt(PLdj)
    return PLdl

def R_OMA(Ganho):
    R_OMA = (1/2)*np.log2(1+((Pt*Ganho)/((sigma2lin)**2)))
    return R_OMA

#%% Parâmetros


N=10                   # Número de Antenas
num_antennas = np.arange(0,N,1)
M=40                    
Pt = 1                  # Pot. Transmissão
kappa_G=1.5
kappa_d=0.5
kappa_k=3.0
kappa_Br=0.5
alpha_G=2.0
alpha_d=2.8
alpha_k=2.8
kappa_Bt=2.0
lambda_k=5      #bps/Hz
d_0=1           #m
C0dB=-30        #dBm
C0lin=10**((C0dB+30)/10)
sigma2dB=-80    #dBm
sigma2lin=10**((sigma2dB+30)/10)
dG=70          #m
dv=2           #m
dk=20          #m
dr = 60        #m
dt = 10        #m

# %% Fading

j=1
k=1
ii = 1000
vetor_R_OMA_Br = np.array([])
vetor_R_OMA_Bt = np.array([])
vetor_media_R_OMA_Br = np.array([])
vetor_media_R_OMA_Bt = np.array([])

for i in range (N):
    for ii in range(ii):

        #dr = random.random()                                # Horizontal Distance entre UEr e STAR_RIS (Não tem valor)
        #dt = random.random()                                # Horizontal Distance entre STAR_RIS e UEt (Não tem valor)
        dSt = np.sqrt(dt**2 + dv**2)                        # d Star_RIS & UEt
        dSr = np.sqrt(dr**2 + dv**2)                        # d STAR_RIS & UEr
        dBr = np.sqrt((dG-dr)**2 + dv**2)*10**-2            # d BS & UEr Normalizado
        dBt = np.sqrt((dG+dt)**2 + dv**2)*10**-2            # d BS & UEt Normalizado

        # Fading BS-UEr (Rice usando soma das var. gaussianas)
        h1_Br = ph_g_rice()
        h2_Br = np.sqrt((kappa_Br)/(1+kappa_Br))*h1_Br
        hLoS_Br_result = (PLd(dBr,alpha_k)*h2_Br)           # Path_Los(d) * h
        # Ganho BS->UEr (percurso hd)
        wr = np.sqrt(Pt/(j))*((hLoS_Br_result)/(np.conjugate(hLoS_Br_result))) # w = inertia velocity weight
        j=j+1
        Ganho_Br = (hLoS_Br_result*wr)**2                   # Ganho = (h*w)²
        # R_OMA (sem STAR_RIS)
        R_OMA_Br_Result = R_OMA(Ganho_Br)                   # Resultado da eq de R_OMA definida na função 
        vetor_R_OMA_Br = np.insert(vetor_R_OMA_Br, 0,R_OMA_Br_Result)   # Alocação do resultado em um vetor

        #Fading BS - UEt (Rayleigh)
        h1_Bt = ph_g_rayleigh(dBt,1)                        # Fading hLos
        h2_Bt = np.sqrt((kappa_Bt)/(1+kappa_Bt))*h1_Bt      # raiz quadrada média de kappa * hLos
        hLoS_Bt_result = (PLd(dBt, alpha_k)*h2_Bt)          # Path_Los(d) * h  
        # Ganho BS->UEt (percurso G+ht, mas sem a STAR_RIS)
        wt = np.sqrt(Pt/(k))*((hLoS_Bt_result)/(np.conjugate(hLoS_Bt_result)))
        k=k+1
        Ganho_Bt = (hLoS_Bt_result*wt)**2                      # Ganho = h²
        # R_OMA (sem STAR_RIS)
        R_OMA_Bt_Result = R_OMA(Ganho_Bt)                   # Resultado da eq de R_OMA definida na função
        vetor_R_OMA_Bt = np.insert(vetor_R_OMA_Bt, 0,R_OMA_Bt_Result)   # Alocação do resultado em um vetor


        ## Fading BS->UEr (percurso hd) (Rayleigh)
        # h1_Br= ph_g_rayleigh(dBr,1)                         # Fading hLos
        # h2_Br = np.sqrt((kappa_Br)/(1+kappa_Br))*h1_Br      # raiz quadrada média de kappa * hLos
        # hLoS_Br_result = (PLd(dBr,alpha_k)*h2_Br)           # Path_Los(d) * h
        # # Ganho BS->UEr (percurso hd)
        # Ganho_Br = (hLoS_Br_result)**2                      # Ganho = h²
        # # R_OMA (sem STAR_RIS)
        # R_OMA_Br_Result = R_OMA(Ganho_Br)                   # Resultado da eq de R_OMA definida na função
        # R_OMA_Br = np.insert(R_OMA_Br, 0,R_OMA_Br_Result)     # Alocação do resultado em um vetor

    # Médias para Br e Bt
    media_R_OMA_Br = np.mean(vetor_R_OMA_Br)
    vetor_media_R_OMA_Br = np.insert(vetor_media_R_OMA_Br, 0, media_R_OMA_Br)

    media_R_OMA_Bt = np.mean(vetor_R_OMA_Bt)
    vetor_media_R_OMA_Bt = np.insert(vetor_media_R_OMA_Bt, 0, media_R_OMA_Bt)

media_da_media_Br = np.mean(vetor_media_R_OMA_Br)
media_da_media_Bt = np.mean(vetor_media_R_OMA_Bt)

print(media_da_media_Br)
print(f"\n {media_da_media_Bt}")


# %% Plotagem

plt.figure(1)
plt.plot(num_antennas, vetor_media_R_OMA_Br, '-bo', label = "BS to User 'r' without STAR-RIS and Rician fading (OMA)")
# plt.scatter(num_antennas, vetor_media_R_OMA_Br)
plt.plot(num_antennas, vetor_media_R_OMA_Bt, '-rx',label = "BS to User 't' without STAR-RIS and Rayleigh fading (OMA)")
# plt.scatter(num_antennas, vetor_media_R_OMA_Bt)
plt.legend(loc = "center left")
plt.xlabel("Number of Antennas at the BS (N)")
plt.ylabel("Achievable Sum Rate(bps/Hz)")
plt.grid(True)
plt.show()





#%%
"""
Sistema com 2Rx (um com Rayleigh (h1) e outro com Rice (h2))

V 1) Implementar o fading (Rayleigh e Rice)
V 2) Implementar PL(dj) para jogar dentro do fading total
3) Implementar peso de velocidade de inércia (w)
4) Rodar o código com as variáveis aleatórias N vezes;
5) Implementar R_OMA e tirar a média das N iterações;
6) Plotar ()


OBS: 
a) Rayleigh é a soma de duas variáveis gaussianas;
b) w é um vetor utilizado da BS até a STAR_RIS 
   (Sinal incidente (Ganho));
c) Para o canal Rice pode-se usar a eq que está no artigo (?)
d) Gerar hLoS (Gaussiana) e hNLoS (soma de 2 gaussiana);






"""


