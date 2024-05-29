# import

import numpy as np
import matplotlib as plt

# %% Funções

# Canal h
def canal_h(alfa, theta):
    h = alfa * np.exp((1j*theta))
    return h

# Distancia Euclidiana Quadrática
def dist_euclid_quad (x, y):
    d_2 = (x-y)*((np.conjugate(x))-(np.conjugate(y)))
    return d_2

# Rayleigh Fading
def ph_g_rayleigh(h, Ph=1):
    g = h**2    
    Ph_Rayleigh = ((1)/(Ph))*(np.exp((-g)/(Ph)))
    return Ph_Rayleigh


#%% Parâmetros

t = np.linspace(0,30,2000)
f1 = 30
f2 = 80

# Sinal
signal_0 = np.exp(1j *2*np.pi*f1*t)
signal_1 = np.exp(1j *2*np.pi*f2*t)
signal_0_conj = np.conjugate(signal_0)
signal_1_conj = np.conjugate(signal_1)

# Módulo
alfa_0 = 1
alfa_1 = 0.5
alfa_2 = 2
alfa_3 = 0.8

# Fase
theta_0 = 0
theta_1 = 60
theta_2 = 80
theta_3 = 135

# Canais
h_0 = canal_h(alfa_0, theta_0)
h_0_conj = np.conjugate(h_0)
h_1 = canal_h(alfa_1, theta_1)
h_1_conj = np.conjugate(h_1)
h_2 = canal_h(alfa_2, theta_2)
h_2_conj = np.conjugate(h_2)
h_3 = canal_h(alfa_3, theta_3)
h_3_conj = np.conjugate(h_3)

# Ruído
n_0 = np.random.rand() + 1j*np.random.rand()
n_2 = np.random.rand() + 1j*np.random.rand()
n_1 = np.random.rand() + 1j*np.random.rand()
n_3 = np.random.rand() + 1j*np.random.rand()

#%% MRRC 1Tx 2Rx

# Sinal no receptor
r0_mmrc_2rx = (h_0*signal_0)+n_0
r1_mmrc_2rx = (h_1*signal_0)+n_1

# Sinal combinado mmrc
s0_til_mmrc_2rx = ((alfa_0**2 + alfa_1**2)*signal_0) + (h_0_conj+n_0) + (h_1_conj*n_1)


#%% MRRC 1TX 4Rx







#%% New Scheme 2tx 1rx

# Manda o sinal s0 e -s1* pelo canal h0 e o sinal
# s1 e s0* pelo canal h1. Após a antena Rx são adicionados
# os ruídos n0 e n1 gerando r0 e r1. Ambos são direcionados
# para o estimador de canal, h0 e h1 são direcionados para o 
# o combinador e gera s0_til e s1_til. Tanto os sinais, quanto 
# os canais são direcionados para o detector de prob. máxima.

# h0
# h1

r0_NS_2Tx = (h_0*signal_0)+(h_1*signal_1)+n_0
r1_NS_2Tx = (-h_0*signal_1_conj)+(h_1*signal_0_conj)+n_1

s0_til_NS_2Tx = (h_0_conj*r0_NS_2Tx) + (h_1*np.conjugate(r1_NS_2Tx))
s1_til_NS_2Tx = (h_1_conj*r0_NS_2Tx) - (h_0*np.conjugate(r1_NS_2Tx))

# Max Prob Decisor







#%% New Scheme with 2Tx & 2Rx
# Resultado da banda base
r_0_NS_2Tx_2Rx = (h_0*signal_0)+(h_1*signal_1)+n_0
r_1_NS_2Tx_2Rx = -(h_0*signal_1_conj)+(h_1*signal_0_conj)+(n_1)
r_2_NS_2Tx_2Rx = (h_2*signal_0)+(h_3*signal_1)+(n_2)
r_3_NS_2Tx_2Rx = -(h_2*signal_1_conj)+(h_3*signal_0)+(n_3)

# Receiver combining scheme for two-branch

#s_0_til = (np.conjugate(h_0)*r_0)+((h_1)*np.conjugate(r_1))
#s_1_til = (np.conjugate(h_1)*r_0)-((h_0)*np.conjugate(r_1))
s0_til_NS_2Tx_2Rx = ((alfa_0**2+alfa_1**2)*signal_0) + (h_0_conj*n_0) + (h_1*np.conjugate(n_1))
s1_til_NS_2Tx_2Rx = ((alfa_0**2+alfa_1**2))-(h_0*np.conjugate(n_1))+(h_1_conj*n_0)

