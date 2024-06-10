#%%

# import

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

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
# def ph_g_rayleigh(h, Ph=1):
#     g = h**2    
#     Ph_Rayleigh = ((1)/(Ph))*(np.exp((-g)/(Ph)))
#     return Ph_Rayleigh


#%% Parâmetros

N = 10000
t = np.linspace(0,30,N)
f1 = 30
f2 = 80

# Sinal
s0_symb = []
   
signal_0 = (1/(np.sqrt(2)))*np.exp(1j *2*np.pi*f1*t)
signal_1 = (1/(np.sqrt(2)))*np.exp(1j *2*np.pi*f2*t)
signal_0_conj = (1/(np.sqrt(2)))*np.conjugate(signal_0)
signal_1_conj = (1/(np.sqrt(2)))*np.conjugate(signal_1)


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
h_0 = canal_h(alfa_0, theta_0)*(1/(np.sqrt(2)))
h_0_conj = np.conjugate(h_0)*(1/(np.sqrt(2)))
h_1 = canal_h(alfa_1, theta_1)*(1/(np.sqrt(2)))
h_1_conj = np.conjugate(h_1)*(1/(np.sqrt(2)))
h_2 = canal_h(alfa_2, theta_2)*(1/(np.sqrt(2)))
h_2_conj = np.conjugate(h_2)*(1/(np.sqrt(2)))
h_3 = canal_h(alfa_3, theta_3)*(1/(np.sqrt(2)))
h_3_conj = np.conjugate(h_3)*(1/(np.sqrt(2)))

# Ruído
n_0 = np.random.rand() + 1j*np.random.rand()*(1/(np.sqrt(2)))
n_2 = np.random.rand() + 1j*np.random.rand()*(1/(np.sqrt(2)))
n_1 = np.random.rand() + 1j*np.random.rand()*(1/(np.sqrt(2)))
n_3 = np.random.rand() + 1j*np.random.rand()*(1/(np.sqrt(2)))

#%% MRRC 1Tx 2Rx

# Sinal no receptor
r0_mmrc_2rx = (h_0*signal_0)+n_0
r1_mmrc_2rx = (h_1*signal_0)+n_1

# Sinal combinado mmrc
s0_til_mmrc_2rx = ((alfa_0**2 + alfa_1**2)*signal_0) + (h_0_conj+n_0) + (h_1_conj*n_1)


# Max Prob decisor (MPD)
d_s0_mmrc_2rx_s0 = dist_euclid_quad(s0_til_mmrc_2rx, signal_0)
d_s0_mmrc_2rx_s1 = dist_euclid_quad(s0_til_mmrc_2rx, signal_1)


# MPD_MRRC_2RX = 

snrindB_range = range(0, 10)
itr = len(snrindB_range)
ber = [None] * itr
ber1 = [None] * itr
ber3 = []
det_symbol = []
ipHat2 = []

for n in range(0, itr):

    snrindB = snrindB_range[n]
    snr = 10.0 ** (snrindB / 10.0)

    no_errors = 0

    for j in range (0, N):

        pt1 = ((alfa_0**2 + alfa_1**2) -1)*(np.abs(signal_0))**2 + d_s0_mmrc_2rx_s0 
        # print(pt2)
        pt2 = ((alfa_0**2 + alfa_1**2) -1)*((np.abs(signal_1))**2) + d_s0_mmrc_2rx_s1 
        # print(pt2)
        if pt1[j]<=pt2[j]:
            MDP_MRRC_2RX = signal_0   # tx
            r_MRRC_2RX = r0_mmrc_2rx  # rx
        if pt1[j]>pt2[j]:
            MDP_MRRC_2RX = signal_1   # tx
            r_MRRC_2RX = r1_mmrc_2rx  # rx
        
        # ipHat = np.real(r_MRRC_2RX)
        # ber2 = np.sum(MDP_MRRC_2RX != ipHat)/N


        # print(MDP_MRRC_2RX)
        #det_symbol = 2 * (r_MRRC_2RX >= 0.5) - 1

        if r_MRRC_2RX[j] >= 0.5:
            r_MRRC_2RX_new = 1
            det_symbol2 = 2 * r_MRRC_2RX_new -1
            det_symbol = np.append(det_symbol, det_symbol2)
        else:
            r_MRRC_2RX_new = 0
            det_symbol2 = 2 * r_MRRC_2RX_new -1
            det_symbol = np.append(det_symbol, det_symbol2)

        
        # no_errors += 1 * (MDP_MRRC_2RX != det_symbol2)

        # print(f"Det_symbol: {det_symbol} ")
        # print(f"no_error: {no_errors}")
        ipHat = (np.real(r_MRRC_2RX)/snrindB)
        #ipHat2 = np.append(ipHat2, ipHat)
        ber2 = (np.sum(det_symbol2 != ipHat)/snrindB)/N
        
    # ipHat = np.real(r_MRRC_2RX)
    # ber2 = np.sum(det_symbol2 != ipHat)/N
    
    #ber[n] = no_errors / n
    ber3 = np.append(ber3,ber2)
    print("SNR in dB:", snrindB)
    print("Numbder of errors:", no_errors)
    print("Error probability:", ber[n])

#%% PLOT
plt.semilogy(snrindB_range, ber3, 'o-')

plt.xlabel('snr(dB)')
plt.ylabel('BER')
plt.grid(True)
plt.title('BPSK Modulation')
plt.legend()
plt.show()
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

# Max Prob Decisor (MPD)

d_s0_NS_2TX_1RX_s0 = dist_euclid_quad(s0_til_NS_2Tx, signal_0)
d_s0_NS_2TX_1RX_s1 = dist_euclid_quad(s0_til_NS_2Tx, signal_1)

MPD_NS_2TX_1RX = ((alfa_0**2 + alfa_1**2) -1)*(np.abs(signal_0))**2 + d_s0_mmrc_2rx_s0 <= ((alfa_0**2 + alfa_1**2) -1)*((np.abs(signal_1))**2) + d_s0_mmrc_2rx_s1   

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

