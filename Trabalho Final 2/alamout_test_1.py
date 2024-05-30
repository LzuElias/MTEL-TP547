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
t = np.linspace(0,30,10000)
f1 = 30
f2 = 80

# Sinal
s0_symb = []
   
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


# Max Prob decisor (MPD)
d_s0_mmrc_2rx_s0 = dist_euclid_quad(s0_til_mmrc_2rx, signal_0)
d_s0_mmrc_2rx_s1 = dist_euclid_quad(s0_til_mmrc_2rx, signal_1)


# MPD_MRRC_2RX = 

snrindB_range = range(0, 10)
itr = len(snrindB_range)
ber = [None] * itr
ber1 = [None] * itr

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

        # print(MDP_MRRC_2RX)
        det_symbol = 2 * (r_MRRC_2RX >= 0) - 1
        # det_symbol_1 = 2 * (r1_mmrc_2rx >= 0) - 1
        no_errors += 1 * (MDP_MRRC_2RX != det_symbol)

        # print(f"Det_symbol: {det_symbol} ")
        # print(f"no_error: {no_errors}")
        
        
    ber[n] = no_errors / n
    print("SNR in dB:", snrindB)
    print("Numbder of errors:", no_errors)
    print("Error probability:", ber[n])

plt.semilogy(snrindB_range, ber, 'o-')

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



#%% BER



# N = 10**6 # number of bits or symbols

# simber=[]
# # Transmitter
# ip = np.random.uniform(0,1,N)>0.5; # generating 0,1 with equal probability

# s = 2*ip-1; # BPSK modulation 0 -> -1; 1 -> 1

#  # white gaussian noise, 0dB variance
# Eb_N0_dB = np.arange(-3,10,1); # multiple Eb/N0 values

# for ii in Eb_N0_dB:
#    # Noise addition

#    n = 1 / np.sqrt(2) * s0_til_mmrc_2rx   
#    #n = 1 / np.sqrt(2) * (np.random.normal(0, 1, N) + np.random.normal(0, 1, N) * 1j);
#    y = s + 10**(-ii/20)*n; # additive white gaussian noise

#    # receiver - hard decision decoding
   
#    ipHat = (alfa_0**2 + alfa_1**2 -1)*(np.abs(signal_0))**2 + d_s0_mmrc_2rx_s0 <= (alfa_0**2 + alfa_1**2 -1)*(np.abs(signal_1))**2 + d_s0_mmrc_2rx_s1   
#    # ipHat = np.real(y)>0;
#    ber = np.sum(ip !=ipHat)/N; # simulated ber
#    simber=np.append(simber,ber)

# theoryBer = 0.5*special.erfc(np.sqrt(10.**(Eb_N0_dB/10))); # theoretical ber


# plt.semilogy(Eb_N0_dB,theoryBer,'b.-');
# plt.semilogy(Eb_N0_dB,simber,'mx-');
# plt.axis([-3, 10, 10**(-5), 0.5])
# plt.grid()
# plt.figlegend(['theory', 'simulation']);
# plt.xlabel('Eb/No, dB');
# plt.ylabel('Bit Error Rate');
# plt.show()

# # %%
# from numpy import sqrt
# import random
# import matplotlib.pyplot as plt

# N = 300000
# snrindB_range = range(0, 10)
# itr = len(snrindB_range)
# ber = [None] * itr
# ber1 = [None] * itr
# tx_symbol = 0
# noise = 0
# ch_coeff = 0
# rx_symbol = 0
# det_symbol = 0
# for n in range(0, itr):

#     snrindB = snrindB_range[n]
#     snr = 10.0 ** (snrindB / 10.0)
#     noise_std = 1 / sqrt(2 * snr)
#     noise_mean = 0

#     no_errors = 0
#     for m in range(0, N):
#         # tx_symbol = 2 * random.randint(0, 1) - 1
#         # noise = random.gauss(noise_mean, noise_std)
#         #rx_symbol = tx_symbol + noise
#         tx_symbol = signal_0
#         rx_symbol = r0_mmrc_2rx
#         det_symbol = 2 * (rx_symbol >= 0) - 1
#         no_errors += 1 * (tx_symbol != det_symbol)

#     ber[n] = no_errors / N
#     print("SNR in dB:", snrindB)
#     print("Numbder of errors:", no_errors)
#     print("Error probability:", ber[n])
# plt.semilogy(snrindB_range, ber, 'o-', label='practical')

# plt.xlabel('snr(dB)')
# plt.ylabel('BER')
# plt.grid(True)
# plt.title('BPSK Modulation')
# plt.legend()
# plt.show()
# %%
