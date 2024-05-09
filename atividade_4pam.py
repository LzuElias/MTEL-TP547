import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sistema
M = 4  # Número de símbolos na constelação
T = 1  # Duração de cada símbolo em segundos
fs = 100  # Frequência de amostragem em Hz
SNR_dB = 10  # Relação sinal-ruído (SNR) em dB

# %% Modulação

# Geração da constelação M-PAM
constellation = np.array([-3, -1, 1, 3])  # Amplitudes dos símbolos

# Geração dos símbolos de entrada
symbols = np.random.randint(0, M, 1000)  # 1000 símbolos aleatórios de 0 a M-1


# Modulação M-PAM
modulated_signal = constellation[symbols]

# Adição de ruído gaussiano branco
SNR_linear = 10**(SNR_dB / 10)  # Conversão de dB para linear
noise_power = SNR_linear
noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))

received_signal = modulated_signal + noise


# %% Demodulação

# Geração do SNR variável ao longo do tempo
time = np.arange(0, len(modulated_signal)*T, T)
SNR_dB_time = 10 + 5 * np.sin(2 * np.pi * 0.01 * time)  # Ruído varia sinusoidalmente

# Adição de ruído gaussiano branco com SNR variável
noise_time = np.array([np.sqrt(1 / (10**(SNR_dB_time[i] / 10))) * np.random.randn() for i in range(len(modulated_signal))])
received_signal_time = modulated_signal + noise_time


# Limiares de decisão
thresholds = [(constellation[i] + constellation[i+1]) / 2 for i in range(len(constellation)-1)]

# Demodulação do sinal
demodulated_symbols = np.zeros_like(symbols)
for i, threshold in enumerate(thresholds):
    demodulated_symbols[np.abs(received_signal - threshold) <= (constellation[1] - constellation[0]) / 2] = i

# Demodulação do sinal com SNR variável
demodulated_symbols_time = np.zeros_like(symbols)
for i, threshold in enumerate(thresholds):
    # Adicionando ruído com SNR variável
    noise_var_time = 1 / (10**(SNR_dB_time[i] / 10))
    noise_time = np.sqrt(noise_var_time) * np.random.randn(len(received_signal))
    
    demodulated_symbols_time[np.abs(received_signal_time - threshold) <= (constellation[1] - constellation[0]) / 2] = i

# %% Plot


# Plotagem do sinal modulado
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(np.arange(0, len(modulated_signal)*T, T), modulated_signal, 'bo', label='Sinal Modulado')
plt.title(f'Sinal Modulado {M}-PAM')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plotagem da constelação do sinal modulado
plt.subplot(122)
plt.plot(constellation, np.zeros_like(constellation), 'ro', label='Constelação Original')
plt.title(f'Constelação do Sinal Modulado {M}-PAM')
plt.xlabel('Amplitude')
plt.ylabel('Zero')
plt.grid(True)
plt.legend()
plt.show()

# Plotagem do sinal demodulado com SNR variável
plt.figure(figsize=(10, 5))
plt.plot(received_signal_time, np.zeros_like(received_signal_time), 'go', label='Sinal Recebido (SNR Variável)')
plt.title(f'Sinal Demodulado {M}-PAM com SNR Variável')
plt.xlabel('Amplitude')
plt.ylabel('Zero')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(constellation, np.zeros_like(constellation), 'ro', label='Constelação Demodulada')
plt.title(f'Constelação do Sinal Demodulado {M}-PAM com SNR Variável')
plt.xlabel('Amplitude')
plt.ylabel('Zero')
plt.grid(True)
plt.legend()
plt.show()
plt.show
# %%
