#%%

import numpy as np

# %% Questão 1 - Carros

# Fila com processo de Poisson. Tempo com variável exponencial
import numpy as np
import matplotlib.pyplot as plt
t = 0 #tempo inicial
lq = 0 #numero de elementos na fila
ls = 1 #numero de elementos no servidor
k1 = 0 #variavel auxiliar
k2 = 0 #variavel auxiliar
k = 0 #numero de partidas
lambda1=3/5 #taxa media de chegada
mu1=1 #taxa media de partida
chegadas = np.random.exponential(1/lambda1,100000) #vetor de chegadas

partidas = np.random.exponential(1/mu1,100000) #vetor de partidas
tc = [0] #tempo de chegadas
tp = [] #tempo de partidas
ts = [] #tempo no sistema

ta = t + chegadas[k1] #proxima chegada
td = t + partidas[k2] #proxima partida
t = np.minimum(ta, td) #proximo tempo de execucao

while (t < 5000): #tempo limite de simulacao

    if ta<td: #evento de chegada
        k1 = k1 + 1
        ta = [] #apaga o tempo de chegada atual
        if ls == 1: #servidor ocupado
            lq = lq + 1 #aumenta a fila
            ta = t + chegadas[k1] #gera nova chegada
            tc = np.append(tc, t) #aramazena tempo atual de chegada

        else: #servidor desocupado
            ls = 1 #ocupa o servidor
            k2 = k2 + 1
            td = t + partidas[k2] #gera nova partida
            ta = t + chegadas[k1] #gera nova chegada
            tc = np.append(tc, t) #aramazena tempo atual de chegada


    elif ta == td: #chegada e partida simultanea

        tc = np.append(tc, t) #armazena o tempo atual de chegada
        tp = np.append(tp, t) #armazena o tempo atual de partida

        k = k + 1 #incrementa a partida

        ts = np.append(ts,(tp[k-1] - tc[k-1])) #calcula o tempo no sistema

        ta = [] #apaga os valores atuais de chegada e partida
        td = []
        k1 = k1 + 1
        k2 = k2 + 1
        td = t + partidas[k2] #calcula a proxima partida
        ta = t + chegadas[k1] #calcula a proxima chegada



    else: #evento de partida
        td = [] #apaga o tempo de partida

        k = k + 1 #incrementa o numero de partidas

        if lq > 0:#existem pacotes na fila
            tp = np.append(tp, t) #armazeno o tempo da partida
            ts = np.append(ts, (tp[k - 1] - tc[k - 1])) #calculo o tempo no sistema
            lq = lq - 1 #diminuo a fila em 1
            k2 = k2 + 1
            td = t + partidas[k2]#calculo a proxima partida

        else: #nao ha pacotes na fila
            tp = np.append(tp, t)#armazeno o tempo da partida
            ts = np.append(ts, (tp[k - 1] - tc[k - 1])) #calculo o tempo no sistema

            td = np.infty #coloco um valor muito grande para a proxima partida
            ls = 0 #esvazio o servidor


    t = np.minimum(ta, td) #calculo o tempo minimo entre chegada e partida

tw=np.mean(ts)-1/mu1
eq=lambda1*np.mean(ts)
ew=lambda1*tw
rho=lambda1/mu1
print("tempo médio de carros no sistema")
print(np.mean(ts))
print("numero medio de carros no sistema")
print(eq)

# %% Questão 2 - Pacotes de Comutação

# Dados do problema
lambdaQ2= 40
CQ2 = 500000 #capacidade
LQ2 = 5000 #comprimento
muQ2 = CQ2/LQ2
lqQ2 = 0 #número de elemento na fila
lsQ2 = 1 #número de elemento no serviço 
K=0   
tQ2=0  #tempo inicial 
tcQ2=tQ2+np.random.exponential((1/lambdaQ2)) #tempo de chegada   
tpQ2=tQ2+np.random.exponential((1/muQ2)) #tempo de partida
tQ2=min(tcQ2,tpQ2)
Tempo_chegadaQ2 =[0]     #variavel do tempo de chegada
Tempo_partidaQ2 =[]      #variavel do tempo de partida
Tq=[]     #tempo no medio que um elemento gasta na fila 
while tQ2<1000:
    if tcQ2<tpQ2:  #chegada
        Tempo_chegadaQ2=np.append(Tempo_chegadaQ2,tQ2) #aloca o tempo de chegada
        tcQ2=tQ2+np.random.exponential((1/lambdaQ2))  #gera um novo tempo de chegada e soma com tq 
        if lsQ2==1: #se servidor ocupado
            lqQ2=lqQ2+1  #elemento na fila
        elif lsQ2==0: #se servidor desocupado
            lsQ2=1  #ocupa servidor
            tpQ2=tQ2+np.random.exponential((1/muQ2)) #novo tempo de partida  
    elif tcQ2>tpQ2: #partida
        K=K+1
        Tempo_partidaQ2 = np.append(Tempo_partidaQ2, tQ2) #aloca tempo de partida
        Tq = np.append(Tq, (Tempo_partidaQ2[K-1]-Tempo_chegadaQ2[K-1])) #cálculo do tempo de sistema
        if lqQ2>0:
            lqQ2=lqQ2-1    #tira o pacote da fila
            tpQ2=tQ2+np.random.exponential((1/muQ2)) #novo tempo de partida
            lsQ2=1  #ocupa servidor
        elif lqQ2==0:
            lsQ2=0
            tpQ2=np.infty
    tQ2=min(tcQ2,tpQ2)

EtsQ2=1/muQ2
EtqQ2 = np.mean(Tq)
Eq=lambdaQ2*EtqQ2
EtwQ2=EtqQ2-EtsQ2
Ew=lambdaQ2*EtwQ2

print(f"O tempo médio de permanência de um pacote no comutador é: {EtqQ2}" )
print(f"O tempo médio de espera na fila: {EtwQ2}")

# %% Questão 3 - Pacotes de Comutação

# Dados do problema
def funcao (contador):
    lambdaQ3= 200
    CQ3 = 256000 #capacidade
    LQ3 = 128*8 #comprimento
    muQ3 = CQ3/LQ3
    lqQ3 = 0 #número de elemento na fila
    lsQ3 = 1 #número de elemento no serviço 
    tQ3=0  #tempo inicial 
    tcQ3=tQ3+np.random.exponential((1/lambdaQ3)) #tempo de chegada   
    tpQ3=tQ3+np.random.exponential((1/muQ3)) #tempo de partida
    tQ3=min(tcQ3,tpQ3)
    Tempo_chegadaQ3 =[0]     #variavel do tempo de chegada
    Tempo_partidaQ3 =[]      #variavel do tempo de partida
    TqQ3=[]     #tempo no medio que um elemento gasta na fila 
    bloqueio = 0
    i=0
    K=0
    while K<10000:
        if tcQ3<tpQ3:  #chegada
            i+=1 #contador
            Tempo_chegadaQ3=np.append(Tempo_chegadaQ3,tQ3) #aloca o tempo de chegada
            tcQ3=tQ3+np.random.exponential((1/lambdaQ3))  #gera um novo tempo de chegada e soma com tq 
            if lsQ3==1: #se servidor ocupado
                if lqQ3 == contador:
                    bloqueio += 1
                elif lqQ3 < contador:
                    Tempo_chegadaQ3 = np.append(Tempo_chegadaQ3,tQ3)
                    lqQ3 += 1
            elif lsQ3==0: #se servidor desocupado
                lsQ3=1  #ocupa servidor
                Tempo_chegadaQ3 = np.append(Tempo_chegadaQ3,tQ3)
                tpQ3=tQ3+np.random.exponential((1/muQ3)) #novo tempo de partida  
        elif tcQ3>tpQ3: #partida
            K=K+1
            Tempo_partidaQ3 = np.append(Tempo_partidaQ3, tQ3) #aloca tempo de partida
            TqQ3 = np.append(TqQ3, (Tempo_partidaQ3[K-1]-Tempo_chegadaQ3[K-1])) #cálculo do tempo de sistema
            if lqQ3>0:
                lqQ3=lqQ3-1    #tira o pacote da fila
                tpQ3=tQ3+np.random.exponential((1/muQ3)) #novo tempo de partida
                lsQ3=1  #ocupa servidor
            elif lqQ3==0:
                lsQ3=0
                tpQ3=np.infty
        tQ3=min(tcQ3,tpQ3)

    Pblock = bloqueio/i
    EtqQ3 = np.mean(TqQ3)
    EqQ3=lambdaQ3*EtqQ3*(1-Pblock)
    Saida = [Pblock, EtqQ3, EqQ3]
    return Saida

buffer_1  = funcao(1)
buffer_5  = funcao(5)
buffer_10 = funcao(10)
buffer_15 = funcao(15)

print(f"A probabilidade de bloqueio, o n° médio de elementos e o tempo médio no sistema para um buffer de tamanho 1 é : {buffer_1}")
print(f"A probabilidade de bloqueio, o n° médio de elementos e o tempo médio no sistema para um buffer de tamanho 5 é : {buffer_5}")
print(f"A probabilidade de bloqueio, o n° médio de elementos e o tempo médio no sistema para um buffer de tamanho 10 é : {buffer_10}")
print(f"A probabilidade de bloqueio, o n° médio de elementos e o tempo médio no sistema para um buffer de tamanho 15 é : {buffer_15}")
#? N deu certo

# %% Questão 4 a)

# Dados do problema
lambdaQ4b= 1
CQ4b = 10000 #capacidade
LQ4b = 2000 #comprimento
muQ4b = CQ4b/LQ4b
rho = lambdaQ4b/muQ4b
lqQ4b = 0 #número de elemento na fila
lsQ4b = 1 #número de elemento no serviço 
K=0   
tQ4b=0  #tempo inicial 
tcQ4b=tQ4b+np.random.exponential((1/lambdaQ4b)) #tempo de chegada   
tpQ4b=tQ4b+(1/muQ4b) #tempo de partida
tQ4b=min(tcQ4b,tpQ4b)
Tempo_chegadaQ4b =[0]     #variavel do tempo de chegada
Tempo_partidaQ4b =[]      #variavel do tempo de partida
TqQ4=[]     #tempo no medio que um elemento gasta na fila 
while tQ4b<10000:
    if tcQ4b<tpQ4b:  #chegada
        Tempo_chegadaQ4b=np.append(Tempo_chegadaQ4b,tQ4b) #aloca o tempo de chegada
        tcQ4b=tQ4b+np.random.exponential((1/lambdaQ4b))  #gera um novo tempo de chegada e soma com tq 
        if lsQ4b==1: #se servidor ocupado
            lqQ4b=lqQ4b+1  #elemento na fila
        elif lsQ4b==0: #se servidor desocupado
            lsQ4b=1  #ocupa servidor
            tpQ4b=tQ4b+(1/muQ4b) #novo tempo de partida  
    elif tcQ4b>tpQ4b: #partida
        K=K+1
        Tempo_partidaQ4b = np.append(Tempo_partidaQ4b, tQ4b) #aloca tempo de partida
        TqQ4 = np.append(TqQ4, (Tempo_partidaQ4b[K-1]-Tempo_chegadaQ4b[K-1])) #cálculo do tempo de sistema
        if lqQ4b>0:
            lqQ4b=lqQ4b-1    #tira o pacote da fila
            tpQ4b=tQ4b+(1/muQ4b) #novo tempo de partida
            lsQ4b=1  #ocupa servidor
        elif lqQ4b==0:
            lsQ4b=0
            tpQ4b=np.infty
    tQ4b=min(tcQ4b,tpQ4b)

EtsQ4b=1/muQ4b
EtqQ4b = np.mean(TqQ4)
Eq=lambdaQ4b*EtqQ4b
EtwQ4b=EtqQ4b-EtsQ4b
Ew=lambdaQ4b*EtwQ4b

print(f"O tempo médio de permanência de um pacote no comutador é: {EtqQ4b}" )
print(f"O tempo médio de espera na fila: {EtwQ4b}")




# %% Questão 4 b)

# Dados do problema
lambdaQ4b= 1
CQ4b = 10000 #capacidade
LQ4b = 2000 #comprimento
muQ4b = CQ4b/LQ4b
lqQ4b = 0 #número de elemento na fila
lsQ4b = 1 #número de elemento no serviço 
K=0   
tQ4b=0  #tempo inicial 
tcQ4b=tQ4b+np.random.exponential((1/lambdaQ4b)) #tempo de chegada   
tpQ4b=tQ4b+np.random.exponential((1/muQ4b)) #tempo de partida
tQ4b=min(tcQ4b,tpQ4b)
Tempo_chegadaQ4b =[0]     #variavel do tempo de chegada
Tempo_partidaQ4b =[]      #variavel do tempo de partida
TqQ4=[]     #tempo no medio que um elemento gasta na fila 
while tQ4b<10000:
    if tcQ4b<tpQ4b:  #chegada
        Tempo_chegadaQ4b=np.append(Tempo_chegadaQ4b,tQ4b) #aloca o tempo de chegada
        tcQ4b=tQ4b+np.random.exponential((1/lambdaQ4b))  #gera um novo tempo de chegada e soma com tq 
        if lsQ4b==1: #se servidor ocupado
            lqQ4b=lqQ4b+1  #elemento na fila
        elif lsQ4b==0: #se servidor desocupado
            lsQ4b=1  #ocupa servidor
            tpQ4b=tQ4b+np.random.exponential((1/muQ4b)) #novo tempo de partida  
    elif tcQ4b>tpQ4b: #partida
        K=K+1
        Tempo_partidaQ4b = np.append(Tempo_partidaQ4b, tQ4b) #aloca tempo de partida
        TqQ4 = np.append(TqQ4, (Tempo_partidaQ4b[K-1]-Tempo_chegadaQ4b[K-1])) #cálculo do tempo de sistema
        if lqQ4b>0:
            lqQ4b=lqQ4b-1    #tira o pacote da fila
            tpQ4b=tQ4b+np.random.exponential((1/muQ4b)) #novo tempo de partida
            lsQ4b=1  #ocupa servidor
        elif lqQ4b==0:
            lsQ4b=0
            tpQ4b=np.infty
    tQ4b=min(tcQ4b,tpQ4b)

EtsQ4b=1/muQ4b
EtqQ4b = np.mean(TqQ4)
Eq=lambdaQ4b*EtqQ4b
EtwQ4b=EtqQ4b-EtsQ4b
Ew=lambdaQ4b*EtwQ4b

print(f"O tempo médio de permanência de um pacote no comutador é: {EtqQ4b}" )
print(f"O tempo médio de espera na fila: {EtwQ4b}")

# %%
