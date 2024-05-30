import numpy as np


N = np.array([
    [1,    -0.5, -0.5,    0,    0,    0],
    [-0.5,    1,    0, -0.5,    0,    0],
    [-0.5,    0,    1,    0, -0.5,    0],
    [0,    -0.5,    0,    1,    0, -0.5],
    [0,       0, -0.5,    0,    1, -0.5],
    [0,       0,    0,    0,    0,    1]
    ])


N_inv = np.linalg.inv(N)

media_passos = np.array([
    [np.sum(N_inv[0])],
    [np.sum(N_inv[1])],
    [np.sum(N_inv[2])],
    [np.sum(N_inv[3])],
    [np.sum(N_inv[4])],
    [np.sum(N_inv[5])],
    
    ])

# print(N_inv)

# print(media_passos)

# %%
N2 = np.array([
    [1,      0,    0,    0],
    [0.3,  0.4,    0,    0],
    [0.3,    0,  0.4,  0.3],
    [0,      0,    0,    1]
    ])


N_inv2 = np.linalg.inv(N2)

media_passos2 = np.array([
    [np.sum(N_inv2[0])],
    [np.sum(N_inv2[1])],
    [np.sum(N_inv2[2])],
    [np.sum(N_inv2[3])],  
    ])

#print(N_inv2)

print(media_passos2)


R = np.array([

    [1,   0],
    [0.3, 0],
    [0, 0.3],
    [0,   1],


])


B = N_inv2*R
print(B)



# %%
