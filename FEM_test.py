import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import triangle as tr
from numba import njit
import matplotlib.animation as animation
import datetime

start = datetime.datetime.now()
print('Время старта: ' + str(start))

@njit
def phi(k, l, N):
    fild = np.zeros((N,N))
    fild[k, l] = 1 
    return fild

Set_phi = dict()

def K(N, h):
    K_M = np.zeros((N**2,N**2))
    for i in range(0,N**2):
        for j in range(0,N**2):
            K_M[i,j] = sum(sum(Set_phi[i]*Set_phi[j])*h)*h
    return K_M

@njit
def grad_x(f_x_y, h, N):
    d_x_f = np.zeros((N,N))
    for i in range(0,N):
        #d_x_f[i, 1:] = np.diff(f_x_y[i,:])/h
        d_x_f[i, 1:] = (f_x_y[i,1:]-f_x_y[i,:-1])/h
    return d_x_f

@njit
def grad_y(f_x_y, h, N):
    d_y_f = np.zeros((N,N))
    for i in range(0,N):
        # d_y_f[1:, i] = np.diff(f_x_y[:,i])/h
        d_y_f[1:, i] = (f_x_y[1:,i]-f_x_y[:-1,i])/h
    return d_y_f

def M(N, h):
    M_M = np.zeros((N**2,N**2))
    for i in range(0,N**2):
        for j in range(0,N**2):
            M_M[i,j] = sum(sum(grad_x(Set_phi[i], h, N)*grad_x(Set_phi[j], h, N)+
                               grad_y(Set_phi[i], h, N)*grad_y(Set_phi[j], h, N)))*h**2
    return M_M

def F(f,N,h):
    rp = np.zeros((N**2))
    for n in range(0,N**2):
        rp[n] = sum(sum(f*Set_phi[n]))*h**2
    return rp

alpha = 10
N = 20
T = 1000
L = 1
x = np.linspace(0, L, N)
y = x
h = x[1]-x[0]
dx = x[1]-x[0]
dt = (dx**2)*0.01
t = np.linspace(0, dt*T, T)

Set_phi = dict()
for k in range(0,N):
    for l in range(0,N):
        Set_phi[k + N*l] = phi(k, l, N)
            

KK = K(N, h)
MM = M(N, h)

Matrix_plus = KK + (alpha**2)*(dt/2)*MM
Matrix_minus = KK - (alpha**2)*(dt/2)*MM

Q = np.zeros((N**2,T))
F_mtrx = np.zeros((N**2,T))
source = np.zeros((N,N,T))

# source[0,:,5:] = 10

# for n in range(0,T):
#     for k in range(0,N):
#         for l in range(0,N):
#             F_mtrx[k + N*l,n] = source[k,l,n]

pkt = np.zeros((N,N))
for k in range(0,N):
    for l in range(0,N):
        Q[k + N*l,0] = 12*np.exp((-(x[k]-L/2)**2 - (y[l]-L/2)**2)/0.1)
        pkt[k,l] = Q[k + N*l,0]

for n in range(0, T-1):
    rp = (dt/2)*(F_mtrx[:,n]+F_mtrx[:,n+1]) + Matrix_minus.dot(Q[:,n])
    Q[:,n+1] = np.linalg.solve(Matrix_plus, rp)
    
Term = np.zeros((N,N,T))
for n in range(0, T):
    for k in range(0, N):
        for l in range(0, N):
            Term[k,l,n] = Q[k + N*l,n]
            
#фиксируем и выводим время окончания работы кода
finish = datetime.datetime.now()
print('Время окончания: ' + str(finish))

# вычитаем время старта из времени окончания
print('Время работы: ' + str(finish - start)) 

# fig, axs = plt.subplots(figsize=(5,5), constrained_layout=True)
# p1 = axs.imshow(Term[:,:,-1], cmap='plasma', aspect='equal', vmin=np.min(Term), vmax=1)
# fig.colorbar(p1)


fig, ax = plt.subplots()

ims = []
for i in range(int(T)):
    im = ax.imshow(Term[:,:,i], animated=True, cmap='plasma', aspect='equal', vmin=0, vmax=12)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=1000)
fig.colorbar(im)
plt.show()