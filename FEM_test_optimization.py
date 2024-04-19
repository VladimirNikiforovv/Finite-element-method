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

@njit
def K(Set_phi, N, h):
    K_M = np.zeros((N**2,N**2))
    for i in range(0,N**2):
        for j in range(0,N**2):
            a = 0
            for k in range(0,N):
                for l in range(0,N):
                    a += Set_phi[k,l,i]*Set_phi[k,l,j]*h**2
            K_M[i,j] = a
    return K_M

@njit
def M(Set_phi, N, h):
    M_M = np.zeros((N**2,N**2))
    
    for i in range(0,N**2):
        for j in range(0,N**2):
            a = 0
            d_x_fi = np.zeros((N,N))
            d_y_fi = np.zeros((N,N))
            d_x_fj = np.zeros((N,N))
            d_y_fj = np.zeros((N,N))
            
            for n in range(0,N):
                d_x_fi[n, 1:] = (Set_phi[n,1:,i] - Set_phi[n,:-1,i])
                d_y_fi[1:, n] = (Set_phi[1:,n,i] - Set_phi[:-1,n,i])
                d_x_fj[n, 1:] = (Set_phi[n,1:,j] - Set_phi[n,:-1,j])
                d_y_fj[1:, n] = (Set_phi[1:,n,j] - Set_phi[:-1,n,j])
            
            for k in range(0,N):
                for l in range(0,N):
                    a += (d_x_fi[k,l]*d_x_fj[k,l]+d_y_fi[k,l]*d_y_fj[k,l])*h
                    
            M_M[i,j] = a 
    return M_M

@njit
def F(Set_phi, f, N, h):
    
    rp = np.zeros((N**2))
    
    for n in range(0,N**2):
        a = 0
        for k in range(0,N):
            for l in range(0,N):
                a += f[k,l,n]*Set_phi[k,l,n]*h**2
                
        rp[n] = a #sum(sum(f*Set_phi[n]))*h**2
        
    return rp

alpha = 30
N = 60
T = 500
L = 1
x = np.linspace(0, L, N)
y = x
h = x[1]-x[0]
dx = x[1]-x[0]
dt = (dx**2)*0.1
t = np.linspace(0, dt*T, T)

Set_phi = np.zeros((N,N, N**2))
for k in range(0,N):
    for l in range(0,N):
        Set_phi[:,:, k + N*l] = phi(k, l, N)
             
KK = K(Set_phi, N, h)
    
MM = M(Set_phi, N, h)

Matrix_plus = KK + (alpha**2)*(dt/2)*MM
Matrix_minus = KK - (alpha**2)*(dt/2)*MM

Q = np.zeros((N**2,T))
F_mtrx = np.zeros((N**2,T))
source = np.zeros((N,N,T))


for n in range(0,T):
    for k in range(0,N-1):
        for l in range(0,N-1):
            source[k,l,n] = np.exp((-(x[k]-L/2)**2 - (y[l]-L/2)**2)/0.01)*np.sin(1000*t[n])
            F_mtrx[k + N*l,n] = source[k,l,n]

# pkt = np.zeros((N,N))
# for k in range(0,N):
#     for l in range(0,N):
#         Q[k + N*l,0] = 12*np.exp((-(x[k]-L/2)**2 - (y[l]-L/2)**2)/0.01)
#         pkt[k,l] = Q[k + N*l,0]

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
    im = ax.imshow(Term[:,:,i], animated=True, cmap='plasma', aspect='equal', vmin=0, vmax=1)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
fig.colorbar(im)

ani.save("heat_2d_gauss60.scor_sin.gif")
plt.show()