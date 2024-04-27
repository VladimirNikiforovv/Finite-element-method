import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numba import int64, float64, complex128
from numba.experimental import jitclass

"""Указание структуры для си интерпретатора"""
struct = [
    ('N', int64), #Колличество узлов на отрезке
    ('T', int64), #Количество отсчетов по времени 
    ('L', float64), #Длинна стороны прямоугольной области  
    ('h', float64), #Дискет по пространству  
    ('dt', float64), #Дискет по времени 
    ('potential', complex128[:,:]), #Массив значения потенциала по узлам   
    ('potential_td', complex128[:,:,:]), #Массив значения потенциала зависящий от времени  по узлам   
    ('init_c', complex128[:,:]), #Массив начальных условий 
    ('Set_phi', float64[:,:,:]), #Массив значений из множества базисных функций  
    ('x', float64[:]), #массив значений по ортам 
    ('y', float64[:]), #массив значений по ортам 
    ('K', complex128[:,:]), #Матрица, соответствующая решению МКЭ, Соответствует обозначению теор. вывода  
    ('M', complex128[:,:]), #Матрица, соответствующая решению МКЭ  
    ('V', complex128[:,:]), #Матрица, соответствующая решению МКЭ  
    ('V_td', complex128[:,:,:]), ##Матрица, соответствующая решению МКЭ 
    ('Q', complex128[:,:]), #Матрица, соответствующая решению МКЭ 
    ('psi', complex128[:,:,:]) #Временная часть базисной функций линейной комбинации
    ]

@jitclass(struct)
class General_FEM:
    """Класс решения уравнения Шредингера методом конечных элементов"""
    def __init__(self, N = 10, T = 100, L = 1, C = 0.01):
        self.N = N 
        self.T = T
        self.potential = np.zeros((N, N), dtype=np.complex128)
        self.potential_td = np.zeros((N, N, T), dtype=np.complex128)
        self.init_c = np.zeros((N, N), dtype=np.complex128)
        self.L = L
        self.x = np.linspace(0, L, N)
        self.y = self.x
        self.h = self.x[1]-self.x[0]
        self.dt = (self.h**2)*C
     
    def set_init(self, init_c):
        """установление начальных условий"""  
        self.init_c = init_c
  
    def set_potential_stat(self, potential):
        """установление значения потенциала независящего от времени"""  
        self.potential = potential  
        
    def set_potential_nonstat(self, potential_td):
        """установление значения потенциала зависящего от времени""" 
        self.potential_td = potential_td  
   
    def calc_set_phi(self):  
        """Заполнения множества базисных функций по пространству"""
        self.Set_phi = np.zeros((self.N,self.N, self.N**2), dtype=np.float64)
        
        def phi(k, l):
            fild = np.zeros((self.N,self.N), dtype=np.float64)
            fild[k, l] = 1 
            return fild
        
        for i in range(0,self.N):
            for j in range(0,self.N):
                self.Set_phi[:,:, i + self.N*j] = phi(i, j)                

    def calc_matrix(self):
        """Вычисление соответствующих матриц метода"""
        self.K = np.zeros((self.N**2,self.N**2), dtype=np.complex128)
        self.M = np.zeros((self.N**2,self.N**2), dtype=np.complex128)
        self.V = np.zeros((self.N**2, self.N**2), dtype=np.complex128)          
        
        for i in range(0,self.N**2):
            for j in range(0,self.N**2):
                a = 0
                b = 0
                с = 0
                
                d_x_fi = np.zeros((self.N,self.N), dtype=np.float64)
                d_y_fi = np.zeros((self.N,self.N), dtype=np.float64)
                d_x_fj = np.zeros((self.N,self.N), dtype=np.float64)
                d_y_fj = np.zeros((self.N,self.N), dtype=np.float64)
                                
                for n in range(0,self.N):                    
                    d_x_fi[n, 1:] = (self.Set_phi[n,1:,i] - 
                                     self.Set_phi[n,:-1,i])
                    d_y_fi[1:, n] = (self.Set_phi[1:,n,i] - 
                                     self.Set_phi[:-1,n,i])
                    d_x_fj[n, 1:] = (self.Set_phi[n,1:,j] - 
                                     self.Set_phi[n,:-1,j])
                    d_y_fj[1:, n] = (self.Set_phi[1:,n,j] - 
                                     self.Set_phi[:-1,n,j])
                
                for k in range(0,self.N):
                    for l in range(0,self.N):
                        
                        a += self.Set_phi[k,l,i]*self.Set_phi[k,l,j]*self.h**2
                        
                        b += (d_x_fi[k,l]*d_x_fj[k,l]+
                              d_y_fi[k,l]*d_y_fj[k,l])
                        
                        с += (self.potential[k,l]*self.Set_phi[k,l,i]*
                              self.Set_phi[k,l,j])*self.h**2
                        
                self.M[i,j] = b
                self.K[i,j] = a
                self.V[i,j] = с
                
    def calc_potential_time_dependent(self):
        """Вычисление соответствующих матриц метода
        потенциала зависящего от времени"""
        self.V_td = np.zeros((self.N**2, self.N**2, self.T), dtype=np.complex128)   
                
        for i in range(0, self.N**2):
            for j in range(0, self.N**2):
                for t in range(0, self.T):
                
                    c = 0
                    for k in range(0,self.N):
                        for l in range(0,self.N):
                            c += (self.potential_td[k,l,t]*self.Set_phi[k,l,i]*
                                  self.Set_phi[k,l,j])*self.h**2
                            
                    self.V_td[i,j,t] = c
                    
    def calc_time_dependent_v_stat(self):
        """Вычисление эволюции по времени,
        с потенциала независящего от времени"""
        Q = np.zeros((self.N**2,self.T), dtype=np.complex128)            
        
        for k in range(0,self.N):
            for l in range(0,self.N):
                Q[k + self.N*l,0] = self.init_c[k,l]
        
        
        lp = (self.K + (1j*self.dt/4)*self.M + (1j*self.dt/2)*self.V)
        for n in range(0, self.T-1):
            rp =np.dot((self.K - (1j*self.dt/4)*self.M - (1j*self.dt/2)*self.V) ,Q[:,n])
            Q[:,n+1] = np.linalg.solve(lp, rp)
            
        self.psi = np.zeros((self.N,self.N,self.T), dtype=np.complex128)
        for n in range(0, self.T):
            for k in range(0, self.N):
                for l in range(0, self.N):
                    self.psi[k,l,n] = Q[k + self.N*l,n]
                    
    def calc_time_dependent_v_nonstat(self):
        """Вычисление эволюции по времени,
        с потенциала зависящего от времени"""
        Q = np.zeros((self.N**2,self.T), dtype=np.complex128)            
        
        for k in range(0,self.N):
            for l in range(0,self.N):
                Q[k + self.N*l,0] = self.init_c[k,l]
        
        
        for n in range(0, self.T-1):
            rp = np.dot((self.K - (1j*self.dt/4)*self.M - 
                 (1j*self.dt/2)*self.V_td[:,:,n]),Q[:,n])
            
            Q[:,n+1] = np.linalg.solve((self.K + (1j*self.dt/4)*self.M + 
                                        (1j*self.dt/2)*self.V_td[:,:,n]), rp)
            
        self.psi = np.zeros((self.N,self.N,self.T), dtype=np.complex128)
        for n in range(0, self.T):
            for k in range(0, self.N):
                for l in range(0, self.N):
                    self.psi[k,l,n] = Q[k + self.N*l,n]



class Schrodinger_Equation:
    """Общее решение с отображением результата"""
    def __init__(self, N = 10, T = 100, L = 1, C = 0.05):
        self.matrix_metod = General_FEM(N = N, T = T, L = L, C = C)              
    
    def animation(self, min_scale = 0, max_scale = 1, name = "default.gif"):
        """Вывод анимации решения"""
        self.fig, ax = plt.subplots()

        self.ims = []
        for i in range(self.matrix_metod.T):
            self.im = ax.imshow((np.abs(self.matrix_metod.psi[:,:,i])**2), animated=True,
                                cmap='gray', aspect='equal', vmin=min_scale, vmax=max_scale)
            self.ims.append([self.im])
        self.ani = animation.ArtistAnimation(self.fig, self.ims, interval=20, blit=True,
                                        repeat_delay=1000)
        self.fig.colorbar(self.im)

        if name != "default.gif":
            self.ani.save(name) 
        plt.show()
