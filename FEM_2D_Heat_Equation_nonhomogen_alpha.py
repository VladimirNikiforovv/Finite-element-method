import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numba import int64, float64
from numba.experimental import jitclass


"""Указание структуры для си интерпретатора"""
struct = [
    ('N', int64), #Колличество узлов на отрезке
    ('T', int64), #Количество отсчетов по времени 
    ('L', float64), #Длинна стороны прямоугольной области 
    ('h', float64), #Дискет по пространству
    ('dt', float64), #Дискет по времени 
    ('alpha', float64[:,:]), #значение коэффициента теплопроводности
    ('source', float64[:,:,:]), #Массив значений функции источника 
    ('init_c', float64[:,:]), #Массив начальных условий 
    ('Set_phi', float64[:,:,:]), #Массив значений из множества базисных функций 
    ('x', float64[:]),#массив значений по ортам 
    ('y', float64[:]),#массив значений по ортам 
    ('K', float64[:,:]), #Матрица, соответствующая решению МКЭ, Соответствует обозначению теор. вывода 
    ('M', float64[:,:]), #Матрица, соответствующая решению МКЭ, Соответствует обозначению теор. вывода 
    ('F', float64[:,:]), #Матрица, соответствующая решению МКЭ, Соответствует обозначению теор. вывода дляфункции источника
    ('Q', float64[:,:]), #Матрица, соответствующая решению МКЭ 
    ('u', float64[:,:,:]) #Временная часть базисной функций линейной комбинации
    ]

@jitclass(struct)
class General_FEM:
    """Класс решения уравнения Шредингера методом конечных элементов"""
    def __init__(self, N = 10, T = 100, L = 1, alpha = 2, C = 0.01):
        
        self.N = N         
        self.T = T 
        self.alpha = np.zeros((N, N), dtype=np.float64)
        self.alpha[:,:] = alpha
        self.source = np.zeros((N, N, T), dtype=np.float64)        
        self.init_c = np.zeros((N, N), dtype=np.float64)        
        self.L = L
        self.x = np.linspace(0, L, N)
        self.y = self.x
        self.h = self.x[1]-self.x[0]
        self.dt = (self.h**2)*C
        
    def set_source(self, source):
        """установление значения источника""" 
        self.source = source
       
    def set_init(self, init_c):
        """установление начальных условий""" 
        self.init_c = init_c
  
    def set_alpha(self, alpha):
        """установление значения коэффициента теплопроводности""" 
        self.alpha = alpha  
   
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
        self.K = np.zeros((self.N**2,self.N**2), dtype=np.float64)
        self.M = np.zeros((self.N**2,self.N**2), dtype=np.float64)
        self.F = np.zeros((self.N**2, self.T), dtype=np.float64)  
        
        for i in range(0,self.N**2):
            for j in range(0,self.N**2):
                a = 0
                b = 0
                d_x_fi = np.zeros((self.N,self.N), dtype=np.float64)
                d_y_fi = np.zeros((self.N,self.N), dtype=np.float64)
                d_x_fj = np.zeros((self.N,self.N), dtype=np.float64)
                d_y_fj = np.zeros((self.N,self.N), dtype=np.float64)
                

               
                for n in range(0, self.N):     
                          
                    d_x_fi[n, 1:] = (self.Set_phi[n,1:,i] - 
                                      self.Set_phi[n,:-1,i])/self.h
                    d_y_fi[1:, n] = (self.Set_phi[1:,n,i] - 
                                      self.Set_phi[:-1,n,i])/self.h                                                           

                    d_x_fj[n, 1:] = (self.alpha[n,1:]**2*(self.Set_phi[n,1:,j] - 
                                      self.Set_phi[n,:-1,j])/self.h  + 
                                     self.Set_phi[n,1:,j]*(self.alpha[n,1:]**2 -
                                                           self.alpha[n,:-1]**2))

                    
                    d_y_fj[1:, n] = (self.alpha[1:,n]**2*(self.Set_phi[1:,n,j] - 
                                      self.Set_phi[:-1,n,j])/self.h + 
                                     self.Set_phi[1:,n,j]*(self.alpha[1:,n]**2 - 
                                                           self.alpha[:-1,n]**2))
                    
                for k in range(0,self.N):
                    for l in range(0,self.N):
                        
                        a += self.Set_phi[k,l,i]*self.Set_phi[k,l,j]*self.h**2
                        
                        b += (d_x_fi[k,l]*d_x_fj[k,l]+
                              d_y_fi[k,l]*d_y_fj[k,l])*self.h**2
                        
                self.M[i,j] = b
                self.K[i,j] = a
                
    def calc_right_part(self):
        """Вычисление соответствующих матриц метода для источника"""
        self.F = np.zeros((self.N**2, self.T), dtype=np.float64)   
                
        for t in range(0, self.T):
            for j in range(0, self.N**2):
                c = 0
                for k in range(0,self.N):
                    for l in range(0,self.N):
                        c += self.source[k,l,t]*self.Set_phi[k,l,j]*self.h**2
                self.F[j,t] = c
                    
    def calc_time_dependent(self):
        """Вычисление эволюции по времени"""                                    
        Q = np.zeros((self.N**2,self.T), dtype=np.float64)            
        
        for k in range(0,self.N):
            for l in range(0,self.N):
                Q[k + self.N*l,0] = self.init_c[k,l]
        
        lp = self.K + (self.dt/2)*self.M
        for n in range(0, self.T-1):
            rp = (self.dt/2)*(self.F[:,n]+self.F[:,n+1]) + (self.K - (self.dt/2)*self.M).dot(Q[:,n])
            Q[:,n+1] = np.linalg.solve(lp, rp)
            
        self.u = np.zeros((self.N,self.N,self.T), dtype=np.float64)
        for n in range(0, self.T):
            for k in range(0, self.N):
                for l in range(0, self.N):
                    self.u[k,l,n] = Q[k + self.N*l,n]

class Heat_Equation:
    """Общее решение с отображением результата"""
    def __init__(self, N = 10, T = 100, L = 1, alpha = 2, C = 0.01):
        self.matrix_metod = General_FEM(N = N, T = T, L = L, alpha = alpha, C = C)              
    
    def animation(self, min_scale = 0, max_scale = 1, name = "default.gif"):
        """Вывод анимации решения"""
        self.fig, ax = plt.subplots()

        self.ims = []
        for i in range(self.matrix_metod.T):
            self.im = ax.imshow(self.matrix_metod.u[:,:,i], animated=True,
                                cmap='plasma', aspect='equal', vmin=min_scale, vmax=max_scale)
            self.ims.append([self.im])
        self.ani = animation.ArtistAnimation(self.fig, self.ims, interval=20, blit=True,
                                        repeat_delay=1000)
        self.fig.colorbar(self.im)

        if name != "default.gif":
            self.ani.save(name) 
        plt.show()

