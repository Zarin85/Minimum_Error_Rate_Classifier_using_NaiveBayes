import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import numpy.linalg as md
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

p_train = pd.read_csv('test.txt', header=None, sep=',', dtype='float64')
train_arr = p_train.values
len_train = train_arr[:, 0].size



class_1 = []
class_2 = []


u_1 = [0,0]
u_2 = [2,2]
s_1 = [[0.25,0.3],[0.3,1]]
s_2 = [[0.5,0],[0,0.5]]



row = []
row1 = []

def pdf(u,s,train_arr):
    s_i = np.linalg.inv(s)
    p = pow(6.2832,2)
    sig= md.det(s)
    p = p*sig
    sqt  = m.sqrt(p)
    dim = 1/sqt
    sub = train_arr - u
    sub_t  =np.array(sub).T   
    e  = np.dot(sub,s_i)
    e_1  =np.dot(e,sub_t)
    e_111  = -0.5 * e_1
    exp  = m.exp(e_111)
    w1= exp*dim
    return w1
        

for i in range(len_train):   
     w1 = pdf(u_1,s_1,train_arr[i,:])
     w2 = pdf(u_2,s_2,train_arr[i,:])

     if(w1>w2):
        class_1.append(train_arr[i,:])
      
     else:
        class_2.append(train_arr[i,:])

    

class_1 = np.array(class_1)
class_2 = np.array(class_2)

x1 = class_1[:, 0]
y1 = class_1[:, 1]
x2 = class_2[:, 0]
y2 = class_2[:, 1]

plt.scatter(x1, y1, color='red',label='class_1', marker='o')
plt.scatter(x2, y2, color='green',label='class_2', marker='*')
plt.legend()
plt.show()


x = np.linspace(-6,6,300)
y = np.linspace(-6,6,300)
X, Y = np.meshgrid(x,y)
Z = np.zeros_like(X)
D= np.zeros_like(X)
len_x = len(x)
len_y = len(y)
for i in range(len_x):
    for j in range(len_y):  
       w1 = pdf(u_1,s_1,np.array([x[i],y[j]]))
       w2 = pdf(u_2,s_2,np.array([x[i],y[j]]))
       if(w1>w2):
        Z[j][i] = w1
      
       else:
         Z[j][i] = w2
       D[j][i] = w1-w2

       
fig = plt.figure()
graph = Axes3D(fig)
graph.scatter(class_1[:,0], class_1[:,1], label='class_1', c='r', marker='o')
graph.scatter(class_2[:,0], class_2[:,1], label='class_2', c='b', marker='*')

graph.plot_surface(X, Y, Z, rstride=8, cstride=8, linewidth=1, antialiased=True, cmap='viridis', alpha=0.8)
graph.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap='viridis', alpha=0.6)
graph.contour3D(X, Y, D, zdir='z', offset=-0.15, cmap='viridis')
graph.set_zlim(-0.15,0.2)
graph.set_zticks(np.linspace(0,0.3,7))

graph.set_xlabel('X axis')
graph.set_ylabel('Y axis')
graph.set_zlabel('Probability Density')
graph.legend()

plt.show()














