import math
import arff, numpy as np
import random
import sys

#----------package import----------

dataset = arff.load(open('C:/Users/xhs/Desktop/ECG5000/ECG5000_TRAIN.arff', 'r'))
data = np.array(dataset['data'])
data =data.astype(np.float64)
datax = data[:500,0:140]
datay = data[:,140]
n,m = datax.shape
#-----------data import-------------------

def Distance(x,y):
    return abs(x-y)

def dtw(X,Y):
    Lx=len(X)
    Ly=len(Y)
    path=[]
    M=[[Distance(X[i],Y[j]) for i in range(Lx)]for j in range(Ly)]
    D=[[0 for i in range(Lx+1)]for j in range(Ly+1)]
    D[0][0]=0
    for i in range(1,Lx+1):
        D[0][i]=sys.maxsize
    for j in range(1,Ly+1):
        D[j][0]=sys.maxsize
    for i in range(1,Ly+1):
        for j in range(1,Lx+1):
            D[i][j]=M[i-1][j-1]+min(D[i-1][j],D[i-1][j-1],D[i][j-1])
    minD=D[Ly][Lx]
    return minD

def dtwsimi(x,y):
    zero=[0]
    simi=(dtw(x,zero)**2+dtw(y,zero)**2-dtw(x,y)**2)/2
    return simi
#-----------------DTF的实现--------------------------------------------------

def buildmatrix(n):
    N = 0
    a = np.zeros([n,n])
    for i in range(0,n):
        k=random.randint(0,3)
        while(k < n-i and N < 10 * n * np.math.log(n,10)):
            a[i][k+i] = 1
            a[k+i][i] = 1
            k = k + random.randint(1,4)
            N = N + 1
    print(N)
    print(10 * n * np.math.log(n,10))
    return a

#-----------------------选出c*nlog(n)对元素-----------------------------------

def cubicRoot(d):
    if d < 0.0 :
        return -cubicRoot(-d)
    else:
        return pow(d, 1.0 / 3.0)
def xu(a,b):
    a3 = 4 * pow(a, 3)
    b2 = 27 * pow(b, 2)
    delta = a3 + b2
    if delta <= 0:  # 3 distinct real roots or 1 real multiple solution
        r3 = 2 * math.sqrt(-a / 3)
        th3 = math.atan2(math.sqrt(-delta / 108), -b / 2) / 3
        ymax = 0
        xopt = 0
        for k in range(0,5,2):
            x = r3 * math.cos(th3 + ((k * 3.14159265) / 3))
            if x>=0:
               y = pow(x, 4) / 4 + a * pow(x, 2) / 2 + b * x
               if y < ymax :
                  ymax = y
                  xopt = x
        return xopt
    else:
        z = math.sqrt(delta / 27)
        x = cubicRoot(0.5 * (-b + z)) + cubicRoot(0.5 * (-b - z))
        y = pow(x, 4) / 4 + a * pow(x, 2) / 2 + b * x
        if x>=0 and y<0:
            x=x
        else:
            x=0
        return x

#-----------------------求解Xji迭代值-----------------------------------------
d=30

#---------------------initial settings---------------------------------------    
X = np.zeros([n,d])
A = np.zeros([n,n])
for i in range(0,n):
    for j in range(i,n):
        A[i][j] = dtwsimi(datax[i],datax[j])
        A[j][i] = A[i][j]


for t in range(10):
    print('t=')
    print(t)
    for i in range(d): 
        for j in range(n):
            R = A - np.dot(X,X.T)
            p = -(X[j][i] **2) - R[j][j]
            q = X[j][i] * R[j][j]
            for k in range(n):
                p = p + X[k][i] **2
                q = q - X[k][i] * R[j][k]
            X[j][i] = xu(p,q)
X1 = np.zeros([n,d+1])
for i in range(500):
    for j in range(d):
        X1[i][j]=X[i][j]
    X1[i][d]=datay[i]
    
       
print(np.dot(X,X.T))
print(np.linalg.norm(R,ord=2)/np.linalg.norm(A,ord=2) ) 
np.savetxt('new.csv',X1 , delimiter = ',')  
