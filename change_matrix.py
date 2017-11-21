import arff, numpy as np


#----------package import----------

dataset = arff.load(open('C:/Users/xhs/Desktop/ECG5000/ECG5000_TEST.arff', 'r'))
data = np.array(dataset['data'])
data =data.astype(np.float64)
datax = data[:,0:140]
datay = data[:,140]
n,m = datax.shape
#-----------data import-------------------
index1=[]
index2=[]
index3=[]
index4=[]
index5=[]
index=[]
for i in range(4500):
    if datay[i]==1:
        index1.append(i)
    elif datay[i]==2:
        index2.append(i)    
    elif datay[i]==3:
        index3.append(i)
    elif datay[i]==4:
        index4.append(i)
    else:
        index5.append(i)
X = np.zeros([110,m+1])
for i in range(22):
    index.append(index1[i])
    index.append(index2[i])
    index.append(index3[i])
    index.append(index4[i])
    index.append(index5[i])
for i in range(110):
    for j in range(m+1):
        X[i][j]=data[index[i]][j]
np.savetxt('newTEST.csv',X , delimiter = ',')  
