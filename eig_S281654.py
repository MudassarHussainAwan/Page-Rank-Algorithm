#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import all the needed libraries

import numpy as np
import pandas as pd
import csv
from scipy.sparse import coo_matrix
from scipy.sparse import linalg as LA
import matplotlib.pyplot as plt
#-------------------------------------------------------------------

#Load the dataSet
X = [ [] , [] ]
with open("graph.txt") as fp:
    reader = csv.reader(fp)
    for row in reader:
        e,s = [int(s) for s in row[0].split() if s.isdigit()]
        X[0].append(e)
        X[1].append(s)
#-------------------------------------------------------------------

#Adacency Matrix (G)
N= 2500
n=len(X[1])
V_e = np.array(X[0])
V_s = np.array(X[1])
V = np.ones(n)
G = coo_matrix((V, (V_e, V_s)), shape=(N, N))

#Visualization of G
plt.spy(G,markersize=0.3)
plt.title('figure1 : Adjacency Matrix',y=-0.1)
#--------------------------------------------------------------------
r = G.sum(axis=1) #sum the rows -> in_degree
c = G.sum(axis=0) #sum the columns -> out_degree
#--------------------------------------------------------------------

#Hyperlink Matrix

#set the weight:
m = np.zeros(n)
for i,col in enumerate(X[1]):
    w=c.A[0][col]
    m[i]=1/w
  

M = coo_matrix((m, (V_e, V_s)),shape = (N,N))
plt.figure()
plt.spy(M,markersize = 0.1)
plt.title('figure2 : Hyperlink Matrix',y=-0.1)

s1 = M.sum(axis=0) #sum up the columns 
print(f"Summing up the columns of Hyperlink matrix{s1.A[0]}")

#----------------------------------------------------------------------

#Modified Hyperlink Matrix

#Acces the entries corresponding to cj=0 and replace them by 1/N
import scipy as sp
Mtilde = M.todense()
Mtilde[:,c.A[0]==0] = 1/N
Mtild = sp.sparse.coo_matrix(Mtilde)
plt.figure()
plt.spy(Mtilde,markersize = 0.01)
plt.title('figure3 : Modified Hyperlink Matrix',y=-0.1)

s2 = Mtild.sum(axis=0) #sum up the columns 
print(f"Summing up the columns of Modified hyperlink matrix{s2.A[0]}")
#---------------------------------------------------------------------

# The power method

def power_method(A):
    
    #initialization
    tol = 1e-06
    x0 = np.ones((N,1))
    nmax = 100
    
    x0 = x0/np.linalg.norm(x0)
    pro = np.array(A*x0)
    lamda = np.dot(np.transpose(x0),pro)
    err = tol*abs(lamda)+1
    iter = 0

    while (err > tol*abs ( lamda )) & (iter <= nmax):
        x = pro;
        x = x/np.linalg.norm(x)
        pro = np.array(A*x)
        lamdanew = np.dot(np.transpose(x),pro)
        err = abs(lamdanew-lamda)
        lamda = lamdanew 
        iter = iter + 1
    
    return lamda ,x 
#-------------------------------------------------------------------------

# Computation of largest eigen Value of modified hyperlink matrix

l,x=power_method(Mtild)
w,v = LA.eigs(Mtild,k=1)
abs(w[0]-l)/abs(w)
#Calculating relative error
err = abs(w[0]-l)/abs(w)
print(f"relative absolute error modified hyperlink matrix: {err}")
#--------------------------------------------------------------------------

# The Google Matrix (A)

alpha =0.85
sigma = (1-alpha)/N
d = np.zeros(n)
for i,col in enumerate(X[1]):
    w=c.A[0][col]
    d[i]= (alpha/w) +sigma
A = coo_matrix((d, (V_e, V_s)),shape = (N,N))
A= A.todense()
A[:,c.A[0]==0] = 1/N
A[A==0]=sigma

s3 = A.sum(axis=0) #sum up the columns
print(f"Summing up the columns of Google matrix{s3.A[0]}")

# Computaation of eigen values
l,x=power_method(A)
w,v = LA.eigs(A,k=2)
print(l,w)
# relative error
r_err = abs(l-w[0].real)/abs(w[0].real)
print(f"Relative error eigen value of A:{r_err}")
#------------------------------------------------------------------------------
#Modified Power methode:

def modified_power_method(A):
    
    #initialization
    tol = 1e-06
    x0 = np.ones((N,1))
    x1= np.random.rand(N,1)
    nmax = 100

    x0 = x0/np.linalg.norm(x0)
    x1 = x1 - np.dot(np.transpose(x0),x1)*x0
    x1 = x1/np.linalg.norm(x1)
    
    pro0 = np.array(A*x0)
    pro1 = np.array(A*x1)
    
    lamda0 = np.dot(np.transpose(x0),pro0)
    lamda1 = np.dot(np.transpose(x1),pro1)
    err = tol*abs(lamda1)+1
    iter = 0

    while (err > tol*abs ( lamda1 )) & (iter <= nmax):
        x0 = pro0;
        x1 = pro1;
       
        x0 = x0/np.linalg.norm(x0)  
        x1 = x1 - np.dot(np.transpose(x0),x1)*x0
        x1 = x1/np.linalg.norm(x1)
        
        pro0 = np.array(A*x0)
        pro1 = np.array(A*x1)
        
        lamdanew0 = np.dot(np.transpose(x0),pro0)
        lamdanew1 = np.dot(np.transpose(x1),pro1)
        
        err = abs(lamdanew1-lamda1)
        lamda1 = lamdanew1
        iter = iter + 1
    
    return lamda0,lamda1 ,x0,x1 

lam0,lam1,x0,x1 = modified_power_method(Mtilde)
print(f'Second eigenvalue of Mtilde:{lam1}')

#--------------------------------------------------------------------------------
# Diagonalization Matrix 

D = np.zeros((N,N))
for i,cj in enumerate(c.A[0]):
    if cj!=0:
        D[i,i] = 1/cj 
z= np.zeros((N,1))

for i,cj in enumerate(c.A[0]):
    if cj!=0:
        z[i] = sigma
    else:
        z[i]= 1/n
        
D = sp.sparse.coo_matrix(D)
plt.figure()
plt.spy(D,markersize = 0.1)
plt.title('figure4 : Diagonal Matrix',y=-0.1)
#-------------------------------------------------------------------------------
#page rank bar graph
x=x[:,0]  # conversion from 1D matrix to a vector
axis=np.arange(x.shape[0])
plt.figure()
plt.bar(axis,x,width=10)
plt.title('page rank',y=-0.2)
#-------------------------------------------------------------------------------

#top 12 ranked

y=x.argsort()[::-1][:n]
axis2=np.arange(y.shape[0])
top_pages=y[0:12]
top_rank = x[top_pages]   #page rank
in_d = r.A[top_pages]     #in degrees
in_d = in_d[:,0]          #out_degrees
out_d = c.A[0][top_pages]
d = {'rank': top_rank, 'in degree': in_d,'out_degree':out_d}
# Dataframe
best_rank = pd.DataFrame(d,index=top_pages)
best_rank.index.set_names("Page No.")
best_rank

