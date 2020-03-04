# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:02:50 2018

@author: zainu
"""

import bonnerlib2 as ab
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
import pickle 
import sklearn.linear_model as lin
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import randn
from sklearn.neural_network  import MLPClassifier


print("Question 1: \n")
print("----a-----")
def genData(mu0,mu1,Sigma0,Sigma1,N): 
    t0 = np.full((N,1),0)
    t1 = np.full((N,1),1)
    t = np.append(t0,t1,axis = 0)
    x0 = np.random.multivariate_normal(mu0,Sigma0,N)
    x1 = np.random.multivariate_normal(mu1,Sigma1,N)
    x = np.append(x0,x1,axis=0)
    suff_x, suff_t = shuffle(x,t)
    return suff_x,suff_t
    
mu0 = [0,-1]
mu1 = [-1,1]
Sigma0 = np.matrix('2.0 0.5; 0.5 1.0')
Sigma1 = np.matrix('1.0 -1.0; -1.0 2.0')
x, t=  genData(mu0,mu1,Sigma0,Sigma1, 1000)
Xtest, Ttest=  genData(mu0,mu1,Sigma0,Sigma1, 10000)

def plotmaker(x,t,s):
    plt.title(s)
    for i in range(len(t)):
        if(t[i] == 0):
            plt.scatter(x[i,0],x[i,1],s=2,c='r')
        else: 
            plt.scatter(x[i,0],x[i,1],s=2,c='b')
    plt.xlim(-5,6)
    plt.ylim(-5,6)
    return plt

print("----b-----\n")

mf = MLPClassifier(solver='sgd',activation='tanh',learning_rate_init=0.01,max_iter=10000,hidden_layer_sizes=(1,))

mf.fit(x,t)

p = plotmaker(x,t,"Question 1(b): Neural net with 1 hidden unit.")
ab.dfContour(mf)
p.show()

print("----c-----\n")


best_acc = 0
curr_acc = 0
best_mf_2 = None

for i in range(9):
    mf = MLPClassifier(solver='sgd',activation='tanh',learning_rate_init=0.01,max_iter=10000,hidden_layer_sizes=(2,))
    mf.fit(x,t)
    curr_acc = mf.score(Xtest,Ttest)
    if(curr_acc > best_acc):
        best_acc = curr_acc
        best_mf_2 = mf
    print curr_acc
    p.subplot(3,3,i+1)
    p = plotmaker(x,t,"")
    ab.dfContour(mf)
p.suptitle("Question 1(c): Neural nets with 2 hidden units.")
p.show()
p = plotmaker(x,t,"")
ab.dfContour(best_mf_2)
p.title("Question 1(c):Best neural net with 2 hidden units.")
p.show()
print "best acc = ",best_acc

best_acc = 0
curr_acc = 0
best_mf_3 = None
print("----d-----\n")
for i in range(9):
    mf = MLPClassifier(solver='sgd',activation='tanh',learning_rate_init=0.01,max_iter=10000,hidden_layer_sizes=(3,))
    mf.fit(x,t)
    curr_acc = mf.score(Xtest,Ttest)
    if(curr_acc > best_acc):
        best_acc = curr_acc
        best_mf_3 = mf
    print curr_acc    
    p.subplot(3,3,i+1)
    p = plotmaker(x,t,"")
    ab.dfContour(mf)
p.suptitle("Question 1(d): Neural nets with 3 hidden units.")
p.show()
p = plotmaker(x,t,"")
ab.dfContour(best_mf_3)
p.title("Question 1(d):Best neural net with 3 hidden units.")
p.show()
print "best acc = ",best_acc

print("----e-----\n")
best_acc = 0
curr_acc = 0
best_mf_4 = None
for i in range(9):
    mf = MLPClassifier(solver='sgd',activation='tanh',learning_rate_init=0.01,max_iter=10000,hidden_layer_sizes=(4,))
    mf.fit(x,t)
    curr_acc = mf.score(Xtest,Ttest)
    if(curr_acc > best_acc):
        best_acc = curr_acc
        best_mf_4 = mf
    print curr_acc
    p.subplot(3,3,i+1)
    p = plotmaker(x,t,"")
    ab.dfContour(mf)
p.suptitle("Question 1(e): Neural nets with 4 hidden units.")
p.show()
p = plotmaker(x,t,"")
ab.dfContour(best_mf_4)
p.title("Question 1(e):Best neural net with 4 hidden units.")
p.show()
print "best acc = ",best_acc

print("----g-----\n")
w = best_mf_3.coefs_[0]
w0 = best_mf_3.intercepts_[0]
xmin = -5
xmax = 6

def plotB(w,w0):
    y1 = -(w0+w[0]*xmin)/w[1]
    y2 = -(w0+w[0]*xmax)/w[1]
    p.plot([xmin,xmax],[y1,y2],color= 'black',linestyle='--')
p = plotmaker(x,t,"")
ab.dfContour(best_mf_3)
plotB(w,w0)
p.title("Question 1(g): Decision boundaries for 3 hidden units")
p.show()

print("----h-----\n")
w = best_mf_2.coefs_[0]
w0 = best_mf_2.intercepts_[0]
xmin = -5
xmax = 6

p = plotmaker(x,t,"")
ab.dfContour(best_mf_2)
plotB(w,w0)
p.title("Question 1(h): Decision boundaries for 2 hidden units")
p.show()

print("----i-----\n")
w = best_mf_4.coefs_[0]
w0 = best_mf_4.intercepts_[0]
xmin = -5
xmax = 6

p = plotmaker(x,t,"")
ab.dfContour(best_mf_4)
plotB(w,w0)
p.title("Question 1(i): Decision boundaries for 4 hidden units")
p.show()


#----j-----
# "The more the units used for neural network, the more precise the accuracy, 
# giving better representation of classifier


print("----k-----\n")
w1 = best_mf_3.coefs_[0]
w0 = best_mf_3.intercepts_[0]


M = 1000
probability = best_mf_3.predict_proba(Xtest)
PN_1 = probability[:,0]
z = probability[:,1]
t = np.linspace(0,1,M)


 

pscore = 0
pres = []
recall = []
for i in range(1000):
    tpos = 0.0
    fpos = 0.0
    tneg = 0.0
    fneg = 0.0
    output = []
    for a in range(1000):
        if z[a] > t[i]:
            output.append(1)
        else:
            output.append(0)
    for b in range(1000):
        if output[b] == 1:
            if output[b] == Ttest[b]:
                tpos = tpos + 1
            else:
                fpos = fpos + 1
        elif output[b] == 0:
            if output[b] == Ttest[b]:
                tneg = tneg +1
            else:
                fneg = fneg+1
    if (tpos+fpos) == 0:
        pscore = 1
    else:
        pscore = tpos/(tpos+fpos)
    recallP = tpos / (tpos+fneg)
    pres.append(pscore)
    recall.append(recallP)
    
plt.plot(recall,pres,c='blue')
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("Question 1(k): precision/recall curve")
plt.show()

print("----l-----\n")
total = 0
for i in range(1,len(recall)):
    total = total + (recall[i-1] - recall[i]) * pres[i]
    
print "The area under precision/recall curve", total

print("Question 3: I dont know")