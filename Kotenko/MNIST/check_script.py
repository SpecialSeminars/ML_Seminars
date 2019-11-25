from functions import *
import numpy as np 
import scipy.optimize

regul = 0.1

trainsetsize =50000

X, y = loadData(trainsetsize,"t")

total_theta = np.load("total_theta_60000_set.npy")

np.save("total_theta_"+str(trainsetsize)+"_set" ,total_theta )

yesorno_trainingset = (np.argmax(np.matmul(X,total_theta.T), axis=1)==y).astype(np.float32)
print("training set eff = "+str(np.mean(yesorno_trainingset)))


examsetsize = 10000
X, y = loadData(examsetsize,"e")

yesorno_examset = (np.argmax(np.matmul(X,total_theta.T), axis=1)==y).astype(np.float32)
print("exam set eff = "+str(np.mean(yesorno_examset)))
