from functions import *
import numpy as np 
import scipy.optimize

regul = 0.1

trainsetsize =50000

X, y = loadData(trainsetsize,"t")

theta_zero = np.zeros( X.shape[1], dtype=np.float32 )

total_theta = np.array([])
for number in range(10):
	result= scipy.optimize.minimize(costFunction,theta_zero,args=(X, (y==number).astype(np.float32), regul ),method="BFGS",jac=True)
	res_theta = result.x
	if number==0:
		total_theta=res_theta
	else:
		total_theta = np.c_[total_theta, res_theta]

total_theta=total_theta.T
np.save("total_theta_"+str(trainsetsize)+"_set" ,total_theta )

yesorno_trainingset = (np.argmax(np.matmul(X,total_theta.T), axis=1)==y).astype(np.float32)
print("training set eff = "+str(np.mean(yesorno_trainingset)))


examsetsize = 10000
X, y = loadData(examsetsize,"e")

yesorno_examset = (np.argmax(np.matmul(X,total_theta.T), axis=1)==y).astype(np.float32)
print("exam set eff = "+str(np.mean(yesorno_examset)))
