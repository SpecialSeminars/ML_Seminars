import numpy as np
import functions 

theta = np.load("total_theta_60000_set.npy")
for i in range(10):
	print(i)
	functions.plotThetaVec(theta[i,:])