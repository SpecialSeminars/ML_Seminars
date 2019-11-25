import gzip
import numpy as np


def loadData(num_images,label): 
	image_size = 28
	if label == "t":
		imfile = 'train-images-idx3-ubyte.gz'
		lbfile = 'train-labels-idx1-ubyte.gz'
	if label == "e":
		imfile = 't10k-images-idx3-ubyte.gz'
		lbfile = 't10k-labels-idx1-ubyte.gz'
	f_im = gzip.open(imfile,'r')
	f_im.read(16)
	buf = f_im.read(image_size * image_size * num_images)
	data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	data = data.reshape(num_images, image_size*image_size, 1)
	data = data.squeeze()/np.float32(256.)
	data = np.concatenate( (np.ones( (num_images,1),dtype=np.float32) , data), axis=1)

	f_lb = gzip.open(lbfile,'r')
	f_lb.read(8)
	buf = f_lb.read(num_images)
	result = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	return data, result

def sigmoid(x):
	return np.float32(1.0)/( np.float32(1.0) + np.exp(-x,dtype=np.float32) )

def costFunction(theta,X,y,lamb):
	m = y.size
	n = X.shape[1]
	J = np.float32(0.0)
	axil_vect = np.ones(n, dtype=np.float32)
	axil_vect[0] = np.float32(0.0)
	hyp = sigmoid(X.dot(theta))
	J+= np.sum( -y*np.log(hyp,dtype=np.float32) - (1-y)*np.log(1-hyp,dtype=np.float32) )/m + np.sum( axil_vect*theta*theta )*lamb/(2*m)
	grad = (X.T).dot(hyp-y)/m+axil_vect*theta*lamb/m
	return J, grad

def plotThetaVec(theta):
	import matplotlib.pyplot as plt
	theta = np.array(theta)
	image_size = 28
	theta = np.delete(theta,0)
	theta = theta.reshape(image_size,image_size)
	image = np.asarray(theta).squeeze()
	plt.imshow(image)
	plt.show()