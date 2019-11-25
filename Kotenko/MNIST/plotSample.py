import gzip
import matplotlib.pyplot as plt


f_im = gzip.open('train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 100

import numpy as np
f_im.read(16)
buf = f_im.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

f_lb = gzip.open('train-labels-idx1-ubyte.gz','r')
f_lb.read(8)

num =0 
fig, ax = plt.subplots(nrows=5, ncols=5)
# plt.axis('off')
for i in range(25):
	print( np.frombuffer(f_lb.read(1), dtype=np.uint8).astype(np.int64) )
for row in ax:
	for col in row:
		image = np.asarray(data[num]).squeeze()
		col.imshow(image)
		num+=1

plt.show()