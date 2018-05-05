from keras import models
from keras.datasets import mnist
(train_images,train_labels), (test_images,test_labels)=mnist.load_data()
my_slice=train_images[10:100]
print(my_slice.shape)

#14*14 slices in the bottom-right corner of all images
my_slice =train_images[:,14:,14:]
#14*14 slices in the middlw of all image
my_slice=train_images[:,7:-7,7:-7]

#explain dot
#output=relu(dot(w,input)+b)
import numpy as np
a=np.array([1,2,3])
b=np.array([4,5,6])
print(a*b)

a=np.random.rand(2,3)
print(a)
b=np.random.rand(3,4)
print(b)
np.dot(a,b)
print(np.dot(a,b))

#----------------------------
#Reshaping
x=np.array([[0,1],[2,3],[4,5]])
print(x.shape)

x=x.reshape((6,1))
print(x)

x=x.reshape((2,3))
print(x)