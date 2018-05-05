from keras import models
from keras.datasets import mnist
(train_images,train_labels), (test_images,test_labels)=mnist.load_data()
print (train_images.shape)
print(len(train_labels))
print(train_labels)
#--------------------------------------
#show a image
digit=train_images[10]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
#--------------------------------------
#Presprocessing
train_images =train_images.reshape(60000,28*28)
train_images =train_images.astype('float32')/255
print(train_images)
test_images =test_images.reshape(10000,28*28)
test_images =test_images.astype('float32')/255
print(train_images)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels=to_categorical(test_labels)



from keras import models
from keras import layers

network = models.Sequential()
#dense layer  or fully conectly === all pixel has edge to all nuron
network.add(layers.Dense(512,activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#آموزش شبکه
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

network.fit(train_images,train_labels,epochs=5,batch_size=128)

#overfiting test 
test_loss,test_acc=network.evaluate(test_images,test_labels)
print('test_acc:',test_acc)