from keras.datasets import reuters
from keras.utils.np_utils import  to_categorical
import numpy as np

import sys
for p in sys.path:
    print(p)

(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=1000)
print(len(train_data))
print(len(test_data))

from Samples.MSTools import MLTools as ms
x_train=ms.vectorize_sequences(train_data)
x_test=ms.vectorize_sequences(test_data)
print(train_data)


one_hot_train_labels= to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

from keras import models
from keras import layers

models = models.Sequential()
models.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
models.add(layers.Dense(64,activation='relu'))
models.add(layers.Dense(46,activation='softmax'))

models.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#create evaluation entity
x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

#history=models.fit(partial_x_train,partial_y_train,epochs=20, batch_size=512,validation_data=(x_val,y_val))

#best epoc is 9
#retrain
history=models.fit(x_train,one_hot_train_labels,epochs=9,batch_size=512)
results=models.evaluate(x_test,one_hot_test_labels)



print(results)
#check Ranom Accuracy withou Create Model


#check for real data
predictions=models.predict(x_test)
#export of network size - size of classifaction labels
print(predictions[0].shape)
#sum of predit is 1
print(np.sum(predictions[0]))
#best predit for topic label
print(np.argmax(predictions[0]))
