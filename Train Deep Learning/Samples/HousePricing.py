from keras.datasets import  boston_housing
from Samples.LinearLayer import build_model

(train_data,train_target),(test_data,test_target)=boston_housing.load_data()

print(train_data.shape, test_data.shape,train_target)

mean= train_data.mean(axis=0)
train_data -=mean

std=train_data.std(axis=0)
train_data /=std

test_data -=mean
test_data /=std

print(test_data)

build_model(train_data)

