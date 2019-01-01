#WE IMPORT REQUIRED FUNCTIONS AND LIBRARIES REQUIRED FOR MAKING THE NETWORK
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D 

#IMPORT THE MNIST DATABASE
from keras.datasets import mnist
#EXTRACTING IMAGES FROM MNIST DATASET
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_TRAIN HAS SHAPE <60000,28,28> i.e 60000 images which are 28x28 in size
print("X_train original shape", X_train.shape)
# y_TRAIN HAS SHAPE <60000,> to store results of classification 60000 images
print("y_train original shape", y_train.shape)
# X_Test HAS SHAPE <10000,28,28> i.e 10000 images which are 28x28 in size
print("X_test original shape", X_test.shape)
# y_TEST HAS SHAPE <10000,> to store results of classification 10000 images
print("y_test original shape", y_test.shape)

#for image processing input should be in a specific format i.e. (batch,height,width,channels)
#batch=60000 for x train and 10000  fo x test
#since we are passing greyscale images channels is  1
X_train =(X_train.reshape(X_train.shape[0], 28, 28, 1)).astype('float32')
X_test =(X_test.reshape(X_test.shape[0], 28, 28, 1)).astype('float32')

#We now normalize the data to a value between [0,1] instead of [0,255] which helps iin training
X_train/=255
X_test/=255

#we know that the output must be 10 classes(0-9 digits)
Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)
#we create a binary matrix with 1 and 0s where 1 represents the number and 0 mens number is not there

#here we start the model using sequential
model = Sequential()

#we add a conv2D filter that acts on an area of 3x3 pixels and applies 32 filters
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
#batch normalization helps prevent overfitting and reduces time taken to for program to run
BatchNormalization(axis=-1)
#activating using RelU- rectifid linear unit
model.add(Activation('relu'))

#another convolution layer
model.add(Conv2D(32, (3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

#pooling layer to prevent overfitting and enable the neurons to learn in greater detail and reduces training time
model.add(MaxPooling2D(pool_size=(2,2)))

#the values obtained after applying all the filters must be flattened before being sent to our dense layer for #classification
model.add(Flatten())

# Fully connected layer or dense layer(hidden layer)
BatchNormalization()
model.add(Dense(400))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))

#output layer that does the classification
model.add(Dense(10))
#classification function
model.add(Activation('softmax'))

#loss calculation
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

#now we fit the data in to our model and run it
model.fit(X_train, Y_train, epochs=2, batch_size=500,verbose=2)

#we test our model by evaluating it with the test data sets
score = model.evaluate(X_test, Y_test)
print()
print('Test accuracy: ', score[1])