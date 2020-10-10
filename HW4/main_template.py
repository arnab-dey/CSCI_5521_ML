import os

# Only modify the code in network architecture section

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from datetime import datetime
# You need to implement your own leaky relu by yourself
from LReLU import LReLU
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset

# You can play with these parameters
batch_size = 64
epochs = 20

# the number of classes
num_classes = 10

# input image dimensions
img_rows, img_cols = 8, 8

# load and normalize data
x_train, y_train, x_valid, y_valid, x_test, y_test = ReadNormalizedOptdigitsDataset('optdigits_train.txt',
                                                                                    'optdigits_valid.txt',
                                                                                    'optdigits_test.txt')

# convert data format to channel last format
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

######################## Modify the codes below ######################
##                  Network Architecture Definition                 ##
######################## Modify the codes below ######################

model = Sequential()
# structure 1 
# model.add(Conv2D(1, kernel_size=(4, 4),
#                  activation='linear',
#                  input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Activation(LReLU)) # use your leaky relu function
# model.add(Dense(num_classes, activation='softmax'))

# structure 2 (add your own network structure below, more details about
# functions please see keras documentation)


# store results and use tensorborad to visualize the training loss
logdir = "./logs/HW4_" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[tensorboard_callback],
          )

# do the evaluation on test data set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
