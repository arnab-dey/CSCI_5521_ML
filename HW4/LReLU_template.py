from keras import backend as K

def LReLU(x):
    # please think how Keras.tf.where can be used to implement the leaky_relu
    # This is an example for sigmoid activation
    activation_vals = keras.backend.sigmoid(x)
    return activation_vals
