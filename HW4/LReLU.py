###########################################################################
# Imports
###########################################################################
from keras import backend as K
###########################################################################
# Function definitions
###########################################################################
def LReLU(x):
    return K.tf.where(x >= 0, x, 0.01 * x)
