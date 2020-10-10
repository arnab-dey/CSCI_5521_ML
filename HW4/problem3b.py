###########################################################################
# Imports
###########################################################################
import LReLU as lrelu
import numpy as np
x = np.array([1., 2.])
x = np.reshape(x, (2, 1))
test = lrelu.LReLU(x)
print(test)