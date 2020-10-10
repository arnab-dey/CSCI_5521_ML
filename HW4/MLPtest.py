###########################################################################
# Imports
###########################################################################
import os
import numpy as np
import MLPtrain as mlp
###########################################################################
# Function definitions
###########################################################################
def MLPtest(test_data, w, v):
    N, D = test_data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    # Append ones to represent x_0
    x_nd_tst = np.hstack((np.ones((N, 1)), test_data))
    ###########################################################################
    # Test on test data
    ###########################################################################
    H = w.shape[1]
    y_kn_tst, z_hn_tst = mlp.forwardPropagation(w, v, x_nd_tst[:, 0:-1].T)
    y_nk_tst = y_kn_tst.T
    z_nh_tst = z_hn_tst.T
    y_predicted_tst = np.argmax(y_nk_tst, axis=1)
    err_tst = (np.sum(y_predicted_tst != x_nd_tst[:, -1])) * (100. / N)
    print('Error on test data = ', err_tst, ' with hidden units = ', H)
    return z_nh_tst[:, 1:]