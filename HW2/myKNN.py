###########################################################################
# Imports
###########################################################################
import numpy as np
from scipy.spatial import distance as scp_sp_dist
###########################################################################
# Function Definitions
###########################################################################
###########################################################################
# This function implements KNN
###########################################################################
def myKNN(trn_data, test_data, k):
    ###########################################################################
    # Extract information from training data
    ###########################################################################
    N, D = trn_data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    ###########################################################################
    # Extract information from test data
    ###########################################################################
    N_test, D_test = test_data.shape
    # Last column represents class label, therefore feature dimension is D_test-1
    D_test -= 1
    ###########################################################################
    # Calculate Euclidean distances from all training data
    ###########################################################################
    prediction = []
    cdist = scp_sp_dist.cdist(trn_data[:, 0:D], test_data[:, 0:D], 'euclidean')
    # Get the k-nearest neighbors i.e. k minimum distances
    knn_index = (np.argpartition(cdist, k, axis=0))[0:k, :]
    # Predict test samples
    for test_sample in range(knn_index.shape[1]):
        knn_digits = trn_data[knn_index[:, test_sample], D]
        cnt_bin = np.bincount(knn_digits.astype(int))
        prediction.append(cnt_bin.argmax())
    return prediction
