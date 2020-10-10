###########################################################################
# Imports
###########################################################################
import numpy as np
import os, sys
import myPCA as mp
import myKNN as mk
###########################################################################
# Training, Test dataset location
###########################################################################
k_arr = [1, 3, 5, 7]
training_data = "./optdigits_train.txt"
testing_data = "./optdigits_test.txt"
###########################################################################
# Check if input files are present in the location
###########################################################################
if not os.path.isfile(training_data):
    sys.exit("Training data file can't be located")
if not os.path.isfile(testing_data):
    sys.exit("testing data file can't be located")
###########################################################################
# Load training data and extract information
###########################################################################
trn_data = np.loadtxt(training_data, delimiter=',')
N, D = trn_data.shape
# Last column represents class label, therefore feature dimension is D-1
D -= 1
###########################################################################
# Load test data and extract information
###########################################################################
test_data = np.loadtxt(testing_data, delimiter=',')
N_test, D_test = test_data.shape
# Last column represents class label, therefore feature dimension is D_test-1
D_test -= 1
###########################################################################
# PCA
###########################################################################
# We see from the proportion of variance graph that minimum number of
# eigen vectors that explain 90% of the variance is 20. Therefore,
# we use reduced dimension of the data to be 20
reduced_dimension = 20
w, e_val = mp.myPCA(trn_data, reduced_dimension)
###########################################################################
# Project training and test data
###########################################################################
# subtract mean before projection
mu_trn = np.mean(trn_data[:, 0:D], axis=0)
mu_test = np.mean(test_data[:, 0:D_test], axis=0)
centered_trn = trn_data[:, 0:D] - mu_trn
centered_test = test_data[:, 0:D_test] - mu_test
# Apply projection
reduced_trn_data = centered_trn[:, 0:D] @ w
reduced_test_data = centered_test[:, 0:D] @ w
D_reduced = reduced_trn_data.shape[1]
D_test_reduced = reduced_test_data.shape[1]
reduced_trn_data = np.append(reduced_trn_data, trn_data[:, D].reshape((N, 1)), axis=1)
reduced_test_data = np.append(reduced_test_data, test_data[:, D_test].reshape((N_test, 1)), axis=1)
###########################################################################
# Prediction and error check on reduced dimension data
###########################################################################
for k in k_arr:
    prediction = mk.myKNN(reduced_trn_data, reduced_test_data, k)
    ###########################################################################
    # test data performance check
    ###########################################################################
    if prediction is not None:
        err = (np.sum(prediction != reduced_test_data[:, D_test_reduced])) * (100. / N_test)
        print("k = ", k, " error rate = ", err)
    else:
        print("Either training or testing data cannot be located")