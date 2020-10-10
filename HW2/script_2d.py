###########################################################################
# Imports
###########################################################################
import numpy as np
import os, sys
import myLDA as ml
import myKNN as mk
###########################################################################
# Training, Test dataset location
###########################################################################
k_arr = [1, 3, 5]
L = [2, 4, 9]
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
# LDA
###########################################################################
for dim in L:
    w, e_val = ml.myLDA(trn_data, dim)
    ###########################################################################
    # Project training and test data
    ###########################################################################
    reduced_trn_data = trn_data[:, 0:D] @ w
    reduced_test_data = test_data[:, 0:D_test] @ w
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
            print("L = ", dim, " k = ", k, " error rate = ", err)
        else:
            print("Either training or testing data cannot be located")