###########################################################################
# Imports
###########################################################################
import numpy as np
import myKNN as mk
import os, sys
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
# Load training data
###########################################################################
trn_data = np.loadtxt(training_data, delimiter=',')
###########################################################################
# Load test data and extract information
###########################################################################
test_data = np.loadtxt(testing_data, delimiter=',')
N_test, D_test = test_data.shape
# Last column represents class label, therefore feature dimension is D_test-1
D_test -= 1
###########################################################################
# Prediction and error check
###########################################################################
for k in k_arr:
    prediction = mk.myKNN(trn_data, test_data, k)
    ###########################################################################
    # test data performance check
    ###########################################################################
    if prediction is not None:
        err = (np.sum(prediction != test_data[:, D_test])) * (100. / N_test)
        print("k = ", k, " error rate = ", err)
    else:
        print("Either training or testing data cannot be located")
