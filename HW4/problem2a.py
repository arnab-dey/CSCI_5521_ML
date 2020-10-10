###########################################################################
# Imports
###########################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import MLPtrain as mlp
import MLPtest as mlpTest
import ReadNormalizedOptdigitsDataset as normopt # Not using this as modified version is used below
import normalizeData as nrmd # Modified version where features with zero variances are removed
###########################################################################
# Script
###########################################################################
H_arr = [3, 6, 9, 12, 15, 18]
train_data = './optdigits_train.txt'
val_data = './optdigits_valid.txt'
test_data = './optdigits_test.txt'
# Uncomment this if you wish to retain features with zero variances
# X_trn, y_trn, X_val, y_val, X_tst, y_tst = normopt.ReadNormalizedOptdigitsDataset(train_data, val_data, test_data)
# Uncomment this if you wish to remove features with zero variances
X_trn, y_trn, X_val, y_val, X_tst, y_tst = nrmd.normalizeDataset(train_data, val_data, test_data)
N_trn = y_trn.shape[0]
N_val = y_val.shape[0]
N_tst = y_tst.shape[0]
y_trn = np.reshape(y_trn, (N_trn, 1))
y_val = np.reshape(y_val, (N_val, 1))
y_tst = np.reshape(y_tst, (N_tst, 1))
norm_train_data = np.hstack((X_trn, y_trn))
norm_val_data = np.hstack((X_val, y_val))
norm_test_data = np.hstack((X_tst, y_tst))

trn_err_arr = [0]*len(H_arr)
val_err_arr = [0]*len(H_arr)
w_arr = [None]*len(H_arr)
v_arr = [None]*len(H_arr)

z, w, v = None, None, None

for index in range(len(H_arr)):
    z, w, v = mlp.MLPTrain(norm_train_data, norm_val_data, 10, H_arr[index])
    if (z is not None) and (w is not None) and (v is not None):
        ###########################################################################
        # Calculate error rate on training set
        ###########################################################################
        x_trn = np.hstack((np.ones((N_trn, 1)), norm_train_data))
        y_kn_trn, z_hn_trn = mlp.forwardPropagation(w, v, x_trn[:, 0:-1].T)
        y_nk_trn = y_kn_trn.T
        y_predicted_trn = np.argmax(y_nk_trn, axis=1)
        err_trn = (np.sum(y_predicted_trn != norm_train_data[:, -1])) * (100. / N_trn)
        ###########################################################################
        # Calculate error rate on validation set
        ###########################################################################
        x_val = np.hstack((np.ones((N_val, 1)), norm_val_data))
        y_kn_val, z_hn_val = mlp.forwardPropagation(w, v, x_val[:, 0:-1].T)
        y_nk_val = y_kn_val.T
        y_predicted_val = np.argmax(y_nk_val, axis=1)
        err_val = (np.sum(y_predicted_val != norm_val_data[:, -1])) * (100. / N_val)
        ###########################################################################
        # Store error rate
        ###########################################################################
        trn_err_arr[index] = err_trn
        val_err_arr[index] = err_val
        w_arr[index] = w
        v_arr[index] = v
    else:
        print('MLPTrain returned None')
###########################################################################
# Plot error rate Vs. hidden units
###########################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5')
ax.grid(which='minor', linestyle="-.", linewidth='0.5')
plt.plot(H_arr, trn_err_arr, 'r', marker='+', label='Training data')
plt.plot(H_arr, val_err_arr, 'g', marker='*', label='Validation data')
plt.legend()
plt.xlabel('No. of hidden units')
plt.ylabel('Error (%)')
plt.draw()
###########################################################################
# Test on test data
###########################################################################
hidden_units = H_arr[4]
w_chosen = w_arr[4]
v_chosen = v_arr[4]
print('Chosen hidden units = ', hidden_units)
z_tst = mlpTest.MLPtest(norm_test_data, w_chosen, v_chosen)
plt.show()

