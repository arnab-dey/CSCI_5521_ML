###########################################################################
# Imports
###########################################################################
import os
import numpy as np
import normalizeData as nrmd
###########################################################################
# Function definitions
###########################################################################
def getLeakyReluVal(w, x):
    z = w.T @ x
    z[z < 0.] *= 0.01
    return z

def getLeakyReluDerivative(w, x):
    z = w.T @ x
    z[z >= 0.] = 1.
    z[z < 0.] = 0.01
    return z

def getLinearOutputFunc(v, z):
    return v.T @ z

def getOutputSoftmax(o):
    exp_o = np.exp(o)
    sum_exp_o = np.sum(exp_o, axis=0)
    y = exp_o/sum_exp_o
    return y

def forwardPropagation(w, v, x):
    N = x.shape[1]
    z = getLeakyReluVal(w, x)
    # Append one to represent z_0
    z = np.vstack((np.ones((1, N)), z))
    # Compute output
    o = getLinearOutputFunc(v, z)
    y = getOutputSoftmax(o)
    return y, z

def backwardPropagation(y, v, z, w, x, r, eta):
    ###########################################################################
    # Compute hidden to output layer weights updates
    ###########################################################################
    diff = (r - y)
    del_v = eta * (z @ diff.T)
    ###########################################################################
    # Compute input to output layer weights updates
    ###########################################################################
    relu_prime = getLeakyReluDerivative(w, x)
    del_w = eta * (((v[1:, :] @ diff) * relu_prime) @ x.T).T
    return del_v, del_w

def MLPTrain(training_data, validation_data, K, H):
    ###########################################################################
    # Load training data and extract information
    ###########################################################################
    trn_data = training_data
    N, D = trn_data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D -= 1
    # Append ones to represent x_0
    x_nd = np.hstack((np.ones((N, 1)), trn_data))
    ###########################################################################
    # Initialize weights
    ###########################################################################
    # Random number within [-0.01,0.01]
    w_dh = ((np.random.rand(D+1, H)*2)-1)*0.01
    v_hk = ((np.random.rand(H+1, K)*2)-1)*0.01

    ###########################################################################
    # Define parameters
    ###########################################################################
    isConverged = False
    # Epoch count and bound to stop iteration
    num_epoch = 0
    epoch_bound = 120
    # Learning rate
    eta = 0.001
    # Momentum
    momentum = 0.6
    prev_del_w_dh = 0.
    prev_del_v_hk = 0.
    prev_loss = 0.
    loss = 0.
    loss_change_thr = 0.01
    patience_count = 0
    patience_limit = 2
    num_mismatch = 0.
    while(isConverged == False):
        prev_loss = loss
        loss = 0.
        num_epoch += 1
        np.random.shuffle(x_nd)
        for sample in range(N):
            x_t = x_nd[sample, 0:-1]
            x_t = np.reshape(x_t, (D + 1, 1))
            r_k = np.zeros(K)
            # Make r_k index equal to 1 corresponding to sample label
            r_k[int(x_nd[sample, -1])] = 1.
            r_k = np.reshape(r_k, (K, 1))
            ###########################################################################
            # Forward propagation
            ###########################################################################
            y_k, z_h = forwardPropagation(w_dh, v_hk, x_t)
            if (np.isnan(y_k).any()):
                print('Encountered nan, epoch = ', num_epoch, '. Returning...')
                return None, None, None
            ###########################################################################
            # Backward propagation
            ###########################################################################
            del_v_hk, del_w_dh = backwardPropagation(y_k, v_hk, z_h, w_dh, x_t, r_k, eta)
            ###########################################################################
            # Update all layers weights
            ###########################################################################
            w_dh += del_w_dh + momentum*prev_del_w_dh
            prev_del_w_dh = del_w_dh
            v_hk += del_v_hk + momentum*prev_del_v_hk
            prev_del_v_hk = del_v_hk
            ###########################################################################
            # Calculate loss
            ###########################################################################
            loss -= r_k.T @ np.log(y_k)
        ###########################################################################
        # Check for convergence
        ###########################################################################
        if (num_epoch == epoch_bound):
            isConverged = True
        if (np.abs(loss-prev_loss) <= loss_change_thr):
            patience_count += 1
            if (patience_count == patience_limit):
                isConverged = True

    ###########################################################################
    # Calculate error rate on training data
    ###########################################################################
    y_kn, z_hn = forwardPropagation(w_dh, v_hk, x_nd[:, 0:-1].T)
    y_nk = y_kn.T
    y_predicted = np.argmax(y_nk, axis=1)
    err_trn = (np.sum(y_predicted != x_nd[:, -1])) * (100. / N)
    print('Error on training data = ', err_trn, ' with hidden units = ', H)
    ###########################################################################
    # Calculate error rate on validation data
    ###########################################################################
    val_data = validation_data
    N_val, D_val = val_data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D_val -= 1
    # Append ones to represent x_0
    xVal_nd = np.hstack((np.ones((N_val, 1)), val_data))
    ###########################################################################
    # Compute hidden layer output
    ###########################################################################
    y_kn, z_hn = forwardPropagation(w_dh, v_hk, xVal_nd[:, 0:-1].T)
    y_nk = y_kn.T
    z_nh = z_hn.T
    if (np.isnan(y_nk).any()):
        print('Encountered nan in validation data. Returning...')
        return None, None, None
    ###########################################################################
    # Check error rate
    ###########################################################################
    y_predicted = np.argmax(y_nk, axis=1)
    err_val = (np.sum(y_predicted != xVal_nd[:, -1])) * (100. / N_val)
    print('Error on validation data = ', err_val, ' with hidden units = ', H)
    return z_nh[: 1:], w_dh, v_hk








