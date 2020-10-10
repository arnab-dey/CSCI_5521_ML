###########################################################################
# Imports
###########################################################################
import os
import numpy as np

###########################################################################
# Function definition
###########################################################################
def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y.T)) ** p

def getGramMat(data, trn_data, pol_degree=3):
    N = data.shape[0]
    N_trn = trn_data.shape[0]
    K = []
    for index in range(N):
        f = (trn_data @ data[index, :])+1.
        K_index = np.power(f, pol_degree)
        K.append(K_index)
    K = np.asarray(K)
    return K.T

def kernPercGD(train_data, train_label, poly_degree):
    if (train_data is None) or (train_label is None):
        return None, None
    # Get number of data
    N, D = train_data.shape

    ###########################################################################
    # Initialization
    ###########################################################################
    alpha = np.zeros((N, 1))
    b = 0
    isConverged = False
    x_train = train_data
    # Calculate gram matrix
    K = getGramMat(x_train, x_train, poly_degree)
    step = 0
    step_limit = 500
    err = True
    while err:
        step += 1
        for sample in range(N):
            trial_prediction = (alpha.T * train_label) @ K[:, sample] + b
            if np.sign(trial_prediction) != train_label[sample]:
                alpha[sample, 0] += 1.
                b += train_label[sample]
        prediction = np.sign(((alpha.T * train_label) @ K).T + b)
        err = (np.sum(prediction[:, 0] != train_label))*100./N
        # print('training err = ', err, '% at step = ', step)
        if err <= 0.1 or step >= step_limit:
            break
    if step < step_limit:
        print('training err = ', err, '%')
    else:
        print('Hit maximum step limit of ', step_limit, ', training err = ', err, '%')
    return alpha, b

