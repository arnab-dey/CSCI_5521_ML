import numpy as np

def ReadNormalizedOptdigitsDataset(training_filename, validation_filename, test_filename):

    # read files
    training_data = np.loadtxt(training_filename, delimiter=',')
    validation_data = np.loadtxt(validation_filename, delimiter=',')
    test_data = np.loadtxt(test_filename, delimiter=',')

    # prepare training data output
    X_trn = training_data[:, 0:-1]
    y_trn = training_data[:, -1].astype(np.int8)
    
    # prepare validation data output
    X_val = validation_data[:, 0:-1]
    y_val = validation_data[:, -1].astype(np.int8)
    
    # prepare test data output
    X_tst = test_data[:, 0:-1]
    y_tst = test_data[:, -1].astype(np.int8)

    # mean and std
    mu = np.mean(X_trn, axis=0)
    s = np.std(X_trn, axis=0)
    
    # normalize data
    X_trn_norm = (X_trn - np.tile(mu, (np.shape(X_trn)[0], 1))) / np.tile(s, (np.shape(X_trn)[0], 1))
    X_val_norm = (X_val - np.tile(mu, (np.shape(X_val)[0], 1))) / np.tile(s, (np.shape(X_val)[0], 1))
    X_tst_norm = (X_tst - np.tile(mu, (np.shape(X_tst)[0], 1))) / np.tile(s, (np.shape(X_tst)[0], 1))
    
    # fix nan
    X_trn_norm[np.isnan(X_trn_norm)]=0
    X_val_norm[np.isnan(X_val_norm)]=0
    X_tst_norm[np.isnan(X_tst_norm)]=0
    
    return X_trn_norm, y_trn, X_val_norm, y_val, X_tst_norm, y_tst

