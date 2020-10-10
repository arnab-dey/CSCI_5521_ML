###########################################################################
# Imports
###########################################################################
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import MLPtrain as mlp
import MLPtest as mlpTest
import normalizeData as nrmd
from sklearn.decomposition import PCA
###########################################################################
# Function to plot data with different colors
###########################################################################
def plot_data(data, num_annotation):
    N_plot, D_plot = data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D_plot -= 1
    ###########################################################################
    # Configure axis and grid
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')
    ax.set_xlim(np.min(data[:, 0:D_plot]), np.max(data[:, 0:D_plot]))
    ###########################################################################
    # Plot each digit with different colors
    ###########################################################################
    colorset = ['darkolivegreen', 'brown', 'coral', 'crimson', 'purple', 'gold', 'blue', 'magenta', 'orangered', 'chocolate']
    n_digit = int(np.max(data[:, D_plot]))
    num_class = n_digit + 1
    for digit in range(num_class):
        class_data = data[data[:, D_plot] == np.asarray(digit)]
        clr = colorset[int(digit)]
        plt.scatter(class_data[:, 0], class_data[:, 1], color=clr, marker="*", label=str(int(digit)))

    for points in range(num_annotation):
        digit = data[points, D_plot]
        clr = colorset[int(digit)]
        plt.annotate(str(int(digit)),
                     (data[points, 0], data[points, 1]),
                     textcoords='offset points',
                     xytext=(0, 0),
                     ha='center', color='black')
    plt.title('Hidden representation with 2 principal components')
    plt.xlabel('Hidden 1')
    plt.ylabel('Hidden 2')
    plt.legend()
    plt.show()
###########################################################################
# Function to plot 3D data with different colors
###########################################################################
def plot_data3D(data, num_annotation):
    N_plot, D_plot = data.shape
    # Last column represents class label, therefore feature dimension is D-1
    D_plot -= 1
    ###########################################################################
    # Configure axis and grid
    ###########################################################################
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5')
    ax.grid(which='minor', linestyle="-.", linewidth='0.5')
    ax.set_xlim(np.min(data[:, 0:D_plot]), np.max(data[:, 0:D_plot]))
    ###########################################################################
    # Plot each digit with different colors
    ###########################################################################
    colorset = ['darkolivegreen', 'brown', 'coral', 'crimson', 'purple', 'gold', 'blue', 'magenta', 'orangered', 'chocolate']
    n_digit = int(np.max(data[:, D_plot]))
    num_class = n_digit + 1
    for digit in range(num_class):
        class_data = data[data[:, D_plot] == np.asarray(digit)]
        clr = colorset[int(digit)]
        ax.scatter(class_data[:, 0], class_data[:, 1], class_data[:, 2], color=clr, marker="*", label=str(int(digit)))

    for points in range(num_annotation):
        digit = data[points, D_plot]
        clr = colorset[int(digit)]
        # plt.annotate(str(int(digit)),
        #              (data[points, 0], data[points, 1], data[points, 2]),
        #              textcoords='offset points',
        #              xytext=(0, 0, 0),
        #              ha='center', color='black')
        ax.text(data[points, 0], data[points, 1], data[points, 2], str(int(digit)), color='black')
    plt.title('Hidden representation with 3 principal components')
    ax.set_xlabel('Hidden 1')
    ax.set_ylabel('Hidden 2')
    ax.set_zlabel('Hidden 3')
    plt.legend()
    plt.show()

###########################################################################
# Script
###########################################################################
train_data = './optdigits_train.txt'
val_data = './optdigits_valid.txt'
test_data = './optdigits_test.txt'
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

###########################################################################
# Train with best number of hidden units
###########################################################################
H = 15
z, w, v = mlp.MLPTrain(norm_train_data, norm_val_data, 10, H)
if (z is None) or (w is None) or (v is None):
    print('Training returned None')
    exit(1)
###########################################################################
# Combine training and validation data and run trained MLP
###########################################################################
norm_trn_val_data = np.vstack((norm_train_data, norm_val_data))
x_trn_val = np.hstack((np.ones((N_trn+N_val, 1)), norm_trn_val_data))
y_kn_trn_val, z_hn_trn_val = mlp.forwardPropagation(w, v, x_trn_val[:, 0:-1].T)
y_nk_trn_val = y_kn_trn_val.T
z_nh_trn_val = z_hn_trn_val.T

y_predicted_trn_val = np.argmax(y_nk_trn_val, axis=1)
err_trn_val = (np.sum(y_predicted_trn_val != norm_trn_val_data[:, -1])) * (100. / (N_trn+N_val))
###########################################################################
# Apply PCA on hidden unit outputs: Principal components = 2
###########################################################################
pca = PCA(n_components=2)
z_n2_projected = pca.fit_transform(z_nh_trn_val)
y_labels = np.reshape(norm_trn_val_data[:, -1], (N_trn+N_val, 1))
z_n2_projected = np.hstack((z_n2_projected, y_labels))
num_annotation = 100
plot_data(z_n2_projected, num_annotation)
###########################################################################
# Apply PCA on hidden unit outputs: Principal components = 3
###########################################################################
pca = PCA(n_components=3)
z_n3_projected = pca.fit_transform(z_nh_trn_val)
z_n3_projected = np.hstack((z_n3_projected, y_labels))

num_annotation = 100
plot_data3D(z_n3_projected, num_annotation)

