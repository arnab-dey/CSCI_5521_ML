###########################################################################
# Imports
###########################################################################
import numpy as np
import os, sys
import myLDA as ml
import matplotlib.pyplot as plt
###########################################################################
# Function to plot data with different colors
###########################################################################
def plot_data(data, num_annotation):
    N_plot, D_plot = data.shape
    # Last column represents class label, therefore feature dimension is D_plot-1
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
    plt.title('Optdigits after LDA')
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.legend()
    plt.show()
###########################################################################
# Training, Test dataset location
###########################################################################
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
# We use reduced dimension to be 2 so that we can plt the data
# for visualization
reduced_dimension = 2
w, e_val = ml.myLDA(trn_data, reduced_dimension)
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
# Plot reduced dimension training and test data
###########################################################################
plt_data = np.vstack((reduced_trn_data, reduced_test_data))
num_annotation = 100
plot_data(plt_data, num_annotation)
