import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def MyLinprog(X, y):
    ###########################################################################
    # Convert all parameters to numpy array to avoid error in case of
    # in case the inputs are not numpy arrays
    ###########################################################################
    x_train = np.asarray(X)
    y_label = np.asarray(y)

    ###########################################################################
    # Get no. of samples and feature length
    ###########################################################################
    num_data, num_feat = np.shape(x_train)
    if y_label.shape[0] < num_data:
        print("insufficient number of labels")
        return
    ###########################################################################
    # Sort data for plotting with different colors
    ###########################################################################
    c_1 = c_minus_1 = 0
    for sample in range(num_data):
        if y_label[sample] == 1:
            c_1 += 1
        else:
            c_minus_1 += 1
    x_1 = np.ones((c_1, num_feat))
    x_minus_1 = np.ones((c_minus_1, num_feat))
    index_1 = index_minus_1 = 0
    for sample in range(num_data):
        if y_label[sample] == 1:
            x_1[index_1] = x_train[sample]
            index_1 += 1
        else:
            x_minus_1[index_minus_1] = x_train[sample]
            index_minus_1 += 1
    x_sorted = (x_1, x_minus_1)

    ###########################################################################
    # Plot initialization figure
    ###########################################################################
    x_axis = np.array([max(x_train[:, 0]), min(x_train[:, 0])])
    #y_axis = (-w_init[0, 0] / w_init[0, 1]) * x_axis
    plt.figure(1)
    plt.title("Initialization")
    colors = ['red', 'blue']
    for data, clr in zip(x_sorted, colors):
        if clr == "red":
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="*")
        else:
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="P")
    #plt.plot(x_axis, y_axis)
    x_lim = (plt.gca()).get_xlim()
    y_lim = (plt.gca()).get_ylim()
    plt.draw()

    ###########################################################################
    # Run linprog
    ###########################################################################
    x_train = np.hstack((x_train, np.ones((num_data, 1))))
    num_feat = num_feat + 1
    f = np.append(np.zeros(num_feat), np.ones(num_data))
    A1 = np.hstack((x_train*np.tile(y_label, (1, num_feat)), np.eye(num_data))) #changed
    A2 = np.hstack((np.zeros((num_data, num_feat)), np.eye(num_data)))
    A = -np.vstack((A1, A2))
    b = np.append(-np.ones(num_data), np.zeros(num_data))
    x = linprog(f, A, b)
    w = x['x'][0:num_feat]

    ###########################################################################
    # Plot converged figure
    ###########################################################################
    y_axis = (-w[0]/w[1])*x_axis +(-w[2]/w[1])*(np.array([1, 1]))
    plt.figure(2)
    plt.title("Converged")
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    colors = ['red', 'blue']
    for data, clr in zip(x_sorted, colors):
        if clr == "red":
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="*")
        else:
            plt.scatter(data[:, 0], data[:, 1], color=clr, marker="P")
    plt.plot(x_axis, y_axis)
    #plt.savefig("problem3.png")
    plt.show()
    plt.close(1)
    plt.close(2)
    return w
