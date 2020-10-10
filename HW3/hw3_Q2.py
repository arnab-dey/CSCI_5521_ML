###########################################################################
# Imports
###########################################################################
import EMG
import matplotlib.pyplot as plt
###########################################################################
# Run EM
###########################################################################
imagepath = './stadium.jpg'
k_arr = [4, 8, 12]
flag = 0
exp_complt_llh_arr = []
###########################################################################
# Run EM: Q2.a
###########################################################################
for k in k_arr:
    print('Running EM with k = ', k, 'with 200 iterations')
    h, m, complete_likelihood = EMG.EMG(imagepath, k, flag)
    exp_complt_llh_arr.append(complete_likelihood)

###########################################################################
# Plot expected value of complete log-likelihood: Q2.b
###########################################################################
color_arr = ['r', 'g', 'b']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.minorticks_on()
ax.grid(which='major', linestyle='-', linewidth='0.5')
ax.grid(which='minor', linestyle="-.", linewidth='0.5')
for index in range(len(k_arr)):
    label = 'k=' + str(k_arr[index])
    plt.plot(exp_complt_llh_arr[index], color_arr[index], marker='+', label=label)
plt.title('Variation of expected value of complete log-likelihood')
plt.xlabel('Number of iterations')
plt.ylabel('Expected log-likelihood')
plt.legend()
plt.show()