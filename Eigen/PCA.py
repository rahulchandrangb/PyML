import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

np.random.seed(23)




mean_vec1 = np.array([0,0,0])
mean_vec2 = np.array([1,1,1])

covar_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
covar_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])

class1_sample = np.random.multivariate_normal(mean_vec1,covar_mat1,20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

class2_sample = np.random.multivariate_normal(mean_vec2,covar_mat2,20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

print "Sample data loaded, now plotting sample values generated"

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:],class1_sample[2,:], 'o', markersize=8, color='black', alpha=0.9, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:],class2_sample[2,:], '^', markersize=8, color='orange', alpha=0.7, label='class2')
plt.title('Samples for class 1 and class 2')
ax.legend(loc='lower left')
plt.show()





