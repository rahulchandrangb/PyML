import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

np.random.seed(23)

mean_vec1 = np.array([0,0,0])
mean_vec2 = np.array([1,1,1])

covar_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
covar_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])

class1_sample = np.random.multivariate_normal(mean_vec1,covar_mat1,20).T
assert class1_sample.shape == (3,20), "The matrix has to be of the dimensions 3x20"

class2_sample = np.random.multivariate_normal(mean_vec2,covar_mat2,20).T
assert class2_sample.shape == (3,20), "The matrix has to be of the dimensions 3x20"

print "Sample data loaded, now plotting sample values generated"
## Plot the samples

#fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111, projection='3d')
#plt.rcParams['legend.fontsize'] = 10
#ax.plot(class1_sample[0,:], class1_sample[1,:],class1_sample[2,:], 'o', markersize=8, color='black', alpha=0.9, label='class1')
#ax.plot(class2_sample[0,:], class2_sample[1,:],class2_sample[2,:], '^', markersize=8, color='orange', alpha=0.7, label='class2')
#plt.title('Samples for class 1 and class 2')
#ax.legend(loc='lower left')
#plt.show()

# Step 1 : We don't need class labels in pca, as it's based on higher variance inference , so let's merge both samples

merged_samples = np.concatenate((class1_sample,class2_sample),axis=1)
assert merged_samples.shape == (3,40), "The matrix has to be of the dimensions 3x40"

# Step 2. Calculate the mean vector [3-dimensional in this case]
##  2.1 Calculate mean
mean_x = np.mean(merged_samples[0,:])
mean_y = np.mean(merged_samples[1,:])
mean_z = np.mean(merged_samples[2,:])

##  2.2 Merge to a mean vector
merged_mean = np.array([[mean_x],[mean_y],[mean_z]])
print"Mean Vector:\n",merged_mean

# Step 3. Calculate the covariance matrix

covar_merged = np.cov([merged_samples[0,:],merged_samples[1,:],merged_samples[2,:]])
print "Covariance matrix:\n",covar_merged

# Step 4. Calculate eigen values and vectors from covariance  matrix

eig_val, eig_vec = np.linalg.eig(covar_merged)
print "Eigen Values:", eig_val
print "Eigen Vectors:", eig_vec

### Let's plot eigen vectors centered at the mean of the samples

class SpaceArrow(FancyArrowPatch):
    def __init__(self,xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs,ys,zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


fig  = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(merged_samples[0,:],merged_samples[1,:],merged_samples[2,:],'o', markersize=8, color='green', alpha=0.3)
ax.plot([mean_x],[mean_y],[mean_z],'o',  markersize=10, color='blue', alpha=0.5)

for v in eig_vec:
    a = SpaceArrow([mean_x, v[0]], [mean_y, v[1]],[mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigen vectors')

# plt.show()

# Step 5. Sort eig vectors on the basis of eig val

eig_pairs = [(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort()
eig_pairs.reverse()
for i in eig_pairs:
    print(i[0])



