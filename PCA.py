import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components = 2)
pca.fit(X)

print(pca.explained_variance_ratio_) #Represent the eigen values
print(pca.components_[0]) #First of two principle component
print(pca.components_[1]) #second principle componet
print(pca.singular_values_)