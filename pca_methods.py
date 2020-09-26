#%%
import time
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.linalg import svd
#%%
numofUnits = 50
numofPrevUnits = 30
weights = np.random.rand(numofPrevUnits,numofUnits)
var2explain = .9
#%%
start_time = time.time()
pca = PCA(n_components = var2explain)
X_pca = pca.fit(weights)
X_reduced = pca.fit_transform(weights)
X_recovered = pca.inverse_transform(X_reduced)
comps = pca.components_
sv = pca.singular_values_
X_reduced_comps = (weights-weights.mean(0)) @ comps.T
varExpSVD = pca.explained_variance_
varExpSVD_rat = pca.explained_variance_ratio_

dur = time.time() - start_time
print(f'--- {dur:.3} seconds --- ')

#%%
start_time = time.time()
mn = tf.reduce_mean(weights,0)
weights_normalized = weights - mn
S, U, Vt = svd(weights_normalized, full_matrices=True)
eigvalSVD = []
n=weights.shape[0]
eigvalSVD = (S ** 2) / (n - 1)
varExpRatio_tf = eigvalSVD / sum(eigvalSVD)
varExpCum_tf = tf.cumsum(varExpRatio_tf)
num_comps_tf = len(varExpCum_tf[varExpCum_tf<var2explain]) + 1
comps_tf = ((Vt)[:,:num_comps_tf])
X_reduced_tf = tf.matmul(weights_normalized, comps_tf)
dur = time.time() - start_time
print(f'--- {dur:.3} seconds --- ')
# %%
