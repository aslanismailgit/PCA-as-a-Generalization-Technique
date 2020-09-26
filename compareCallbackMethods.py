#%%
import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.linalg import svd
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

# from keras.callbacks import LambdaCallback
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# %% --- 4.05 seconds --- 
class PCAGeneralizerSK(keras.callbacks.Callback):
    def __init__(self,var2explain):
        """ Save params in constructor
        """
        self.var2explain = var2explain

    def on_train_batch_end(self , batch, logs=None):
        var2explain = self.var2explain
        weights = model.layers[1].get_weights()[0]
        weightsB = model.layers[1].get_weights()[1]           

        pca = PCA(n_components = var2explain)
        X_pca = pca.fit(weights)
        # reduced is in new dim
        X_reduced = pca.fit_transform(weights)
        X_recovered = pca.inverse_transform(X_reduced)
        # np.dot(X, self.components_) + self.mean_
        comps = pca.components_
        print("---------------->", comps.shape)
        model.layers[1].set_weights([X_recovered,weightsB])
 
#%% --- 2.49 seconds --- 
class PCAGeneralizer(keras.callbacks.Callback):
    def __init__(self,var2explain):
        """ Save params in constructor
        """
        self.var2explain = var2explain

    def on_train_batch_end(self , batch, logs=None):
        var2explain = self.var2explain
        weights = model.layers[1].get_weights()[0]
        weightsB = model.layers[1].get_weights()[1]           
        mn = tf.reduce_mean(weights,0)
        weights_normalized = weights - mn
        S, U, Vt = svd(weights_normalized, full_matrices=True)
        eigvalSVD = []
        n=weights.shape[0]
        eigvalSVD = (S ** 2) / (n - 1)
        varExpRatio_tf = eigvalSVD / sum(eigvalSVD)
        varExpCum_tf = tf.cumsum(varExpRatio_tf)
        num_comps_tf = len(varExpCum_tf[varExpCum_tf<var2explain]) + 1
        print("---------------->", num_comps_tf)

        comps_tf = ((Vt)[:,:num_comps_tf])
        X_reduced_tf = tf.matmul(weights_normalized, comps_tf)
        weights_reproduced = tf.tensordot(X_reduced_tf,tf.transpose(comps_tf),axes=1) + mn
        model.layers[1].set_weights([weights_reproduced,weightsB])
 


#%%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images.shape
train_images.shape
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images[:3000]
train_labels = train_labels[:3000]
test_images = test_images[:2000]
test_labels = test_labels[:2000]
#%%
dr_rate = .95
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
start_time = time.time()
history = model.fit(train_images, train_labels, 
                epochs=2, 
                batch_size=512,
                verbose=0,
                validation_data=(test_images,test_labels),
                callbacks=[PCAGeneralizerSK(var2explain=dr_rate)])
dur = time.time() - start_time
print(f'--- {dur:.3} seconds --- ')

# %%
