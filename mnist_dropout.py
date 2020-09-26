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
# from keras.callbacks import LambdaCallback
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
#%%
dr_rate_arr = [.90, .75, .50]
for dr_rate in dr_rate_arr:    
    for i in range(10):
        print("run ------------------------>  ", i)
        print("DR rate -------------------->  ", dr_rate)
        tf.keras.backend.clear_session()
        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
        model.add(layers.Dropout(dr_rate))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        #%
        history = model.fit(train_images, train_labels, 
                        epochs=50, 
                        batch_size=512,
                        verbose=0,
                        validation_data=(test_images,test_labels))
                        # callbacks=[PCAGeneralizer()])

        _id = time.strftime("_%m_%d_%H_%M_%S")
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("./dropout/history" + "_dr_" + str(dr_rate) + _id + ".csv")
        print("history csv created")
# %%
