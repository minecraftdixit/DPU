import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers  
import numpy as np
import random
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

config = dict(
    lr = 0.001,
    EPOCHS = 10,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 64,
    SEED = 42,
    GPU_ID=0,
    val_split = 0.2
)

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(config['BATCH_SIZE'])
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(config['BATCH_SIZE'])

# exit(0)

class CNNLayer(tf.keras.layers.Layer):
    def __init__(self,out_channels,kernel_size=3):
        super(CNNLayer,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels,kernel_size=kernel_size,padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(tf.nn.relu)

        # tf.keras.activations.relu(a)
        # tf.keras.layers.Activation('relu')(a)
        # tf.nn.relu(a)
        # tf.keras.layers.Activation(tf.nn.relu)(a)

    def call(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNarchitecture(tf.keras.Model):
    def __init__(self,n_class=10):
        super(CNNarchitecture,self).__init__()
        self.cn1_a = CNNLayer(64,3)
        self.cn1_b = CNNLayer(64,3)
        self.cn2 = CNNLayer(256,3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.relu2 = tf.keras.layers.Activation(tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(n_class)

    def call(self,x):
        x1 = self.cn1_a(x)
        x2 = self.cn1_b(x)
        x = tf.concat([x1,x2],axis=3)
        x = self.cn2(x)
        # x = tf.reshape(x,(x.shape[0],-1))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu2(x)
        x = self.dense2(x)
        return x 
        

model = CNNarchitecture()
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr=config['lr']),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
history = model.fit(train_data,epochs=config['EPOCHS'],batch_size=config['BATCH_SIZE'],verbose=1)
print(model.summary())

try:
    model.save('cifar10_model')
except Exception as e:
    print(e)

print(model.evaluate(test_data,verbose=1))