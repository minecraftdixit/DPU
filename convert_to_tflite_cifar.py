import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import load_model  
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
# print("imports complete")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

# cifar10_model = load_model('cifar10_model')

cifar10_converter = tf.lite.TFLiteConverter.from_saved_model('cifar10_model')
tflite_cifar10 = cifar10_converter.convert()

with open('cifar10_tflite.tflite','wb') as f:
    f.write(tflite_cifar10)

# gesture_converter = tf.lite.TFLiteConverter.from_saved_model('gesture_recogition_model')
# tflite_gesture = gesture_converter.convert()

# with open('gesture_tflite.tflite','wb') as f:
#     f.write(tflite_gesture)
# print(cifar10_model.evaluate(x_test,y_test,verbose=1))