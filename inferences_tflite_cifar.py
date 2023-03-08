import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import load_model  
import numpy as np
import random
import time
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
# print("imports complete")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

config = dict(
    BATCH_SIZE = 32,
    IMAGE_SIZE = 28,
    test_csv = "sign_mnist_test.csv",
    train_csv = "sign_mnist_train.csv"
)

def evaluate_per_batch(tflite_interpreter,x_test,y_test,img_size,n_class,channels):
    predictions = []
    ip_gesture = tflite_interpreter.get_input_details()
    op_gesture = tflite_interpreter.get_output_details()
    tflite_interpreter.resize_tensor_input(ip_gesture[0]['index'], (32, img_size, img_size, channels))
    tflite_interpreter.resize_tensor_input(op_gesture[0]['index'], (32, n_class))
    tflite_interpreter.allocate_tensors()

    n_samples = x_test.shape[0]
    perfect_batch = n_samples // 32
    remaining_batch = n_samples % 32
    print('perfect_batch ',perfect_batch)
    print('remaining_batch',remaining_batch)

    i = 0;
    j = 0
    while (j < perfect_batch):
        tflite_interpreter.set_tensor(ip_gesture[0]['index'],x_test[i:i+32])
        i = i + 32
        j += 1
        tflite_interpreter.invoke()

        tflite_infer = tflite_interpreter.get_tensor(op_gesture[0]['index'])
        predictions.extend(tflite_infer)
        print(len(predictions),j,end='\r')

    tflite_interpreter.resize_tensor_input(ip_gesture[0]['index'], (remaining_batch, img_size, img_size, channels))
    tflite_interpreter.resize_tensor_input(op_gesture[0]['index'], (remaining_batch, n_class))
    tflite_interpreter.allocate_tensors()

    tflite_interpreter.set_tensor(ip_gesture[0]['index'],x_test[-remaining_batch:])
    tflite_interpreter.invoke()

    tflite_infer = tflite_interpreter.get_tensor(op_gesture[0]['index'])
    predictions.extend(tflite_infer)

    y_testt = y_test.reshape(y_test.shape[0])
    predictions = np.array(predictions)
    y_pred = predictions.argmax(axis=1)
    return (y_testt == y_pred).sum() / x_test.shape[0]




# exit(0)

# -----------------------------loading cifar10 model -----------------------------------

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype("float32") / 255

cifar10_model = tf.keras.models.load_model('cifar10_model')

print(cifar10_model.evaluate(x_test,y_test,batch_size=config['BATCH_SIZE'],verbose=1))

cifar10_int = tf.lite.Interpreter(model_path='cifar10_tflite.tflite')

preds = evaluate_per_batch(cifar10_int,x_test,y_test,32,10,3)
# print(x_test.shape)
print('accuracy:',preds)
print(x_test.shape)
