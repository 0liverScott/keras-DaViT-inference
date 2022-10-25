import os
import time

from keras_cv_attention_models import davit
mm = davit.DaViT_T(pretrained="weights/davit.DaViT_T_0.9502.h5")

""" Run predict """
import tensorflow as tf
from tensorflow import keras
for i in os.listdir('input'):
    # input()
    t = time.time()
    img = tf.image.decode_image(tf.io.read_file(f'input/{i}'), channels=3, dtype=tf.dtypes.float32)
    # imm = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    pred = mm(tf.expand_dims(tf.image.resize(img, mm.input_shape[1:3]), 0)).numpy()
    pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
    CLASS_INDEX = {"0": [0, "Mask"], "1": [1, "BadMask"], "2": [2, "NoMask"], "3": ""}
    results = []
    top = 1
    for pre in pred:
        top_indices = pre.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pre[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    print(f"{results} {i} {time.time()-t}")
    # print(f'{i} {pred} {round(time.time()-t,4)}s estimated')
