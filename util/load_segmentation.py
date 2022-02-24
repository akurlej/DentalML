import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import os

N_CLASS = 2 #jaw...or not jaw...
CHANNELS = 1

def display_mask_image(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def parse_image(img_shape: list, img: str):
  
    msk = tf.strings.regex_replace(img, "Images", "Segmentation")
    img = tf.image.decode_png(tf.io.read_file(img),channels=CHANNELS)
    msk = tf.image.decode_png(tf.io.read_file(msk),channels=CHANNELS)
    
    img = tf.image.resize(img,img_shape)
    msk = tf.image.resize(msk,img_shape)

    msk = tf.where(msk < 1, np.dtype('uint8').type(0), np.dtype('uint8').type(1))
    return {"image":img,"mask": msk}


def generate_dataset(dataPath, imageType, seed=42, img_shape=[572,572]):
    #list
    images = glob(os.path.join(dataPath,imageType))

    train_dataset = tf.data.Dataset.list_files(os.path.join(dataPath,imageType), seed=seed)
    train_dataset = train_dataset.map(lambda x: parse_image(img_shape,x))
    
    return train_dataset
