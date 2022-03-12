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
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]), cmap='gray')
    plt.axis('off')
  plt.show()

def parse_image(img_shape: list, img: str):
  
    msk = tf.strings.regex_replace(img, "Images", "Segmentation")
    img = tf.image.decode_png(tf.io.read_file(img),channels=CHANNELS)
    msk = tf.image.decode_png(tf.io.read_file(msk),channels=CHANNELS)
    
    img = tf.image.resize(img,img_shape,antialias=True)
    msk = tf.image.resize(msk,img_shape,antialias=True)

    msk = tf.where(msk < 1, np.dtype('uint8').type(0), np.dtype('uint8').type(1))    
    img = tf.cast(img, tf.uint8)
    msk = tf.cast(msk, tf.uint8)
    return (img,msk)

def augment_image(img,mask,seed,enable):
  #from https://dergipark.org.tr/tr/download/article-file/1817118
  #horizontal flip resulted increased performance
  if enable[0]:
    do_flip = tf.random.uniform([],seed=seed) > 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
  #image brightness
  if enable[1]:
    img = tf.image.adjust_brightness(img, \
            delta = tf.random.uniform([], minval=-0.25, maxval=0.25,\
                                     dtype=tf.dtypes.float32,seed=seed))

  return (img,mask)


def generate_dataset(dataPath, imageType, seed=42, img_shape=(512,512,1), batch_size = 16,enable_augmentation = (1,1),\
                     repeat_count=3, kfold = 1):
    #list
    datasets = []
    img_shape = [img_shape[0], img_shape[1]]

    if not any(enable_augmentation):
      print("no image augmentaton")
      repeat_count = 1
    else:
      print("image augmentation")
      
    files = tf.data.Dataset.list_files(os.path.join(dataPath,imageType), seed=seed).repeat(count=repeat_count)
    for k in range(kfold):
      train_dataset = files.shard(num_shards = kfold, index = k)
      train_dataset = train_dataset.shuffle(seed=seed, buffer_size = len(train_dataset), reshuffle_each_iteration=True)
      train_dataset = train_dataset.map(lambda x: parse_image(img_shape,x),num_parallel_calls=tf.data.AUTOTUNE)
      train_dataset = train_dataset.map(lambda x,y: augment_image(x,y,seed,enable_augmentation),\
                      num_parallel_calls=tf.data.AUTOTUNE)
      train_dataset = train_dataset.batch(batch_size)
      train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
      datasets.append(train_dataset)

    return datasets
