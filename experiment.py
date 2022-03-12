from model import UNET
from util import *
import tensorflow as tf
import numpy as np
import sklearn as sk
import pickle

import os
import tarfile

dataPath = os.path.join(".","Data","Images")
maskPath = os.path.join(".","Data","Segmentation")
dataType = "*.png"

SEED = 42
DATA_SPLIT = 1

OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
BATCH_SIZE = 16
IMAGE_SHAPE = (512,512,1)
KFOLD  = 3 #MUST BE >=2
EPOCHS = 2


def learn_with_cv(folds,enable_status, repeat):
    dataset = generate_dataset(dataPath, dataType, seed=SEED, img_shape = IMAGE_SHAPE, batch_size = BATCH_SIZE,
                          kfold = folds, repeat_count = repeat, enable_augmentation = enable_status)
    histories = []
    for k in range(folds):
        print("Doing k-fold {} of {}".format(k+1,folds))
        test_dataset = dataset[k]
        train_datasets = []
        for k2 in range(folds):
            if k2 == k:
                continue
            train_datasets.append(dataset[k2])
        train_dataset = train_datasets[0]
        for ds in train_datasets[1:]:
            train_dataset = train_dataset.concatenate(ds)

        model = UNET(input_shape = IMAGE_SHAPE)
        model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)
        history = model.fit(train_dataset, validation_data = test_dataset, \
                                epochs = EPOCHS, verbose = 1)
        histories.append(history.history)

    history  = {}
    history["loss"] = np.mean([hstry["loss"] for hstry in histories],axis=0)
    history["accuracy"] = np.mean([hstry["accuracy"] for hstry in histories],axis=0)
    history["val_loss"] = np.mean([hstry["val_loss"] for hstry in histories],axis=0)
    history["val_accuracy"] = np.mean([hstry["val_accuracy"] for hstry in histories],axis=0)
    return history

things2enable = {"nothing" : (0,0), "brightness":(0,1), "flip": (1,0), "both":(1,1)}
EPOCHS = 150
folds = 5
for en in things2enable.keys():
    cv_history = learn_with_cv(folds = folds, enable_status=things2enable[en], repeat=5)
    pickle.dump(cv_history, open("fold{}_cv_{}.pkl".format(folds,en),"wb"))
