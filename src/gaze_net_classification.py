#!/usr/bin/env python
import os
import time
import torch
import pdb
import shutil
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
# from skimage.io import imsave
from scipy.misc import imsave
import matplotlib.pyplot as plt
# from skimage.transform import resize

from util import *
from logger import Logger
import gazenetGenerator as gaze_gen

def main():
    num_class = 6
    batch_size = 1
    time_step = 32
    epochs = 50
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    eval_freq = 1       # epoch
    print_freq = 1      # iteration
    dataset_path = '../../gaze-net/gaze_dataset_classification'
    img_size = 128
    log_path = '../log'
    logger = Logger(log_path, 'classification')

    model = models.alexnet(pretrained=True).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum = momentum, weight_decay=weight_decay)
    # define generator
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training', crop=False,
                                batch_size=batch_size, crop_with_gaze=True,
                                crop_with_gaze_size=img_size,class_mode='categorical')
    # small dataset, error using validation split
    val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', crop=False,
                                batch_size=batch_size, crop_with_gaze=True,
                                crop_with_gaze_size=img_size, class_mode='categorical')
    def test(train_data):
        img_seq, target = next(train_data)
        img = img_seq[0,1,:,:,:]
        plt.imshow(img)
        plt.show()
    test(train_data)
    print("finished here")





if __name__ == '__main__':
    main()
