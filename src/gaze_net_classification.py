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
    TRAIN = True
    num_class = 6
    batch_size = 4
    gaze_gen_batch_size = 1
    gaze_gen_time_steps = 6
    epochs = 1
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    eval_freq = 1       # epoch
    print_freq = 1      # iteration
    dataset_path = '../../gaze-net/gaze_dataset_old'
    img_size = (224,224)
    log_path = '../log'
    logger = Logger(log_path, 'classification')

    model = models.alexnet(pretrained=True).cuda()
    # print(model)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum = momentum, weight_decay=weight_decay)
    # define generator
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training', crop=False,
                                batch_size=gaze_gen_batch_size, crop_with_gaze=True,time_steps=gaze_gen_time_steps,
                                crop_with_gaze_size=img_size[0],class_mode='categorical')
    # small dataset, error using validation split
    val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', crop=False,
                                batch_size=gaze_gen_batch_size, crop_with_gaze=True,time_steps=gaze_gen_time_steps,
                                crop_with_gaze_size=img_size[0], class_mode='categorical')

    para = {'bs': batch_size, 'img_size': img_size, 'num_class': num_class,
            'print_freq': print_freq}

    if TRAIN:
        print("training mode")
        best_acc = 0
        for epoch in range(epochs):
            print(learning_rate)
            adjust_learning_rate(optimizer,epoch,learning_rate)
            print('Epoch: {}'.format(epoch))
            train(train_data, model, criterion, optimizer, epoch, logger, para)
            # if epoch % eval_freq ==0 or epoch == epochs-1:
            #     acc = validate(val_data,model,criterion,optimizer,epoch,logger,para)
            #     is_best = acc>best_acc
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'arch': arch,
            #         'state_dict': model.state_dict(),
            #         'best_acc': best_acc,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best)
    # else:
    #     print("let's test the model")
    #     model = load_checkpoint(model)
    #     print("get input test and visualize mode")
    #     print("visualizing the training data")
    #
    #     vis_data_path = '../vis/train'
    #     if not os.path.exists(vis_data_path):
    #         os.makedirs(vis_data_path)
    #     acc = validate(train_data, model, criterion, -1, \
    #                     logger, para, True, vis_data_path)
    #     print("visualization for validation data")
    #     vis_data_path = '../vis/val/'
    #     if not os.path.exists(vis_data_path):
    #         os.makedirs(vis_data_path)
    #     acc = validate(val_data, model, criterion, -1, \
    #                     logger, para, True, vis_data_path)
    print("finished here")


def train(train_data,model,criterion,optimizer,epoch,logger,para):
    bs = para['bs']
    img_size = para['img_size']
    num_class = para['num_class']
    print_freq = para['print_freq']

    model.train()
    end = time.time()
    train_num = len(train_data)

    for i in range(train_num):
        # data_time.update(time.time()  - end)
        img_seq,target_seq = next(train_data)
        img_seq = normalize(img_seq)
        img_seq = np.reshape(img_seq,(-1,3,) + img_size)
        img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq).cuda(), requires_grad=True)
        target_seq_var = torch.autograd.Variable(torch.Tensor(target_seq).cuda()).long()

        optimizer.zero_grad()
        output = model(img_seq_var)
        output = output.view(-1,num_class)
        loss = criterion(output, target_seq_var)

        loss.backward()
        optimizer.step()
        time_cnt = time.time() - end
        end = time.time()
        end = time.time()

        if i % print_freq == 0:
            output = F.softmax(output, dim=1)
            acc_frame = metric_frame(output,target_seq_var)
            # acc_frame = acc_frame / (1.0 * )


#

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='../model/spatial/checkpoint.pth.tar'):
    print("save checkpoint")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../model/spatial/model_best.pth.tar')

def load_checkpoint(model, filename='../model/spatial/checkpoint.pth.32.tar'):
    if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    return model

if __name__ == '__main__':
    main()
