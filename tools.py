#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch
import torch.utils.data as Torchdata
import numpy as np
from scipy import io
import random
import os
from tqdm import tqdm

################### get data set
def get_dataset(dataset_name, target_folder='./Datasets/'):
    palette = None
    folder = target_folder + dataset_name + '/'
    if dataset_name == 'IndianPines':
        #load the image
        img = io.loadmat(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']        
        gt = io.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        rgb_bands = (43, 21, 11) #AVIRIS sensor
        ignored_labels = [0]
    elif dataset_name == 'PaviaU':
        # load the image
        img = io.loadmat(folder + 'PaviaU.mat')['paviaU']
        gt = io.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        rgb_bands = (55, 41, 12)
        ignored_labels = [0]
    elif dataset_name == 'PaviaC':
        # Load the image
        img = io.loadmat(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = io.loadmat(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        # Load the image
        img = io.loadmat(folder + 'Salinas_corrected.mat')['salinas_corrected']
        gt = io.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
                        'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        rgb_bands = (43, 21, 11) #I don't sure
        ignored_labels = [0]
    elif dataset_name == 'SalinaA':
        # Load the image
        img = io.loadmat(folder + 'SalinasA_corrected.mat')['salinasA_corrected']
        gt = io.loadmat(folder + 'SalinasA_gt.mat')['salinasA_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Corn_senesced_green_weeds', 
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 
                        'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk']
        rgb_bands = (43, 21, 11) # I don't sure
        ignored_labels = [0]
    elif dataset_name == 'KSC':
        # Load the image
        img = io.loadmat(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = io.loadmat(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    else:
        raise ValueError("{} dataset is unknown.".format(dataset_name))
        
    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    n_bands = img.shape[-1]
    for band in range(n_bands):
        min_val = np.min(img[:,:,band])
        max_val = np.max(img[:,:,band])
        img[:,:,band] = (img[:,:,band] - min_val) / (max_val - min_val)
    return img, gt, label_values, ignored_labels, rgb_bands, palette

####################### get train test split
def sample_gt(gt, train_size, mode='fixed_withone'):
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
       if mode == 'random':
           train_size = float(train_size)/100 #dengbin:20181011
    
    if mode == 'random_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features
            train_len = int(np.ceil(train_size*len(X)))
            train_indices += random.sample(X, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    
    elif mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) # x,y features

            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt
###################################### torch datasets
class HyperX(torch.utils.data.Dataset):

    def __init__(self, data, gt, dataset_name, patch_size=5, flip_argument=True, rotated_argument=True):
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = patch_size
        self.flip_augmentation = flip_argument
        self.rotated_augmentation = rotated_argument
        self.name = dataset_name
        
        p = self.patch_size // 2
        # add padding
        if self.patch_size > 1:
            self.data = np.pad(self.data, ((p,p),(p,p),(0,0)), mode='constant')
            self.label = np.pad(self.label, p, mode='constant')
        else:
            self.flip_argument = False
            self.rotated_argument = False
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
        
    def resetGt(self, gt):
        self.label = gt
        p = self.patch_size // 2
        # add padding
        if self.patch_size > 1:
            self.label = np.pad(gt, p, mode='constant')
            
        self.indices = []
        for c in np.unique(self.label):
            if c == 0:
                continue
            c_indices = np.nonzero(self.label == c)
            X = list(zip(*c_indices))
            self.indices += X
            
    @staticmethod
    def flip(*arrays):
        #horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        # if horizontal:
            # arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays
    
    # dengbin
    @staticmethod
    def rotated(*arrays):
        p = np.random.random()
        if p < 0.25:
            arrays = [np.rot90(arr) for arr in arrays]
        elif p < 0.5:
            arrays = [np.rot90(arr, 2) for arr in arrays]
        elif p < 0.75:
            arrays = [np.rot90(arr, 3) for arr in arrays]
        else:
            pass
        return arrays

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.rotated_augmentation and self.patch_size > 1:
            # Perform data rotated augmentation (only on 2D patches) #dengbin 20181018
            data, label = self.rotated(data, label)
        
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            #data = data[:, 0, 0]
            label = label[0, 0]

        return data, label-1
############################################################ save model
def save_model(model, model_name, dataset_name, **kwargs):
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir) #dengbin:20181011
     if isinstance(model, torch.nn.Module):
         filename = "non_augmentation_sample{sample_size}_run{run}_epoch{epoch}_{metric:.2f}".format(**kwargs)
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
         filename2 = "non_augmentation_sample{}_run{}".format(kwargs['sample_size'], kwargs['run'])
         torch.save(model.state_dict(), model_dir + filename2 + '.pth')
############################################################ save and get samples/results         
def get_sample(dataset_name, sample_size, run):
    sample_file = './trainTestSplit/' + dataset_name + '/sample' + str(sample_size) + '_run' + str(run) + '.mat'
    data = io.loadmat(sample_file)
    train_gt = data['train_gt']
    test_gt = data['test_gt']
    return train_gt, test_gt

def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})
    
def get_result(dataset_name, sample_size, run):
    scores_dir = './results/' + dataset_name + '/'
    scores_file = scores_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    scores = io.loadmat(scores_file)
    return scores

def save_result(result, dataset_name, sample_size, run):
    scores_dir = './results/' + dataset_name + '/'
    if not os.path.isdir(scores_dir):
        os.makedirs(scores_dir)
    scores_file = scores_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(scores_file,result)
####################################################################