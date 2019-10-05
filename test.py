#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch
import torch.utils.data as Torchdata
import numpy as np
from tqdm import tqdm

from tools import *
from net import *

# Parameters setting
DATASET = 'PaviaU' # PaviaU; Salinas; KSC
N_RUNS = 10 # the runing times of the experiments
SAMPLE_SIZE = 10 # training samples per class
PATCH_SIZE = 5 # Hyperparameter: patch size
FOLDER = './Datasets/' # the dataset folder
PRE_ALL = False  # wheather the full map is predicted
FEATURE_DIM = 64 # Hyperparameter: the number of convolutional filters
GPU = 0
# file paths of checkpoints
CHECKPOINT_ENCODER = 'checkpoints/Bing_Encoder/' + DATASET + '/'
CHECKPOINT_RELATION = 'checkpoints/Bing_Relation/' + DATASET + '/'

        
##########TEST##############################
def test():
    # datasets prepare
    ''' img: array 3D; gt: array 2D;'''
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
    # Number of classes + unidefind label
    N_CLASSES = len(LABEL_VALUES) - 1
    # Number of bands
    N_BANDS = img.shape[-1]
    # run the experiment several times
    for run in range(N_RUNS):
        # Sample get from training spectra
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)
        
        ## test for all pixels
        if PRE_ALL:
            test_gt = np.ones_like(test_gt)
        
        print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                     np.count_nonzero(gt)))
        print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
        # for test
        train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
        train_loader = Torchdata.DataLoader(train_dataset,
                                           batch_size=N_CLASSES*SAMPLE_SIZE,
                                           shuffle=False)
        tr_data, tr_labels = train_loader.__iter__().next()
        tr_data = tr_data.cuda(GPU)
        
        test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
        test_loader = Torchdata.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False)
        # init neural networks
        
        feature_encoder = CNNEncoder(N_BANDS,FEATURE_DIM)
        relation_network = RelationNetwork(PATCH_SIZE,FEATURE_DIM)

        # load weight from train
        if CHECKPOINT_ENCODER is not None:
            encoder_file = CHECKPOINT_ENCODER + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
            with torch.cuda.device(GPU):
                feature_encoder.load_state_dict(torch.load(encoder_file))
        else:
            raise('No Chenkpoints for Encoder Net')
        if CHECKPOINT_RELATION is not None:
            relation_file = CHECKPOINT_RELATION + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
            with torch.cuda.device(GPU):
                relation_network.load_state_dict(torch.load(relation_file))
        else:
            raise('No Chenkpoints for Relation Net')
            
        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)
        
        print('Testing...')
        feature_encoder.eval()
        relation_network.eval()
        accuracy, total = 0., 0.
        #scores_all = np.zeros((len(test_loader),N_CLASSES))
        test_labels = np.zeros(len(test_loader))
        pre_labels = np.zeros(len(test_loader))
        pad_pre_gt = np.zeros_like(test_dataset.label)
        pad_test_indices = test_dataset.indices
        for batch_idx, (te_data, te_labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
            with torch.no_grad():
                te_data, te_labels = te_data.cuda(GPU), te_labels.cuda(GPU)               
                tr_features = feature_encoder(tr_data)
                te_features = feature_encoder(te_data)
                tr_features_ext = tr_features.unsqueeze(0)
                te_features_ext = te_features.unsqueeze(0).repeat(N_CLASSES*SAMPLE_SIZE,1,1,1,1)
                te_features_ext = torch.transpose(te_features_ext,0,1)
                trte_pairs = torch.cat((tr_features_ext,te_features_ext),2).view(-1,FEATURE_DIM*2,PATCH_SIZE,PATCH_SIZE)
                trte_relations = relation_network(trte_pairs).view(-1,SAMPLE_SIZE)
                #scores = torch.mean(trte_relations,dim=1)
                scores, _ = torch.max(trte_relations,dim=1)
                #scores_all[batch_idx,:] = scores
                _, output = torch.max(scores,dim=0)
                pre_labels[batch_idx] = output.item() + 1
                test_labels[batch_idx] = te_labels.item() + 1
                pad_pre_gt[pad_test_indices[batch_idx]] = output.item() + 1
                accuracy += output.item() == te_labels.item()
                total +=1
        rate = accuracy / total
        print('Accuracy:', rate)
        # save sores
        results = dict()
        results['OA'] = rate
        results['gt'] = gt
        results['test_gt'] = test_gt
        results['train_gt'] = train_gt
        p = PATCH_SIZE // 2
        wp,hp = pad_pre_gt.shape
        pre_gt = pad_pre_gt[p:wp-p,p:hp-p]
        results['pre_gt'] = np.asarray(pre_gt,dtype='uint8')
        if PRE_ALL:
            results['pre_all'] = np.asarray(pre_gt,dtype='uint8')
        results['test_labels'] = test_labels
        results['pre_labels'] = pre_labels
        # expand train_gt by superpxiel
        save_folder = DATASET
        save_result(results, save_folder, SAMPLE_SIZE, run)
#############################################################################              
def main():
    test()
        
if __name__ == '__main__':
    main()     
