#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch

import torch.utils.data as Torchdata
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from scipy import io
import os

from tools import *
from net import *

# Parameters setting
DATASET = 'PaviaU' # PaviaU; KSC; Salinas
SAMPLE_ALREADY = True # whether randomly generated training samples are ready
N_RUNS = 10 # the runing times of the experiments
SAMPLE_SIZE = 10 # training samples per class
BATCH_SIZE_PER_CLASS =  SAMPLE_SIZE // 2 # batch size of each class
PATCH_SIZE = 5 # Hyperparameter: patch size 
FLIP_ARGUMENT = False # whether need data argumentation of flipping data; default: False
ROTATED_ARGUMENT = False # whether need data argumentation of rotated data; default: False
ITER_NUM = 1000  # the total number of training iter; default: 50000
TEST_NUM = 5  # the total number of test in the training process
SAMPLING_MODE = 'fixed_withone' # fixed number for each class
FOLDER = './Datasets/' # the dataset folder
LEARNING_RATE = 0.1 # 0.01 good / 0.1 fast for SGD; 0.001 for Adam
FEATURE_DIM = 64 # Hyperparameter: the number of convolutional filters
GPU = 0 #



##########TRAIN##################    
def train():
    # datasets prepare
    ''' img: array 3D; gt: array 2D;'''
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)
    # Number of classes + unidefind label
    N_CLASSES = len(LABEL_VALUES) - 1
    # Number of bands
    N_BANDS = img.shape[-1]
    # run the experiment several times
    for run in range(N_RUNS):
        # Sample random training spectra
        if SAMPLE_ALREADY:
            train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run)
        else:
            train_gt, test_gt = sample_gt(gt, SAMPLE_SIZE, mode=SAMPLING_MODE)
            save_sample(train_gt, test_gt, DATASET, SAMPLE_SIZE, run)
        
        print("{} samples selected (over {})".format(np.count_nonzero(train_gt),
                                                     np.count_nonzero(gt)))
        print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
        # for test
        train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
        train_loader = Torchdata.DataLoader(train_dataset,
                                           batch_size=N_CLASSES*SAMPLE_SIZE,
                                           shuffle=False)
        tr_data, tr_labels = train_loader.__iter__().next()
        tr_data, tr_labels = tr_data.cuda(GPU), tr_labels.cuda(GPU)
        
        test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
        test_loader = Torchdata.DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False)
        # init neural networks
        print("init neural networks")

        feature_encoder = CNNEncoder(N_BANDS,FEATURE_DIM)
        relation_network = RelationNetwork(PATCH_SIZE,FEATURE_DIM)

        feature_encoder.apply(weights_init)
        relation_network.apply(weights_init)

        feature_encoder.cuda(GPU)
        relation_network.cuda(GPU)

        feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
        feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=ITER_NUM//2,gamma=0.1)
        
        relation_network_optim = torch.optim.SGD(relation_network.parameters(),lr=LEARNING_RATE, weight_decay=0.0005)
        relation_network_scheduler = StepLR(relation_network_optim,step_size=ITER_NUM//2,gamma=0.1)
        
        # training
        OA = np.zeros(TEST_NUM)
        oa_iter = 0
        test_iter = ITER_NUM // TEST_NUM 
        display_iter = 10
        losses = np.zeros(ITER_NUM+1)
        mean_losses = np.zeros(ITER_NUM+1)
        
        # init torch data
        task_train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
        task_test_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
        
        for iter_ in tqdm(range(1, ITER_NUM + 1), desc='Training the network'):
        
            task_train_gt, rest_gt = sample_gt(train_gt, 1, mode='fixed_withone')
            task_test_gt, rest_gt = sample_gt(train_gt, BATCH_SIZE_PER_CLASS, mode='fixed_withone') 
            #task_test_gt, rest_gt = sample_gt(rest_gt, BATCH_SIZE_PER_CLASS, mode='fixed_withone') #ICME using rest_gt
            # task train
            task_train_dataset.resetGt(task_train_gt)
            task_train_loader = Torchdata.DataLoader(task_train_dataset,
                                           batch_size=N_CLASSES,
                                           shuffle=False)
            
            # task test
            task_test_dataset.resetGt(task_test_gt)
            task_test_loader = Torchdata.DataLoader(task_test_dataset,
                                           batch_size=N_CLASSES*BATCH_SIZE_PER_CLASS,
                                           shuffle=True)
            # sample datas
            samples, sample_labels = task_train_loader.__iter__().next()
            batches, batch_labels = task_test_loader.__iter__().next()
            
            # calculate features
            feature_encoder.train()
            relation_network.train()
            sample_features = feature_encoder(samples.cuda(GPU)) # 
            batch_features = feature_encoder(batches.cuda(GPU)) #
            
            #calculate relations
            sample_features_ext = sample_features.unsqueeze(0).repeat(N_CLASSES*BATCH_SIZE_PER_CLASS,1,1,1,1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(N_CLASSES,1,1,1,1)
            batch_features_ext = torch.transpose(batch_features_ext,0,1)
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,PATCH_SIZE,PATCH_SIZE)
            relations = relation_network(relation_pairs).view(-1,N_CLASSES)

            mse = nn.MSELoss().cuda(GPU)
            one_hot_labels = torch.zeros(N_CLASSES*BATCH_SIZE_PER_CLASS, N_CLASSES).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU)
            loss = mse(relations,one_hot_labels)


            # training

            feature_encoder.zero_grad()
            relation_network.zero_grad()

            loss.backward()

            #torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
            #torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

            feature_encoder_optim.step()
            relation_network_optim.step()
            
            feature_encoder_scheduler.step()
            relation_network_scheduler.step()
            
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0,iter_ - 10):iter_+1])
            if display_iter and iter_ % display_iter == 0:
                string = 'Train (ITER_NUM {}/{})\tLoss: {:.6f}'
                string = string.format(
                    iter_, ITER_NUM, mean_losses[iter_])
                tqdm.write(string)
            
            # Testing
            if iter_ % test_iter == 0:
                print('Testing...')
                feature_encoder.eval()
                relation_network.eval()
                accuracy, total = 0., 0.
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
                        _, output = torch.max(scores,dim=0)
                        accuracy += output.item() == te_labels.item()
                        total +=1
                rate = accuracy / total
                OA[oa_iter] = rate
                oa_iter += 1
                print('Accuracy:', rate)
                # save networks
                save_encoder = 'Bing_Encoder'
                save_relation = 'Bing_Relation'
                with torch.cuda.device(GPU):
                    save_model(feature_encoder, save_encoder, train_loader.dataset.name, sample_size=SAMPLE_SIZE, run=run, epoch=iter_, metric=rate)
                    save_model(relation_network, save_relation, train_loader.dataset.name, sample_size=SAMPLE_SIZE, run=run, epoch=iter_, metric=rate)
                    if iter_ == ITER_NUM:
                        model_encoder_dir = './checkpoints/' + save_encoder + '/' + train_loader.dataset.name + '/'
                        model_relation_dir = './checkpoints/' + save_relation + '/' + train_loader.dataset.name + '/'
                        model_encoder_file = model_encoder_dir + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
                        model_relation_file = model_relation_dir + 'non_augmentation_sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
                        torch.save(feature_encoder.state_dict(), model_encoder_file)
                        torch.save(relation_network.state_dict(), model_relation_file)
        loss_dir = './results/losses/' + DATASET 
        if not os.path.isdir(loss_dir):
            os.makedirs(loss_dir)
        loss_file = loss_dir + '/' + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '_dim' + str(FEATURE_DIM) +'.mat'
        io.savemat(loss_file, {'losses':losses, 'accuracy':OA})
#############################################################################              
def main():
    train()

if __name__ == '__main__':
    main()     
        
