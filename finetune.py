#!/usr/bin/env python
# encoding: utf-8


from utils.getdata import load_data_for_finetune,load_data_for_finetune_imbalance,load_data_for_finetune_balance
import torch.optim as optim
from scripts.CMCL import NodeTextCLModel,NodeImageCLModel
from utils.util import seed_torch,evaluate_performance
import os
# from torchgeometry.losses import FocalLoss
from utils.focalloss import FocalLoss
from scripts.modules_model import Classifier,FineTuneProjection
import torch
from utils import args
import torch.nn as nn
import warnings
import numpy as np

# warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



class FineTune(nn.Module):
    def __init__(self, pretrained_encoder):
        super(FineTune, self).__init__()

        self.ft_project = FineTuneProjection(args.finetune_embedding_dim, args.node_feature_project_dim)
        self.ft_encoder = pretrained_encoder
        self.classifier = Classifier(input_dim=args.node_feature_project_dim, output_dim=args.nclass)


    def forward(self, features, adj):

        features_pro = self.ft_project(features)
        embed = self.ft_encoder(features_pro, adj)
        output = self.classifier(embed)

        return output

def fine_tune(pretrained_model_dir,finetuned_feature_dir):

    best_f1_macro = 0.0
    best_epoch = 0
    best_model = {}

    #  imbalanced data with initial ratio
    if args.imbalance_ratio == 0:

        adj, features, labels, idx_train, idx_val, idx_test = load_data_for_finetune(finetuned_feature_dir)

    #  balanced data
    elif args.imbalance_ratio == 1:
        adj, features, labels, idx_train, idx_val, idx_test = load_data_for_finetune_balance(finetuned_feature_dir)

    #  imbalanced data with specific ratio
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_data_for_finetune_imbalance(finetuned_feature_dir)

    # features = torch.from_numpy(np.genfromtxt(finetuned_feature_dir, delimiter=' ')).float()

    if args.dataset != 'instagram':
        model = NodeTextCLModel(features)
    else:
        model = NodeImageCLModel(features)

    model.eval()
    model_dict = torch.load(pretrained_model_dir)
    model_dict_copy = {}
    for key,v in model_dict.items():
        if '_orig' in key:
            model_dict_copy[key.replace('_orig','')] = v
        elif '_mask' in key:
            continue
        else:
            model_dict_copy[key] = v

    model.load_state_dict(model_dict_copy)

    ft_encoder = model.node_encoder
    model.eval()

    ft_model = FineTune(ft_encoder)

    optimizer = optim.Adam(ft_model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    ft_model.train()

    focaloss = FocalLoss(device=args.finetune_device, alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')

    ft_model = ft_model.to(args.finetune_device)
    features = features['features'].to(args.finetune_device)
    adj = adj.to(args.finetune_device)
    labels = labels.to(args.finetune_device)
    idx_train = idx_train.to(args.finetune_device)
    idx_val = idx_val.to(args.finetune_device)
    idx_test = idx_test.to(args.finetune_device)
    focaloss = focaloss.to(args.finetune_device)

    for epoch in range(args.ft_epochs):
        optimizer.zero_grad()

        output = ft_model(features, adj)

        loss_train = focaloss(output[idx_train], labels[idx_train])
        f1_micro_train, f1_macro_train, auc_train = evaluate_performance(labels[idx_train], output[idx_train])

        loss_train.backward()
        optimizer.step()

        with torch.no_grad():
            ft_model.eval()

            output = ft_model(features, adj)

            f1_micro_val, f1_macro_val, auc_val = evaluate_performance(labels[idx_val],output[idx_val])
            if best_f1_macro < f1_macro_val:

                best_f1_macro = f1_macro_val
                best_epoch = epoch

                print(' Saving model ...')
                best_model = ft_model.state_dict().copy()

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'f1_micro_train: {:.4f}'.format(f1_micro_train),
                  'f1_macro_train: {:.4f}'.format(f1_macro_train),
                  'auc_train: {:.4f}'.format(auc_train),

                  'f1_micro_val: {:.4f}'.format(f1_micro_val),
                  'f1_macro_val: {:.4f}'.format(f1_macro_val),
                  'auc_val: {:.4f}'.format(auc_val)
                  )

    print("Load model from epoch {}".format(best_epoch))
    ft_model.load_state_dict(best_model)

    ft_model.eval()


    output = ft_model(features, adj)

    f1_micro_test, f1_macro_test, auc_test = evaluate_performance(labels[idx_test], output[idx_test])

    print('Model Testing:',
          'f1_micro_test: {:.4f}'.format(f1_micro_test),
          'f1_macro_test: {:.4f}'.format(f1_macro_test),
          'auc_test: {:.4f}'.format(auc_test)
          )

if __name__ == "__main__":

    finetune_feature_path = r'./finetune/aminer_data/finetune_feature_202301141028.txt'
    pretrained_model_dir = r'./pretrain/aminer_node_text_202301141028.pt'

    fine_tune(pretrained_model_dir,finetune_feature_path)

