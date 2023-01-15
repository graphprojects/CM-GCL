#!/usr/bin/env python
# encoding: utf-8


import argparse
import torch
import sys

# argv = sys.argv
# # dataset = argv[1]
# # dataset = 'acm'
# # dataset = 'aminer'


def aminer_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_entity_matching_path', type=str, default=r".\data\aminer_data\entityid_text_label.txt",help='path about the matching of node id and text id')
    parser.add_argument('--author_feature_path', type=str,default=r'.\data\aminer_data\id_author_feature.txt',help='path of aminer feature')
    parser.add_argument('--feature_path', type=str,default=r'.\data\aminer_data\keyword_feature.txt',help='path of aminer feature')
    parser.add_argument('--relation_path', type=str,default=r'.\data\aminer_data\relation.txt', help='path of aminer relationships')
    parser.add_argument('--pos_path', type=str, default=r'.\data\aminer_data\pos_index_5.txt',help='path of aminer pos label')
    parser.add_argument('--id_content_path', type=str, default=r'.\data\aminer_data\id_label_content.csv',help='to get the finetune features')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--ft_epochs', type=int, default=200,
                        help='Number of epochs to train during fine-tuning.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--node_dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--prune', default=True, help='network pruning for model pre-training')
    parser.add_argument('--prune_ratio', type=float, default=0.2, help='network pruning for model fine-tuning')

    parser.add_argument('--number_samples', type=int, default=1000,
                        help='number of samples in co-modality pre-training')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--finetune_device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--image_encoder_model', type=str, default='swin_small_patch4_window7_224',
                        help='resnet50, please refer to for more models')
    # parser.add_argument('--text_encoder_model', type=str, default='bert-base-uncased',
    #                     help='[bert-base-uncased,distilbert-base-uncased]')
    # parser.add_argument('--text_encoder_tokenizer', type=str, default='bert-base-uncased',
    #                     help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--text_encoder_model', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--text_encoder_tokenizer', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--max_length', type=int, default=70, help='the length of text')

    parser.add_argument('--node_encoder_model', type=str, default='gcn', help='[gcn,gat,sage,gin]')

    parser.add_argument('--nheads', type=int, default=8, help='gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='gat')
    parser.add_argument('--image_size', type=int, default=224, help='the size of image')
    parser.add_argument('--imbalance_setting', type=str, default='focal', help='[reweight,upsample,focal]')
    parser.add_argument('--imbalance_up_scale', type=float, default=10.0, help='the scale for upsampling')

    parser.add_argument('--imbalance_ratio', type=float, default= 0,
                        help='[0.01, 0.1, 0, 1] if 0 then original imbalance ratio if 1 then balanced data')
    parser.add_argument('--node_feature_dim', type=int, default=768, help='the dimension of aminer feature')
    parser.add_argument('--node_feature_project_dim', type=int, default=200, help='the dimension of projected feature')
    parser.add_argument('--node_embedding_dim', type=int, default=200, help='the dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--nclass', type=int, default=5, help='the number of class')

    parser.add_argument('--num_projection_layers', type=int, default=1,
                        help='the number of project layer for text/image and node')
    parser.add_argument('--feat_projection_dim', type=int, default=200,
                        help='the dimension of projected feature for nodes')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='the dimension of projected embedding of text/image and node')
    parser.add_argument('--pos', default=True)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch_size', type=int, default=100,help='the size for each batch for co-modality pretraining')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers')
    parser.add_argument('--patience', type=int, default=30, help='the number of epochs that the metric is not improved')
    parser.add_argument('--factor', type=int, default=0.5, help='the factor to change the learning rate')
    parser.add_argument('--temperature', type=int, default=0.1, help='the factor to change the learning rate')
    parser.add_argument('--pretrained', default=True, help="for text/image encoder")
    parser.add_argument('--trainable', default=False, help="for text/image encoder")
    parser.add_argument('--image_embedding_dim', type=int, default=768, help='the dimension of image embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='the dimension of text embedding')
    parser.add_argument('--finetune_embedding_dim', type=int, default=768, help='the dimension of finetuen feature')
    parser.add_argument('--focal_alpha', type=int, default=0.75, help='focal loss')
    parser.add_argument('--focal_gamma', type=int, default=1.0, help='focal loss')
    parser.add_argument('--finetune', default=True, help='finetune the model')

    args, _ = parser.parse_known_args()

    return args

def yelp_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_entity_matching_path', type=str, default=r'',help='path about the matching of node id and text id')
    parser.add_argument('--feature_path', type=str, default=r'', help='path of yelp text feature')    # base data here here  with 0.25
    parser.add_argument('--embed_path', type=str, default=r'',
                        help='path of yelp feature')
    parser.add_argument('--relation_path', type=str, default=r'',help='path of yelp relationships')
    parser.add_argument('--pos_path', type=str, default=r'', help='path of yelp pos label')
    parser.add_argument('--yelp_concate',default=False, help='whether concatenate yelp feature and embedding')
    parser.add_argument('--id_content_path', type=str, default=r'',help='to get the finetune features')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--pretrain_epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--ft_epochs', type=int, default=60,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--node_dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--prune', default=True, help='network pruning for model pre-training')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='network pruning for model fine-tuning')

    parser.add_argument('--number_samples', type=int, default=1000,
                        help='number of samples in co-modality pre-training')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--finetune_device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--image_encoder_model', type=str, default='swin_small_patch4_window7_224',
                        help='resnet50, please refer to for more models')
    parser.add_argument('--text_encoder_model', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--text_encoder_tokenizer', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--max_length', type=int, default=200, help='the length of text')
    parser.add_argument('--feat_projection_dim', type=int, default=200,
                        help='the dimension of projected feature for nodes')
    parser.add_argument('--node_encoder_model', type=str, default='gcn', help='[gcn,gat,sage,gin]')
    parser.add_argument('--nheads', type=int, default=8, help='gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='gat')
    parser.add_argument('--image_size', type=int, default=224, help='the size of image')
    parser.add_argument('--imbalance_setting', type=str, default='focal', help='[reweight,upsample,focal]')
    parser.add_argument('--imbalance_up_scale', type=float, default=10.0, help='the scale for upsampling')

    parser.add_argument('--imbalance_ratio', type=float, default=0,
                        help='[0.01, 0.1, 0, 1] if 0 then original imbalance ratio if 1 then balanced data')

    parser.add_argument('--node_feature_dim', type=int, default=768, help='the dimension of yelp feature')
    parser.add_argument('--node_feature_project_dim', type=int, default=200, help='the dimension of projected feature')
    parser.add_argument('--node_embedding_dim', type=int, default=200, help='the dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--nclass', type=int, default=2, help='the number of class')

    parser.add_argument('--num_projection_layers', type=int, default=1,
                        help='the number of project layer for text/image and node')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='the dimension of projected embedding of text/image and node')

    parser.add_argument('--pos', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch_size', type=int, default=50,
                        help='the size for each batch for co-modality pretraining')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers')
    parser.add_argument('--patience', type=int, default=1, help='the number of epochs that the metric is not improved')
    parser.add_argument('--factor', type=int, default=0.5, help='the factor to change the learning rate')
    parser.add_argument('--temperature', type=int, default=1.0, help='the factor to change the learning rate')

    parser.add_argument('--pretrained', default=True,help = "for text/image encoder")
    parser.add_argument('--trainable', default=False, help = "for text/image encoder")

    parser.add_argument('--image_embedding_dim', type=int, default=768, help='the dimension of image embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='the dimension of text embedding')
    parser.add_argument('--finetune_embedding_dim', type=int, default=768, help='the dimension of finetuen feature')
    parser.add_argument('--focal_alpha', type=int, default=0.5, help='focal loss')
    parser.add_argument('--focal_gamma', type=int, default=0.6, help='focal loss')
    parser.add_argument('--finetune', default=True, help='finetune the model')


    args, _ = parser.parse_known_args()

    return args

def github_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_entity_matching_path', type=str,default=r'',help='path about the matching of node id and text id')
    parser.add_argument('--feature_path', type=str, default=r'',
                        help='path of github feature')
    parser.add_argument('--relation_path', type=str, default=r'',
                        help='path of github relationships')
    parser.add_argument('--pos_path', type=str, default=r'',
                        help='path of github pos label')
    parser.add_argument('--label_path', type=str, default=r'',
                        help='path of label')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--pretrain_epochs', type=int, default=60,
                        help='Number of epochs to train.')
    parser.add_argument('--ft_epochs', type=int, default=60,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--node_dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--prune', default=True, help='network pruning for model pre-training')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='network pruning for model fine-tuning')

    parser.add_argument('--number_samples', type=int, default=1000,
                        help='number of samples in co-modality pre-training')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--finetune_device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--id_content_path', type=str,
                        default=r'',
                        help='to get the finetune features')
    parser.add_argument('--image_encoder_model', type=str, default='swin_small_patch4_window7_224',
                        help='resnet50, please refer to for more models')
    parser.add_argument('--text_encoder_model', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--text_encoder_tokenizer', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--max_length', type=int, default=200, help='the length of text')
    parser.add_argument('--feat_projection_dim', type=int, default=200,
                        help='the dimension of projected feature for nodes')
    parser.add_argument('--node_encoder_model', type=str, default='gcn', help='[gcn,gat,sage,gin]')
    parser.add_argument('--nheads', type=int, default=8, help='gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='gat')
    parser.add_argument('--image_size', type=int, default=224, help='the size of image')
    parser.add_argument('--imbalance_setting', type=str, default='focal', help='[reweight,upsample,focal]')
    parser.add_argument('--imbalance_up_scale', type=float, default=10.0, help='the scale for upsampling')

    parser.add_argument('--imbalance_ratio', type=float, default=0,
                        help='[0.01, 0.1, 0, 1] if 0 then original imbalance ratio if 1 then balanced data')

    parser.add_argument('--node_feature_dim', type=int, default=128, help='the dimension of github feature')
    parser.add_argument('--node_feature_project_dim', type=int, default=200, help='the dimension of projected feature')
    parser.add_argument('--node_embedding_dim', type=int, default=200, help='the dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--nclass', type=int, default=2, help='the number of class')

    parser.add_argument('--num_projection_layers', type=int, default=1,
                        help='the number of project layer for text/image and node')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='the dimension of projected embedding of text/image and node')

    parser.add_argument('--pos', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch_size', type=int, default=80,
                        help='the size for each batch for co-modality pretraining')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of workers')
    parser.add_argument('--patience', type=int, default=5, help='the number of epochs that the metric is not improved')
    parser.add_argument('--factor', type=int, default=0.5, help='the factor to change the learning rate')
    parser.add_argument('--temperature', type=int, default=1.0, help='the factor to change the learning rate')

    parser.add_argument('--pretrained', default=True,help = "for text/image encoder")
    parser.add_argument('--trainable', default=True, help = "for text/image encoder")
    parser.add_argument('--image_embedding_dim', type=int, default=768, help='the dimension of image embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='the dimension of text embedding')
    parser.add_argument('--finetune_embedding_dim', type=int, default=768, help='the dimension of finetuen feature')
    parser.add_argument('--focal_alpha', type=int, default=0.25, help='focal loss')
    parser.add_argument('--focal_gamma', type=int, default=2.0, help='focal loss')
    parser.add_argument('--finetune', default=True, help='finetune the model')

    args, _ = parser.parse_known_args()

    return args

def instagram_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--node_entity_matching_path', type=str,
                        default=r'',
                        help='path about the matching of node id and picture id')
    parser.add_argument('--feature_path', type=str, default=r'',
                        help='path of instagram feature')
    parser.add_argument('--relation_path', type=str,
                        default=r'',
                        help='path of instagram relationships')
    parser.add_argument('--pos_path', type=str, default=r'',
                        help='path of instagram pos label')
    parser.add_argument('--id_content_path', type=str, default=r'',
                        help='to get the finetune features')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--pretrain_epochs', type=int, default=1,
                        help='Number of epochs to train.')   # 60
    parser.add_argument('--ft_epochs', type=int, default=60,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--node_dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--prune', default=True, help='network pruning for model pre-training')
    parser.add_argument('--prune_ratio', type=float, default=0.5, help='network pruning for model fine-tuning')

    parser.add_argument('--number_samples', type=int, default=1000,
                        help='number of samples in co-modality pre-training')
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--finetune_device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='use GPU')
    parser.add_argument('--image_encoder_model', type=str, default='swin_small_patch4_window7_224',
                        help='resnet50, please refer to for more models')
    parser.add_argument('--text_encoder_model', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--text_encoder_tokenizer', type=str, default='distilbert-base-uncased',
                        help='[bert-base-uncased,distilbert-base-uncased]')
    parser.add_argument('--max_length', type=int, default=200, help='the length of text')
    parser.add_argument('--feat_projection_dim', type=int, default=200,
                        help='the dimension of projected feature for nodes')

    parser.add_argument('--node_encoder_model', type=str, default='gcn', help='[gcn,gat,sage,gin]')
    parser.add_argument('--nheads', type=int, default=8, help='gat')
    parser.add_argument('--alpha', type=float, default=0.2, help='gat')

    parser.add_argument('--imbalance_setting', type=str, default='focal', help='[reweight,upsample,focal]')
    parser.add_argument('--imbalance_up_scale', type=float, default=10.0, help='the scale for upsampling')

    parser.add_argument('--imbalance_ratio', type=float, default=0,
                        help='[0.01, 0.1, 0, 1] if 0 then original imbalance ratio if 1 then balanced data')

    parser.add_argument('--node_feature_dim', type=int, default=768, help='the dimension of instagram feature')
    parser.add_argument('--node_feature_project_dim', type=int, default=200, help='the dimension of projected feature')
    parser.add_argument('--node_embedding_dim', type=int, default=200, help='the dimension of node embedding')
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--nclass', type=int, default=2, help='the number of class')

    parser.add_argument('--image_size', type=int, default=224, help='the size of image')
    parser.add_argument('--image_embedding_dim', type=int, default=768, help='the dimension of image embedding')
    parser.add_argument('--text_embedding_dim', type=int, default=768, help='the dimension of text embedding')
    parser.add_argument('--finetune_embedding_dim', type=int, default=768, help='the dimension of finetuen feature')

    parser.add_argument('--num_projection_layers', type=int, default=1,
                        help='the number of project layer for text/image and node')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='the dimension of projected embedding of text/image and node')

    parser.add_argument('--pos', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--batch_size', type=int, default=30,
                        help='the size for each batch for co-modality pretraining')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers')
    parser.add_argument('--patience', type=int, default=5, help='the number of epochs that the metric is not improved')
    parser.add_argument('--factor', type=int, default=0.5, help='the factor to change the learning rate')
    parser.add_argument('--temperature', type=int, default=1.0, help='the factor to change the learning rate')

    parser.add_argument('--pretrained', default=True,help = "for text/image encoder")
    parser.add_argument('--trainable', default=True, help = "for text/image encoder")

    parser.add_argument('--focal_alpha', type=int, default=0.25, help='focal loss')
    parser.add_argument('--focal_gamma', type=int, default=1.0, help='focal loss')
    parser.add_argument('--finetune', default=True, help='finetune the model')

    args, _ = parser.parse_known_args()

    return args


def set_params(dataset):
    if dataset == "aminer":
        args = aminer_params()
    elif dataset == "yelp":
        args = yelp_params()
    elif dataset == "github":
        args = github_params()
    elif dataset == "instagram":
        args = instagram_params()

    args.dataset = dataset

    return args


dataset = 'aminer'
# dataset = 'yelp'
# dataset = 'github'
# dataset = 'instagram'
args = set_params(dataset)
