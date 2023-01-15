from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig
from transformers import BertModel,BertConfig,BertTokenizer
import torch.nn.functional as F
from scripts.modules_graph import GraphConvolution,GraphAttentionLayer,GraphSageConv
from utils import args
from transformers import DistilBertTokenizer
import cv2
from matplotlib import pyplot as plt
import torch


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, transform,model_name=args.image_encoder_model, pretrained=args.pretrained, trainable=args.trainable
    ):
        super().__init__()
        # loading the pretrained model
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )

        for p in self.model.parameters():
            p.requires_grad = trainable

        self.transforms = transform
    def forward(self, x):
        return self.model(x)

    def get_finetune_embed(self,img):

        self.model.train()
        image = plt.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        x = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=args.text_encoder_model, pretrained=args.pretrained, trainable=args.trainable):
        super().__init__()
        if pretrained:
            if args.text_encoder_model == 'distilbert-base-uncased':
                self.model = DistilBertModel.from_pretrained(model_name)
                self.tokenizer = DistilBertTokenizer.from_pretrained(args.text_encoder_tokenizer)
            elif args.text_encoder_model == 'bert-base-uncased':
                self.model = BertModel.from_pretrained(model_name)
                self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder_tokenizer)
        else:
            if args.text_encoder_model == 'distilbert-base-uncased':
                self.model = DistilBertModel(config=DistilBertConfig())
                self.tokenizer = DistilBertTokenizer.from_pretrained(args.text_encoder_tokenizer)
            elif args.text_encoder_model == 'bert-base-uncased':
                self.model = BertModel(config=BertConfig())
                self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder_tokenizer)

        for p in self.model.parameters():
            p.requires_grad = trainable

        # Using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

    def get_finetune_embed(self,text):

        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt').to(args.device)
        self.model.train()
        return self.model(**tokens).last_hidden_state[:,self.target_token_idx,:]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=args.projection_dim,
        dropout=args.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class FeatureProjection(nn.Module):
    def __init__(
            self, feature,
            projection_dim = args.feat_projection_dim
    ):
        super().__init__()

        self.projection = {k: nn.Linear(v.shape[1], projection_dim).to(args.device) for k, v in feature.items()}
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, feature):
        project_feature = []
        for k,v in feature.items():
            projected = self.projection[k](v)
            x = self.gelu(projected)
            x = self.layer_norm(x)
            project_feature.append(x)

        if len(project_feature) >1:
            # return torch.cat([project_feature[0], project_feature[1]])
            return torch.cat(project_feature,dim=0)
        else:
            return project_feature[0]
        # return torch.cat([project_feature[0],project_feature[1]])

class NodeEncoder(nn.Module):
    def __init__(self):
        super(NodeEncoder, self).__init__()

        self.dropout = args.node_dropout
        if args.node_encoder_model == 'gcn':
            self.gc1 = GraphConvolution(args.feat_projection_dim, args.hidden_dim)
            self.gc2 = GraphConvolution(args.hidden_dim, args.out_dim)

        elif args.node_encoder_model == 'gat':

            self.attentions = [GraphAttentionLayer(args.feat_projection_dim, args.hidden_dim, dropout=self.dropout, alpha=args.alpha, concat=True) for _ in
                               range(args.nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(args.hidden_dim * args.nheads, args.out_dim, dropout=args.dropout, alpha=args.alpha, concat=False)
        elif args.node_encoder_model == 'sage':
            self.sage1 = GraphSageConv(args.feat_projection_dim, args.hidden_dim)
            self.sage2 = GraphSageConv(args.hidden_dim, args.out_dim)


    def forward(self, x, adj):
        if args.node_encoder_model == 'gcn':
            x1 = F.relu(self.gc1(x, adj))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = self.gc2(x1, adj)

        elif args.node_encoder_model == 'gat':
            x = F.dropout(x, self.dropout, training=self.training)
            x1 = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.elu(self.out_att(x1, adj))

        elif args.node_encoder_model == 'sage':

            x1 = F.relu(self.sage1(x, adj))
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x2 = F.relu(self.sage2(x1, adj))
            x2 = F.dropout(x2, self.dropout, training=self.training)

        return x2


class NodeClassifier(nn.Module):
    def __init__(self):
        super(NodeClassifier, self).__init__()

        self.nodencoder = NodeEncoder()
        self.mlp = nn.Linear(args.out_dim, args.nclass)
        self.dropout = args.dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.nodencoder(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x[:,:])

        return x


class Classifier(nn.Module):
    def __init__(self,input_dim= args.out_dim,output_dim=args.nclass):
        super(Classifier, self).__init__()


        self.mlp = nn.Linear(input_dim, output_dim)
        self.dropout = args.dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x):
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x[:,:])

        return x


class FineTuneProjection(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.layer_norm(x)
        return x
