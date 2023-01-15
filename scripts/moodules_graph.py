import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.modules_layer import GraphConvolution,GraphSageConv,GraphAttentionLayer
from utils.params import args

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,dataset,feature,projection_dim=200):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.dataset = dataset
        if dataset == 'aminer':
            self.projection = FeatureProjection(feature,projection_dim).to(args.device)


    def forward(self, x, adj):
        if self.dataset == 'aminer':
            x = self.projection(x)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x =F.dropout(F.relu(self.gc2(x, adj)))
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,dataset,feature,projection_dim=200):
        super(SAGE, self).__init__()

        self.sage1 = GraphSageConv(nfeat, nhid)
        self.sage2 = GraphSageConv(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.reset_parameters()

        self.dropout = dropout
        self.dataset = dataset
        if dataset == 'aminer':
            self.projection = FeatureProjection(feature,projection_dim).to(args.device)

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        if self.dataset == 'aminer':
            x = self.projection(x)
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.sage2(x, adj)))
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

class SAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,dataset,feature,projection_dim=200):
        super(SAGE, self).__init__()

        self.sage1 = GraphSageConv(nfeat, nhid)
        self.sage2 = GraphSageConv(nhid, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.reset_parameters()

        self.dropout = dropout
        self.dataset = dataset
        if dataset == 'aminer':
            self.projection = FeatureProjection(feature,projection_dim).to(args.device)

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        if self.dataset == 'aminer':
            x = self.projection(x)
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.sage2(x, adj)))
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dataset,feature,projection_dim=200,alpha=0.2, nheads=1):
        super(GAT, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        if args.dataset == 'aminer':
            self.projection = FeatureProjection(feature, projection_dim).to(args.device)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

        self.dataset = dataset

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)
        nn.init.normal_(self.out_proj.weight, std=0.05)

    def forward(self, x, adj):
        if self.dataset == 'aminer':
            x = self.projection(x)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

class FeatureProjection(nn.Module):
    def __init__(
            self, feature,
            projection_dim
    ):
        super().__init__()
        # device = torch.device("cuda:0" if  torch.cuda.is_available() else "cpu")
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

        if len(project_feature) ==2:
            return torch.cat([project_feature[0],project_feature[1]])
        elif len(project_feature) ==3:
            return torch.cat([project_feature[0],project_feature[1],project_feature[2]])
        else:
            return torch.tensor(project_feature[0].clone().detach())


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        if isinstance(adj, torch.sparse.FloatTensor):
            adj = adj.to_dense()

        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



