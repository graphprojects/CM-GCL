import cv2
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import scipy.sparse as sp
import torch
import random
from utils.params import args


class NodeImageDataset():
    def __init__(self, entity_id,image,label, transforms):


        self.image_filenames = list(image)
        self.entity_id = entity_id
        self.transforms = transforms
        self.label = label

    def __getitem__(self, idx):

        item = {}
        # print(idx)

        try:
            image = plt.imread(self.image_filenames[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
        except:
            image = np.random.randn(224,224,3)
            # print("error happened")
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['entity'] = self.entity_id[idx]
        item['label'] = self.label[idx]

        return item

    def __len__(self):
        return len(self.image_filenames)


class NodeTextDataset():
    def __init__(self, entity_id,text,label,tokenizer):

        self.text = list(text)

        self.encoded_text = tokenizer(
            self.text, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt'
        )

        self.entity_id = entity_id
        self.label = label

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_text.items()
        }
        item['text'] = self.text[idx]
        item['entity'] = self.entity_id[idx]
        item['label'] = self.label[idx]

        return item


    def __len__(self):
        return len(self.text)


# get transformation for image data
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(args.image_size, args.image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(args.image_size, args.image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

def split_imbalance(labels,train_ratio,val_ratio,test_ratio,imbalance_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    label_max = int(max(labels.tolist())+1)
    minority_index = [item for item in range(label_max) if labels.tolist().count(item) <  len(labels.tolist())/num_classes]

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        if num_classes > 2:
            if i in minority_index: c_num = int(len(labels.tolist()) / num_classes * imbalance_ratio)
        else:
            if i in minority_index: c_num = int((len(labels.tolist())- labels.tolist().count(i)) * imbalance_ratio)

        print('The number of class {:d}: {:d}'.format(i,c_num))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)


    return train_idx, val_idx, test_idx, c_num_mat

def split_balance(labels,train_ratio,val_ratio,test_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = []
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    count_0, count_1 = labels.tolist().count(0), labels.tolist().count(1)
    count_min = min(count_0,count_1)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = count_min

        print('The number of class {:d}: {:d}'.format(i,c_num))

        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels,train_ratio,val_ratio,test_ratio):

    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('The number of class {:d}: {:d}'.format(i,c_num))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/10 *train_ratio)
            c_num_mat[i,1] = int(c_num/10 * val_ratio)
            c_num_mat[i,2] = int(c_num/10 * test_ratio)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]


    return train_idx, val_idx, test_idx, c_num_mat

def load_data_for_pretrain():

    if args.dataset == 'instagram':
        embed_dir = args.feature_path
        relation_dir = args.relation_path
        pos_dir = args.pos_path

        idx_features_labels = np.round(np.genfromtxt(embed_dir,
                                            dtype=np.float, delimiter=' ', invalid_raise=True),4)

        features = idx_features_labels[:, 770:]
        features = sp.csr_matrix(features, dtype=np.float32)
        total_num = features.shape[0]
        features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}


    elif args.dataset == 'github':

        embed_dir = args.feature_path

        relation_dir = args.relation_path
        pos_dir = args.pos_path

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = idx_features_labels[:,0]
        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}

        total_num = features.shape[0]

    elif args.dataset == 'yelp':

        fea_dir = args.feature_path
        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(fea_dir, dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        pos_dir = args.pos_path
        if args.yelp_concate:
            embed_dir = args.embed_path
            idx_embeds_labels = np.genfromtxt(embed_dir,
                                              dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
            embeds = idx_embeds_labels[:, 2:]

            features = np.concatenate((features, embeds), axis=1)

        features = sp.csr_matrix(features, dtype=np.float32)
        total_num = features.shape[0]
        features = normalize(features)

        number = 67395
        label = torch.tensor(encode_onehot(idx_features_labels[:number, 1]))

        # build graph
        idx = np.array([i for i in range(features.shape[0])])
        idx_map = {j: i for i, j in enumerate(idx)}

        features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}

    elif args.dataset == 'aminer':


        fea_dir = args.feature_path
        author_fea_dir = args.author_feature_path

        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True, skip_header=True)
        paper_features = idx_features_labels[:, 2:]
        author_features = np.genfromtxt(author_fea_dir, dtype=np.dtype(str), delimiter=',', invalid_raise=True,
                                        skip_header=True)[:, 2:]

        number = 18089

        paper_features = sp.csr_matrix(paper_features, dtype=np.float32)
        author_features = sp.csr_matrix(author_features, dtype=np.float32)
        paper_features = normalize(paper_features)
        paper_features = torch.FloatTensor(np.array(paper_features.todense()))
        author_features = torch.FloatTensor(np.array(author_features.todense()))
        features = {'paper_feature': paper_features.to(args.device), 'author_features': author_features.to(args.device)}
        label = encode_onehot(idx_features_labels[:number, 1].astype('float'))

        # build graph
        idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        total_num = paper_features.shape[0] + author_features.shape[0]
        idx_map = {j: i for i, j in enumerate(idx)}

        pos_dir = args.pos_path

    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(total_num,total_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    pos_unordered = np.genfromtxt(pos_dir,
                                    dtype=np.int32)
    pos = np.array(list(map(idx_map.get, pos_unordered.flatten())),
                     dtype=np.int32).reshape(pos_unordered.shape)
    pos = sp.coo_matrix((np.ones(pos.shape[0]), (pos[:, 0], pos[:, 1])),
                        shape=(number,number),
                        dtype=np.int32)
    # build symmetric adjacency matrix
    pos = pos + pos.T.multiply(pos.T > pos) - pos.multiply(pos.T > pos)
    pos = pos + sp.eye(pos.shape[0])


    labels = torch.LongTensor(np.where(label)[1]).to(args.device)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(args.device)
    pos = sparse_mx_to_torch_sparse_tensor(pos).to(args.device)

    return adj, features, labels, pos

def load_data_for_finetune(finetuned_feature_dir):

    if args.dataset == 'instagram':
        embed_dir = args.feature_path
        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'github':

        embed_dir = args.feature_path

        relation_dir = args.relation_path

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = idx_features_labels[:,0]

        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'yelp':

        fea_dir = args.feature_path
        relation_dir = args.relation_path
        number = 67395

        idx_features_labels = np.genfromtxt(fea_dir, dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        features = normalize(features)
        features = sp.csr_matrix(features, dtype=np.float32)

        total_num = features.shape[0]
        label_data = idx_features_labels[:number,1]
        label = torch.tensor(encode_onehot(label_data))

        # build graph
        idx = np.array([i for i in range(features.shape[0])])
        idx_map = {j: i for i, j in enumerate(idx)}

        features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}

    elif args.dataset == 'aminer':

        fea_dir = args.feature_path

        author_fea_dir = args.author_feature_path

        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True, skip_header=True)
        features = torch.from_numpy(np.genfromtxt(finetuned_feature_dir, delimiter=' ')).float()
        features = sp.csr_matrix(features, dtype=np.float32)
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        features = {'features': features.to(args.device)}

        paper_features = idx_features_labels[:, 2:]
        author_features = np.genfromtxt(author_fea_dir, dtype=np.dtype(str), delimiter=',', invalid_raise=True,
                                        skip_header=True)[:, 2:]
        number = 18089

        # paper_features = sp.csr_matrix(paper_features, dtype=np.float32)
        # author_features = sp.csr_matrix(author_features, dtype=np.float32)
        # paper_features = normalize(paper_features)
        # paper_features = torch.FloatTensor(np.array(paper_features.todense()))
        # author_features = torch.FloatTensor(np.array(author_features.todense()))
        # features = {'paper_feature': paper_features.to(args.device), 'author_features': author_features.to(args.device)}
        label_data = idx_features_labels[:number, 1].astype('float')
        label = encode_onehot(label_data)



        # build graph
        # idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        # total_num = paper_features.shape[0] + author_features.shape[0]

        idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        total_num = paper_features.shape[0] + author_features.shape[0]
        idx_map = {j: i for i, j in enumerate(idx)}


    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(total_num, total_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    if args.dataset == 'yelp':
        idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data), train_ratio=4, val_ratio=2,
                                                                test_ratio=4)
    else:
        idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data),train_ratio=7,val_ratio=1,test_ratio=2)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    labels = torch.LongTensor(np.where(label)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_for_finetune_imbalance(finetuned_feature_dir):

    if args.dataset == 'instagram':
        embed_dir = args.feature_path
        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'github':

        embed_dir = args.feature_path

        relation_dir = args.relation_path

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = idx_features_labels[:,0]

        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'yelp':

        fea_dir = args.feature_path
        relation_dir = args.relation_path
        number = 67395

        idx_features_labels = np.genfromtxt(fea_dir, dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        features = normalize(features)
        features = sp.csr_matrix(features, dtype=np.float32)

        total_num = features.shape[0]
        label_data = idx_features_labels[:number,1]
        label = torch.tensor(encode_onehot(label_data))

        # build graph
        idx = np.array([i for i in range(features.shape[0])])
        idx_map = {j: i for i, j in enumerate(idx)}

        features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}

    elif args.dataset == 'aminer':

        fea_dir = args.feature_path

        author_fea_dir = args.author_feature_path

        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True, skip_header=True)
        features = torch.from_numpy(np.genfromtxt(finetuned_feature_dir, delimiter=' ')).float()
        features = sp.csr_matrix(features, dtype=np.float32)
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        features = {'features': features.to(args.device)}

        paper_features = idx_features_labels[:, 2:]
        author_features = np.genfromtxt(author_fea_dir, dtype=np.dtype(str), delimiter=',', invalid_raise=True,
                                        skip_header=True)[:, 2:]
        number = 18089

        # paper_features = sp.csr_matrix(paper_features, dtype=np.float32)
        # author_features = sp.csr_matrix(author_features, dtype=np.float32)
        # paper_features = normalize(paper_features)
        # paper_features = torch.FloatTensor(np.array(paper_features.todense()))
        # author_features = torch.FloatTensor(np.array(author_features.todense()))
        # features = {'paper_feature': paper_features.to(args.device), 'author_features': author_features.to(args.device)}
        label_data = idx_features_labels[:number, 1].astype('float')
        label = encode_onehot(label_data)



        # build graph
        # idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        # total_num = paper_features.shape[0] + author_features.shape[0]

        idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        total_num = paper_features.shape[0] + author_features.shape[0]
        idx_map = {j: i for i, j in enumerate(idx)}


    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(total_num, total_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    if args.dataset == 'yelp':
        # idx_train, idx_val, idx_test, c_num_mat = split_genuine(torch.tensor(label_data),train_ratio=7,val_ratio=1,test_ratio=2)
        idx_train, idx_val, idx_test, c_num_mat = split_imbalance(torch.tensor(label_data), train_ratio=4, val_ratio=2,
                                                                  test_ratio=2, imbalance_ratio=args.imbalance_ratio)
    else:
        idx_train, idx_val, idx_test, c_num_mat = split_imbalance(torch.tensor(label_data), train_ratio=7, val_ratio=1,
                                                                  test_ratio=2, imbalance_ratio=args.imbalance_ratio)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    labels = torch.LongTensor(np.where(label)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_for_finetune_balance(finetuned_feature_dir):

    if args.dataset == 'instagram':
        embed_dir = args.feature_path
        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = sp.csr_matrix(idx_features_labels[:, 2:], dtype=np.float32)
        number = 8225
        label = encode_onehot(idx_features_labels[:number, 1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'github':

        embed_dir = args.feature_path

        relation_dir = args.relation_path

        idx_features_labels = np.genfromtxt(embed_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True)

        features = sp.csr_matrix(idx_features_labels[:, :], dtype=np.float32)

        label_data = idx_features_labels[:,0]

        label = encode_onehot(label_data)

        # build graph
        idx = np.array(list(range(idx_features_labels.shape[0])), dtype=np.int32)
        # idx = np.array(idx_features_labels[:, 0], dtype=np.float)
        idx_map = {j: i for i, j in enumerate(idx)}

    elif args.dataset == 'yelp':

        fea_dir = args.feature_path
        relation_dir = args.relation_path
        number = 67395

        idx_features_labels = np.genfromtxt(fea_dir, dtype=np.dtype(str), delimiter=' ', invalid_raise=True)
        features = idx_features_labels[:, 2:]
        features = normalize(features)
        features = sp.csr_matrix(features, dtype=np.float32)

        total_num = features.shape[0]
        label_data = idx_features_labels[:number,1]
        label = torch.tensor(encode_onehot(label_data))

        # build graph
        idx = np.array([i for i in range(features.shape[0])])
        idx_map = {j: i for i, j in enumerate(idx)}

        features = {'feature': torch.FloatTensor(np.array(features.todense())).to(args.device)}

    elif args.dataset == 'aminer':

        fea_dir = args.feature_path

        author_fea_dir = args.author_feature_path

        relation_dir = args.relation_path
        idx_features_labels = np.genfromtxt(fea_dir,
                                            dtype=np.dtype(str), delimiter=',', invalid_raise=True, skip_header=True)
        features = torch.from_numpy(np.genfromtxt(finetuned_feature_dir, delimiter=' ')).float()
        features = sp.csr_matrix(features, dtype=np.float32)
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
        features = {'features': features.to(args.device)}

        paper_features = idx_features_labels[:, 2:]
        author_features = np.genfromtxt(author_fea_dir, dtype=np.dtype(str), delimiter=',', invalid_raise=True,
                                        skip_header=True)[:, 2:]
        number = 18089

        # paper_features = sp.csr_matrix(paper_features, dtype=np.float32)
        # author_features = sp.csr_matrix(author_features, dtype=np.float32)
        # paper_features = normalize(paper_features)
        # paper_features = torch.FloatTensor(np.array(paper_features.todense()))
        # author_features = torch.FloatTensor(np.array(author_features.todense()))
        # features = {'paper_feature': paper_features.to(args.device), 'author_features': author_features.to(args.device)}
        label_data = idx_features_labels[:number, 1].astype('float')
        label = encode_onehot(label_data)



        # build graph
        # idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        # total_num = paper_features.shape[0] + author_features.shape[0]

        idx = np.array([i for i in range(paper_features.shape[0] + author_features.shape[0])])
        total_num = paper_features.shape[0] + author_features.shape[0]
        idx_map = {j: i for i, j in enumerate(idx)}


    edges_unordered = np.genfromtxt(relation_dir,
                                    dtype=np.int32)[:, :-1]

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(total_num, total_num),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    if args.dataset == 'yelp':
        idx_train, idx_val, idx_test, c_num_mat = split_balance(torch.tensor(label_data), train_ratio=4, val_ratio=2,
                                                                test_ratio=4)
    else:
        idx_train, idx_val, idx_test, c_num_mat = split_balance(torch.tensor(label_data), train_ratio=7, val_ratio=1,
                                                                test_ratio=2)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    labels = torch.LongTensor(np.where(label)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def encode_onehot(labels):

    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum+1e-6, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

