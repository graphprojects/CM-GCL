#!/usr/bin/env python
# encoding: utf-8

import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer,BertTokenizer
from utils.getdata import NodeTextDataset,load_data_for_pretrain
from scripts.CMCL import NodeTextCLModel
from utils.util import AvgMeter, get_lr
from utils.params import args
import torch.nn.utils.prune as prune
from utils.util import seed_torch
import finetune
import warnings
warnings.filterwarnings("ignore")
_BACKEND = 'pytorch'

seed_torch(4)

def make_train_valid_dfs():
    dataframe = pd.read_csv(args.node_entity_matching_path, sep='\t')
    max_id = dataframe.shape[0] if not args.debug else args.number_samples
    text_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        text_ids, size=int(0.2 * len(text_ids)), replace=False
    )

    train_ids = [id_ for id_ in text_ids if id_ not in valid_ids]
    dataframe['id'] = list(dataframe.index)
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    dataset = NodeTextDataset(
        dataframe["entity_id"].values,
        dataframe["text"].values,
        dataframe['label'].values,
        tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, feature,adj,train_loader, optimizer, lr_scheduler, step,pos):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(args.device) for k, v in batch.items() if k != "text"}
        loss, node_embed_prune,node_embeds = model(batch, feature, adj, pos)
        optimizer.zero_grad()
        if args.prune:
            for name, module in model.named_modules():
                if 'text_encoder.model' in name:
                    if args.text_encoder_model == 'distilbert-base-uncased':
                        if '.attention.self.q_lin' in name or '.attention.self.k_lin' in name or '.attention.self.v_lin' in name or '.attention.out_lin' in name or '.ffn.lin' in name:
                            module.weight = module.weight_orig.clone()
                            module.bias = module.bias_orig.clone()
                    elif args.text_encoder_model == 'bert-base-uncased':
                        if '.attention.self.query' in name or '.attention.self.key' in name or '.attention.self.value' in name or '.attention.output.dense' in name or '.intermediate.dense' in name or '.intermediate.dense' in name or '.attention.output.dense' in name or 'model.pooler.dense' in name:
                            module.weight = module.weight_orig.clone()
                            module.bias = module.bias_orig.clone()

                elif 'node_encoder_prune.' in name:
                    module.weight = module.weight_orig.clone()
                    module.bias = module.bias_orig.clone()

        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["input_ids"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter,node_embed_prune,node_embeds


def valid_epoch(model, feature,adj,valid_loader,pos):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(args.device) for k, v in batch.items() if k != "text"}
        loss, node_embed_prune,node_embeds = model(batch, feature, adj,pos)

        count = batch["input_ids"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter,node_embed_prune,node_embeds


def main():
    my_time = time.strftime('%Y%m%d%H%M', time.gmtime(time.time()))

    train_df, valid_df = make_train_valid_dfs()   # make train, validation datasets
    if args.text_encoder_model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained(args.text_encoder_tokenizer) # read distil-bert language model
    elif args.text_encoder_model == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder_tokenizer)  # read bert language model
    train_loader = build_loaders(train_df, tokenizer, mode="train")     #
    valid_loader = build_loaders(valid_df, tokenizer,mode="valid")

    adj, features, labels, pos = load_data_for_pretrain()
    model = NodeTextCLModel(features).to(args.device)

    if args.prune:
        for name, module in model.named_modules():
            if 'text_encoder.model' in name:
                if args.text_encoder_model == 'distilbert-base-uncased':
                    if '.attention.self.q_lin' in name or '.attention.self.k_lin' in name or '.attention.self.v_lin' in name or '.attention.out_lin' in name or '.ffn.lin' in name:
                        prune.l1_unstructured(module, name='weight', amount=int(module.weight.shape[0]*module.weight.shape[1]*args.prune_ratio))
                        prune.l1_unstructured(module, name='bias', amount=int(module.bias.shape[0]  * args.prune_ratio))
                elif args.text_encoder_model == 'bert-base-uncased':
                    if '.attention.self.query' in name or '.attention.self.key' in name or '.attention.self.value' in name or '.attention.output.dense' in name or '.intermediate.dense' in name or '.intermediate.dense' in name or '.attention.output.dense' in name or 'model.pooler.dense' in name:
                        prune.l1_unstructured(module, name='weight', amount=int(module.weight.shape[0] * module.weight.shape[1] * args.prune_ratio))
                        prune.l1_unstructured(module, name='bias', amount=int(module.bias.shape[0] * args.prune_ratio))

            elif 'node_encoder_prune.' in name:
                prune.l1_unstructured(module, name='weight', amount=int(module.weight.shape[0]*module.weight.shape[1]*args.prune_ratio))
                prune.l1_unstructured(module, name='bias',amount=int(module.bias.shape[0] * args.prune_ratio))

    optimizer = torch.optim.AdamW(
        model.parameters(), weight_decay=0
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.patience, factor=args.factor
    )
    step = "epoch"
    best_loss = float('inf')

    for epoch in range(args.pretrain_epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss,node_embed_prune_train,node_embeds_train = train_epoch(model,features,adj, train_loader, optimizer, lr_scheduler, step,pos)
        model.eval()

        save_model_path = "./pretrain/{}_node_text_{}.pt".format(args.dataset, my_time)

        if epoch % 5 == 0:
            with torch.no_grad():
                valid_loss,node_embed_prune_val,node_embeds_val = valid_epoch(model,features,adj, valid_loader, pos)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), save_model_path)
                print("Saved Best Model!")

    # model.eval()
    with torch.no_grad():

        df = pd.read_csv(args.id_content_path)
        finetune_feature_path = r'./finetune/{}_data/finetune_feature_{}.txt'.format(args.dataset,my_time)
        fo = open(finetune_feature_path, 'w', encoding='utf8')
        for i in range(df.shape[0]):
            embed_ = model.text_encoder.get_finetune_embed(df.loc[i, 'content'])
            np.savetxt(fo, np.array(embed_.detach().cpu()).reshape([1, 768]))
            if i % 1000 == 0:
                print("{} fine-tuned features have been generated!".format(i))

    save_embed_path = r'./finetune/{}_data/node_embed_{}.txt'.format(args.dataset,my_time)
    np.savetxt(save_embed_path, node_embeds_train.cpu().data.numpy())

    print("The fine-tuned features are save in {}!".format(finetune_feature_path))
    print("The pre-trained encoders are save in {}!".format(save_model_path))

    if args.finetune:
        finetune.fine_tune(save_model_path,finetune_feature_path)





if __name__ == "__main__":
    main()

