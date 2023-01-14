# CM-GCL
Co-Modality Graph Contrastive Learning for Imbalanced Node Classification
====
Official source code of "Co-Modality Graph Contrastive Learning for Imbalanced Node Classification
" (NeurIPS 2022 https://openreview.net/pdf?id=f_kvHrM4Q0).

## Requirements

This code is developed and tested with python 3.8.10. and the required packages are listed in the `requirements.txt`. 

Please run `pip install -r requirements.txt` to install all the dependencies. 

## Usage
### Data Download
As the data size is too large, please click on the link to download the [AMiner Data](https://drive.google.com/file/d/16X9y8iBL62j88YdhDaZQLOheuYlxFWiD/view?usp=sharing).
Once download the data, please unzip the file and further put the unzipped data into the aminer_data folder.

### Model Pre-training
For co-modality pre-training, we consider two types of modality combinations, i.e., graph modality and text modality, and 
graph modality and image modality in our paper:

```main_graph_text_gcl.py``` contains the code of graph contrastive learning over real-world datasets including raw text information and graph data.

```main_graph_image_gcl.py``` contains the code of graph contrastive learning over real-world datasets including raw images and graph data.

### Model Fine-tuning

```finetune.py``` contains the code of model finetuning for downstream tasks. 

The default setting for AMiner data is all set. If you want to train the model, please run the code ```python main_graph_text_gcl.py``` over AMiner data after installing all required packages.
It may take a while, you can also simply run the code ```python finetune.py``` to reproduce our results. We also provide a sample running log for the code above.


## Dataset

AMiner data is a paper-citation academic network containing the raw text content. The dataset of this paper is already provided to train our model. Please feel free to refer to the website for more details https://www.aminer.cn/aminer_data.

Yelp data is a review social graph containg the raw text content. Please refer to the website for more details http://odds.cs.stonybrook.edu/yelpchi-dataset/. If you need the raw text data for your research purpose, please email to the author listed on the website.

GitHub data is a graph for detecting malicious repository on social coding platform. Since our model uses raw text data for model training, it may cause privacy issues if we public the data. 

Instagram data is a social graph including raw text and image data for detecting drug traffickers on social platform. As our model needs raw image and text data for model training, it may cause privacy issues. 
If you want to implement our code (the combination of graph modality and image modality) on your datasets, please follow the format we provided in the instagram_data folder.



## Contact
Yiyue Qian - yqian5@nd.edu or yxq250@case.edu

Discussions, suggestions and questions are always welcome!



## Citation
If our work helps you, please cite our paper:

```
@inproceedings{qianco,
  title={Co-Modality Graph Contrastive Learning for Imbalanced Node Classification},
  author={Qian, Yiyue and Zhang, Chunhui and Zhang, Yiming and Wen, Qianlong and Ye, Yanfang and Zhang, Chuxu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Logger
This is a sample running logger which records the output and the model performance for AMiner data (also provide a output.log in aminer_data folder):
'''
python main_graph_text_gcl.py

Epoch: 1
100%|██████████| 145/145 [00:23<00:00,  6.19it/s, lr=0.001, train_loss=41.7]
100%|██████████| 37/37 [00:07<00:00,  5.08it/s, valid_loss=10.6]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 2
100%|██████████| 145/145 [00:22<00:00,  6.36it/s, lr=0.001, train_loss=12.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 3
100%|██████████| 145/145 [00:22<00:00,  6.51it/s, lr=0.001, train_loss=9.57]
Epoch: 4
100%|██████████| 145/145 [00:22<00:00,  6.48it/s, lr=0.001, train_loss=9.43]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 5
100%|██████████| 145/145 [00:22<00:00,  6.48it/s, lr=0.001, train_loss=9.35]
Epoch: 6
100%|██████████| 145/145 [00:29<00:00,  4.85it/s, lr=0.001, train_loss=9.29]
100%|██████████| 37/37 [00:07<00:00,  4.76it/s, valid_loss=9.05]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 7
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=9.19]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 8
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=9.04]
Epoch: 9
100%|██████████| 145/145 [00:25<00:00,  5.59it/s, lr=0.001, train_loss=9.02]
Epoch: 10
100%|██████████| 145/145 [00:29<00:00,  4.92it/s, lr=0.001, train_loss=8.71]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 11
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=8.62]
100%|██████████| 37/37 [00:07<00:00,  4.92it/s, valid_loss=8.81]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 12
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=8.27]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 13
100%|██████████| 145/145 [00:24<00:00,  5.87it/s, lr=0.001, train_loss=7.85]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 14
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=7.65]
Epoch: 15
100%|██████████| 145/145 [00:25<00:00,  5.73it/s, lr=0.001, train_loss=7.33]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 16
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=7.05]
100%|██████████| 37/37 [00:07<00:00,  4.89it/s, valid_loss=7.21]
Saved Best Model!
Epoch: 17
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=6.89]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 18
100%|██████████| 145/145 [00:24<00:00,  5.91it/s, lr=0.001, train_loss=6.71]
Epoch: 19
100%|██████████| 145/145 [00:24<00:00,  5.85it/s, lr=0.001, train_loss=6.62]
Epoch: 20
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=6.49]
Epoch: 21
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=6.45]
100%|██████████| 37/37 [00:07<00:00,  5.01it/s, valid_loss=7.04]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 22
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=6.42]
Epoch: 23
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=6.34]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 24
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=6.28]
Epoch: 25
100%|██████████| 145/145 [00:24<00:00,  5.82it/s, lr=0.001, train_loss=6.29]
Epoch: 26
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=6.19]
100%|██████████| 37/37 [00:07<00:00,  4.89it/s, valid_loss=7]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 27
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=6.09]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 28
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=6.06]
Epoch: 29
100%|██████████| 145/145 [00:25<00:00,  5.69it/s, lr=0.001, train_loss=5.83]
Epoch: 30
100%|██████████| 145/145 [00:25<00:00,  5.62it/s, lr=0.001, train_loss=5.59]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 31
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=5.4]
100%|██████████| 37/37 [00:07<00:00,  4.92it/s, valid_loss=6.69]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 32
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=5.25]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 33
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=5]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 34
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=4.86]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 35
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=4.66]
Epoch: 36
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=4.18]
100%|██████████| 37/37 [00:07<00:00,  4.86it/s, valid_loss=6.16]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 37
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=3.99]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 38
100%|██████████| 145/145 [00:24<00:00,  5.94it/s, lr=0.001, train_loss=3.69]
Epoch: 39
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=3.53]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 40
100%|██████████| 145/145 [00:25<00:00,  5.80it/s, lr=0.001, train_loss=3.42]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 41
100%|██████████| 145/145 [00:25<00:00,  5.72it/s, lr=0.001, train_loss=3.35]
100%|██████████| 37/37 [00:07<00:00,  4.91it/s, valid_loss=4.58]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 42
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=3.3]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 43
100%|██████████| 145/145 [00:25<00:00,  5.72it/s, lr=0.001, train_loss=3.24]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 44
100%|██████████| 145/145 [00:25<00:00,  5.69it/s, lr=0.001, train_loss=3.16]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 45
100%|██████████| 145/145 [00:25<00:00,  5.73it/s, lr=0.001, train_loss=3.17]
Epoch: 46
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=3.15]
100%|██████████| 37/37 [00:08<00:00,  4.12it/s, valid_loss=4.48]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 47
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=3.09]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 48
100%|██████████| 145/145 [00:25<00:00,  5.67it/s, lr=0.001, train_loss=3.06]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 49
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=3.01]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 50
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.99]
Epoch: 51
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.99]
100%|██████████| 37/37 [00:07<00:00,  4.80it/s, valid_loss=4.06]
Saved Best Model!
Epoch: 52
100%|██████████| 145/145 [00:25<00:00,  5.67it/s, lr=0.001, train_loss=2.94]
Epoch: 53
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.88]
Epoch: 54
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=2.89]
Epoch: 55
100%|██████████| 145/145 [00:25<00:00,  5.76it/s, lr=0.001, train_loss=2.9]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 56
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.86]
100%|██████████| 37/37 [00:07<00:00,  4.86it/s, valid_loss=3.52]
Saved Best Model!
Epoch: 57
100%|██████████| 145/145 [00:24<00:00,  5.90it/s, lr=0.001, train_loss=2.85]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 58
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.83]
Epoch: 59
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.78]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 60
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.76]
Epoch: 61
100%|██████████| 145/145 [00:25<00:00,  5.65it/s, lr=0.001, train_loss=2.83]
100%|██████████| 37/37 [00:07<00:00,  4.84it/s, valid_loss=3.69]
Epoch: 62
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.74]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 63
100%|██████████| 145/145 [00:24<00:00,  5.80it/s, lr=0.001, train_loss=2.74]
Epoch: 64
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.75]
Epoch: 65
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.73]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 66
100%|██████████| 145/145 [00:24<00:00,  5.91it/s, lr=0.001, train_loss=2.68]
100%|██████████| 37/37 [00:07<00:00,  4.91it/s, valid_loss=3.34]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 67
100%|██████████| 145/145 [00:24<00:00,  5.80it/s, lr=0.001, train_loss=2.69]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 68
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=2.68]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 69
100%|██████████| 145/145 [00:25<00:00,  5.69it/s, lr=0.001, train_loss=2.67]
Epoch: 70
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.65]
Epoch: 71
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.67]
100%|██████████| 37/37 [00:07<00:00,  5.03it/s, valid_loss=3.03]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 72
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.7]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 73
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=2.65]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 74
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=2.63]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 75
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.64]
Epoch: 76
100%|██████████| 145/145 [00:24<00:00,  5.82it/s, lr=0.001, train_loss=2.63]
100%|██████████| 37/37 [00:07<00:00,  4.94it/s, valid_loss=3.11]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 77
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=2.62]
Epoch: 78
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=2.61]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 79
100%|██████████| 145/145 [00:24<00:00,  5.85it/s, lr=0.001, train_loss=2.62]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 80
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=2.6]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 81
100%|██████████| 145/145 [00:24<00:00,  5.88it/s, lr=0.001, train_loss=2.61]
100%|██████████| 37/37 [00:07<00:00,  4.95it/s, valid_loss=3.42]
Epoch: 82
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=2.6]
Epoch: 83
100%|██████████| 145/145 [00:24<00:00,  5.85it/s, lr=0.001, train_loss=2.58]
Epoch: 84
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=2.59]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 85
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.57]
Epoch: 86
100%|██████████| 145/145 [00:25<00:00,  5.68it/s, lr=0.001, train_loss=2.57]
100%|██████████| 37/37 [00:07<00:00,  4.81it/s, valid_loss=3.05]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 87
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.56]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 88
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.58]
Epoch: 89
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.54]
Epoch: 90
100%|██████████| 145/145 [00:24<00:00,  5.89it/s, lr=0.001, train_loss=2.59]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 91
100%|██████████| 145/145 [00:24<00:00,  5.91it/s, lr=0.001, train_loss=2.58]
100%|██████████| 37/37 [00:07<00:00,  5.03it/s, valid_loss=2.67]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 92
100%|██████████| 145/145 [00:24<00:00,  5.88it/s, lr=0.001, train_loss=2.58]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 93
100%|██████████| 145/145 [00:24<00:00,  5.82it/s, lr=0.001, train_loss=2.57]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 94
100%|██████████| 145/145 [00:24<00:00,  5.90it/s, lr=0.001, train_loss=2.57]
Epoch: 95
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.54]
Epoch: 96
100%|██████████| 145/145 [00:24<00:00,  5.90it/s, lr=0.001, train_loss=2.56]
100%|██████████| 37/37 [00:07<00:00,  5.01it/s, valid_loss=2.68]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 97
100%|██████████| 145/145 [00:24<00:00,  5.88it/s, lr=0.001, train_loss=2.53]
Epoch: 98
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=2.54]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 99
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.51]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 100
100%|██████████| 145/145 [00:24<00:00,  5.91it/s, lr=0.001, train_loss=2.52]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 101
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.51]
100%|██████████| 37/37 [00:07<00:00,  4.78it/s, valid_loss=2.63]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 102
100%|██████████| 145/145 [00:25<00:00,  5.61it/s, lr=0.001, train_loss=2.5]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 103
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.51]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 104
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=2.49]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 105
100%|██████████| 145/145 [00:24<00:00,  5.88it/s, lr=0.001, train_loss=2.51]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 106
100%|██████████| 145/145 [00:24<00:00,  5.93it/s, lr=0.001, train_loss=2.5]
100%|██████████| 37/37 [00:07<00:00,  4.88it/s, valid_loss=2.6]
Saved Best Model!
Epoch: 107
100%|██████████| 145/145 [00:24<00:00,  5.91it/s, lr=0.001, train_loss=2.48]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 108
100%|██████████| 145/145 [00:24<00:00,  5.87it/s, lr=0.001, train_loss=2.48]
Epoch: 109
100%|██████████| 145/145 [00:24<00:00,  5.92it/s, lr=0.001, train_loss=2.49]
Epoch: 110
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.5]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 111
100%|██████████| 145/145 [00:24<00:00,  5.87it/s, lr=0.001, train_loss=2.48]
100%|██████████| 37/37 [00:07<00:00,  4.96it/s, valid_loss=2.56]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 112
100%|██████████| 145/145 [00:24<00:00,  5.87it/s, lr=0.001, train_loss=2.48]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 113
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.48]
Epoch: 114
100%|██████████| 145/145 [00:25<00:00,  5.65it/s, lr=0.001, train_loss=2.49]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 115
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=2.47]
Epoch: 116
100%|██████████| 145/145 [00:25<00:00,  5.66it/s, lr=0.001, train_loss=2.47]
100%|██████████| 37/37 [00:07<00:00,  4.69it/s, valid_loss=2.57]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 117
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=2.47]
Epoch: 118
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.47]
Epoch: 119
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=2.47]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 120
100%|██████████| 145/145 [00:25<00:00,  5.74it/s, lr=0.001, train_loss=2.47]
Epoch: 121
100%|██████████| 145/145 [00:25<00:00,  5.72it/s, lr=0.001, train_loss=2.47]
100%|██████████| 37/37 [00:07<00:00,  4.88it/s, valid_loss=2.55]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 122
100%|██████████| 145/145 [00:25<00:00,  5.80it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 123
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=2.46]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 124
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.44]
Epoch: 125
100%|██████████| 145/145 [00:25<00:00,  5.73it/s, lr=0.001, train_loss=2.46]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 126
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.45]
100%|██████████| 37/37 [00:07<00:00,  4.92it/s, valid_loss=2.51]
Saved Best Model!
Epoch: 127
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 128
100%|██████████| 145/145 [00:24<00:00,  5.83it/s, lr=0.001, train_loss=2.44]
Epoch: 129
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 130
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=2.44]
Epoch: 131
100%|██████████| 145/145 [00:24<00:00,  5.89it/s, lr=0.001, train_loss=2.44]
100%|██████████| 37/37 [00:07<00:00,  4.87it/s, valid_loss=2.5]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 132
100%|██████████| 145/145 [00:24<00:00,  5.82it/s, lr=0.001, train_loss=2.43]
Epoch: 133
100%|██████████| 145/145 [00:25<00:00,  5.80it/s, lr=0.001, train_loss=2.47]
Epoch: 134
100%|██████████| 145/145 [00:25<00:00,  5.79it/s, lr=0.001, train_loss=2.46]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 135
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 136
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.45]
100%|██████████| 37/37 [00:07<00:00,  4.85it/s, valid_loss=2.52]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 137
100%|██████████| 145/145 [00:24<00:00,  5.82it/s, lr=0.001, train_loss=2.46]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 138
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=2.42]
Epoch: 139
100%|██████████| 145/145 [00:24<00:00,  5.84it/s, lr=0.001, train_loss=2.43]
Epoch: 140
100%|██████████| 145/145 [00:25<00:00,  5.75it/s, lr=0.001, train_loss=2.43]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 141
100%|██████████| 145/145 [00:24<00:00,  5.86it/s, lr=0.001, train_loss=2.42]
100%|██████████| 37/37 [00:07<00:00,  4.96it/s, valid_loss=2.47]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 142
100%|██████████| 145/145 [00:35<00:00,  4.08it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 143
100%|██████████| 145/145 [00:31<00:00,  4.67it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 144
100%|██████████| 145/145 [00:26<00:00,  5.41it/s, lr=0.001, train_loss=2.44]
Epoch: 145
100%|██████████| 145/145 [00:24<00:00,  5.92it/s, lr=0.001, train_loss=2.43]
Epoch: 146
100%|██████████| 145/145 [00:26<00:00,  5.55it/s, lr=0.001, train_loss=2.44]
100%|██████████| 37/37 [00:08<00:00,  4.23it/s, valid_loss=2.49]
Epoch: 147
100%|██████████| 145/145 [00:25<00:00,  5.66it/s, lr=0.001, train_loss=2.43]
Epoch: 148
100%|██████████| 145/145 [00:25<00:00,  5.68it/s, lr=0.001, train_loss=2.42]
Epoch: 149
100%|██████████| 145/145 [00:24<00:00,  5.81it/s, lr=0.001, train_loss=2.42]
Epoch: 150
100%|██████████| 145/145 [00:25<00:00,  5.78it/s, lr=0.001, train_loss=2.44]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 151
100%|██████████| 145/145 [00:25<00:00,  5.73it/s, lr=0.001, train_loss=2.43]
100%|██████████| 37/37 [00:07<00:00,  4.94it/s, valid_loss=2.49]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 152
100%|██████████| 145/145 [00:34<00:00,  4.16it/s, lr=0.001, train_loss=2.41]
Epoch: 153
100%|██████████| 145/145 [00:30<00:00,  4.73it/s, lr=0.001, train_loss=2.42]
Epoch: 154
100%|██████████| 145/145 [00:34<00:00,  4.20it/s, lr=0.001, train_loss=2.41]
Epoch: 155
100%|██████████| 145/145 [00:29<00:00,  4.99it/s, lr=0.001, train_loss=2.42]
Epoch: 156
100%|██████████| 145/145 [00:29<00:00,  4.89it/s, lr=0.001, train_loss=2.42]
100%|██████████| 37/37 [00:08<00:00,  4.21it/s, valid_loss=2.5]
Epoch: 157
100%|██████████| 145/145 [00:34<00:00,  4.26it/s, lr=0.001, train_loss=2.42]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 158
100%|██████████| 145/145 [00:34<00:00,  4.22it/s, lr=0.001, train_loss=2.43]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 159
100%|██████████| 145/145 [00:33<00:00,  4.28it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 160
100%|██████████| 145/145 [00:25<00:00,  5.67it/s, lr=0.001, train_loss=2.4]
Epoch: 161
100%|██████████| 145/145 [00:25<00:00,  5.70it/s, lr=0.001, train_loss=2.43]
100%|██████████| 37/37 [00:07<00:00,  4.89it/s, valid_loss=2.48]
Epoch: 162
100%|██████████| 145/145 [00:25<00:00,  5.64it/s, lr=0.001, train_loss=2.42]
Epoch: 163
100%|██████████| 145/145 [00:25<00:00,  5.72it/s, lr=0.001, train_loss=2.44]
Epoch: 164
100%|██████████| 145/145 [00:25<00:00,  5.77it/s, lr=0.001, train_loss=2.4]
Epoch: 165
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.42]
Epoch: 166
100%|██████████| 145/145 [00:25<00:00,  5.63it/s, lr=0.001, train_loss=2.42]
100%|██████████| 37/37 [00:07<00:00,  4.90it/s, valid_loss=2.47]
  0%|          | 0/145 [00:00<?, ?it/s]Saved Best Model!
Epoch: 167
100%|██████████| 145/145 [00:25<00:00,  5.58it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 168
100%|██████████| 145/145 [00:26<00:00,  5.57it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 169
100%|██████████| 145/145 [00:26<00:00,  5.57it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 170
100%|██████████| 145/145 [00:26<00:00,  5.57it/s, lr=0.001, train_loss=2.42]
Epoch: 171
100%|██████████| 145/145 [00:25<00:00,  5.62it/s, lr=0.001, train_loss=2.41]
100%|██████████| 37/37 [00:07<00:00,  4.85it/s, valid_loss=2.48]
Epoch: 172
100%|██████████| 145/145 [00:25<00:00,  5.71it/s, lr=0.001, train_loss=2.41]
Epoch: 173
100%|██████████| 145/145 [00:25<00:00,  5.65it/s, lr=0.001, train_loss=2.4]
Epoch: 174
100%|██████████| 145/145 [00:25<00:00,  5.67it/s, lr=0.001, train_loss=2.41]
Epoch: 175
100%|██████████| 145/145 [00:27<00:00,  5.28it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 176
100%|██████████| 145/145 [00:38<00:00,  3.81it/s, lr=0.001, train_loss=2.41]
100%|██████████| 37/37 [00:09<00:00,  4.11it/s, valid_loss=2.49]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 177
100%|██████████| 145/145 [00:29<00:00,  4.98it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 178
100%|██████████| 145/145 [00:28<00:00,  5.04it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 179
100%|██████████| 145/145 [00:27<00:00,  5.32it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 180
100%|██████████| 145/145 [00:27<00:00,  5.25it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 181
100%|██████████| 145/145 [00:27<00:00,  5.21it/s, lr=0.001, train_loss=2.39]
100%|██████████| 37/37 [00:08<00:00,  4.61it/s, valid_loss=2.49]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 182
100%|██████████| 145/145 [00:27<00:00,  5.28it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 183
100%|██████████| 145/145 [00:27<00:00,  5.32it/s, lr=0.001, train_loss=2.4]
Epoch: 184
100%|██████████| 145/145 [00:27<00:00,  5.34it/s, lr=0.001, train_loss=2.4]
Epoch: 185
100%|██████████| 145/145 [00:26<00:00,  5.40it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 186
100%|██████████| 145/145 [00:26<00:00,  5.49it/s, lr=0.001, train_loss=2.39]
100%|██████████| 37/37 [00:07<00:00,  4.83it/s, valid_loss=2.44]
Saved Best Model!
Epoch: 187
100%|██████████| 145/145 [00:26<00:00,  5.48it/s, lr=0.001, train_loss=2.39]
Epoch: 188
100%|██████████| 145/145 [00:25<00:00,  5.59it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 189
100%|██████████| 145/145 [00:26<00:00,  5.53it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 190
100%|██████████| 145/145 [00:26<00:00,  5.54it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 191
100%|██████████| 145/145 [00:26<00:00,  5.51it/s, lr=0.001, train_loss=2.4]
100%|██████████| 37/37 [00:07<00:00,  4.85it/s, valid_loss=2.45]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 192
100%|██████████| 145/145 [00:26<00:00,  5.56it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 193
100%|██████████| 145/145 [00:25<00:00,  5.61it/s, lr=0.001, train_loss=2.4]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 194
100%|██████████| 145/145 [00:25<00:00,  5.63it/s, lr=0.001, train_loss=2.42]
Epoch: 195
100%|██████████| 145/145 [00:26<00:00,  5.53it/s, lr=0.001, train_loss=2.4]
Epoch: 196
100%|██████████| 145/145 [00:25<00:00,  5.59it/s, lr=0.001, train_loss=2.41]
100%|██████████| 37/37 [00:07<00:00,  4.74it/s, valid_loss=2.46]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 197
100%|██████████| 145/145 [00:26<00:00,  5.56it/s, lr=0.001, train_loss=2.4]
Epoch: 198
100%|██████████| 145/145 [00:26<00:00,  5.57it/s, lr=0.001, train_loss=2.4]
Epoch: 199
100%|██████████| 145/145 [00:26<00:00,  5.48it/s, lr=0.001, train_loss=2.41]
  0%|          | 0/145 [00:00<?, ?it/s]Epoch: 200
100%|██████████| 145/145 [00:26<00:00,  5.47it/s, lr=0.001, train_loss=2.39]
0 fine-tuned features have been generated!
1000 fine-tuned features have been generated!
2000 fine-tuned features have been generated!
3000 fine-tuned features have been generated!
4000 fine-tuned features have been generated!
5000 fine-tuned features have been generated!
6000 fine-tuned features have been generated!
7000 fine-tuned features have been generated!
8000 fine-tuned features have been generated!
9000 fine-tuned features have been generated!
10000 fine-tuned features have been generated!
11000 fine-tuned features have been generated!
12000 fine-tuned features have been generated!
13000 fine-tuned features have been generated!
14000 fine-tuned features have been generated!
15000 fine-tuned features have been generated!
16000 fine-tuned features have been generated!
17000 fine-tuned features have been generated!
18000 fine-tuned features have been generated!
19000 fine-tuned features have been generated!
20000 fine-tuned features have been generated!
21000 fine-tuned features have been generated!
22000 fine-tuned features have been generated!
23000 fine-tuned features have been generated!
24000 fine-tuned features have been generated!
25000 fine-tuned features have been generated!
26000 fine-tuned features have been generated!
27000 fine-tuned features have been generated!
28000 fine-tuned features have been generated!
29000 fine-tuned features have been generated!
30000 fine-tuned features have been generated!
31000 fine-tuned features have been generated!
32000 fine-tuned features have been generated!
33000 fine-tuned features have been generated!
34000 fine-tuned features have been generated!
35000 fine-tuned features have been generated!
36000 fine-tuned features have been generated!
37000 fine-tuned features have been generated!
38000 fine-tuned features have been generated!
39000 fine-tuned features have been generated!
40000 fine-tuned features have been generated!
The fine-tuned features are save in ./finetune/aminer_data/finetune_feature_202301141240.txt!
The pre-trained encoders are save in ./pretrain/aminer_node_text_202301141240.pt!
The number of class 0: 2262
The number of class 1: 4919
The number of class 2: 3054
The number of class 3: 3810
The number of class 4: 4044
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertModel: ['vocab_transform.bias', 'vocab_projector.weight', 'vocab_layer_norm.weight', 'vocab_layer_norm.bias', 'vocab_transform.weight', 'vocab_projector.bias']
- This IS expected if you are initializing DistilBertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
 Saving model ...
Epoch: 0001 loss_train: 3.1521 f1_micro_train: 0.2235 f1_macro_train: 0.0733 auc_train: 0.4539 f1_micro_val: 0.1760 f1_macro_val: 0.1004 auc_val: 0.4673
 Saving model ...
Epoch: 0002 loss_train: 1.7816 f1_micro_train: 0.1624 f1_macro_train: 0.0930 auc_train: 0.4625 f1_micro_val: 0.2518 f1_macro_val: 0.1575 auc_val: 0.5712
Epoch: 0003 loss_train: 1.1987 f1_micro_train: 0.2457 f1_macro_train: 0.1528 auc_train: 0.5615 f1_micro_val: 0.2717 f1_macro_val: 0.0855 auc_val: 0.5705
Epoch: 0004 loss_train: 1.0930 f1_micro_train: 0.2720 f1_macro_train: 0.0855 auc_train: 0.5690 f1_micro_val: 0.2717 f1_macro_val: 0.0855 auc_val: 0.5953
Epoch: 0005 loss_train: 1.0468 f1_micro_train: 0.2720 f1_macro_train: 0.0855 auc_train: 0.5914 f1_micro_val: 0.2811 f1_macro_val: 0.1139 auc_val: 0.6332
Epoch: 0006 loss_train: 0.9701 f1_micro_train: 0.2781 f1_macro_train: 0.1071 auc_train: 0.6322 f1_micro_val: 0.2817 f1_macro_val: 0.1204 auc_val: 0.6716
 Saving model ...
Epoch: 0007 loss_train: 0.9249 f1_micro_train: 0.2833 f1_macro_train: 0.1162 auc_train: 0.6704 f1_micro_val: 0.3481 f1_macro_val: 0.2107 auc_val: 0.6829
Epoch: 0008 loss_train: 0.9266 f1_micro_train: 0.3534 f1_macro_train: 0.2153 auc_train: 0.6839 f1_micro_val: 0.2690 f1_macro_val: 0.1959 auc_val: 0.6988
Epoch: 0009 loss_train: 0.9373 f1_micro_train: 0.2660 f1_macro_train: 0.1953 auc_train: 0.6999 f1_micro_val: 0.2789 f1_macro_val: 0.1942 auc_val: 0.7117
 Saving model ...
Epoch: 0010 loss_train: 0.9409 f1_micro_train: 0.2824 f1_macro_train: 0.2038 auc_train: 0.7115 f1_micro_val: 0.3547 f1_macro_val: 0.2879 auc_val: 0.7243
 Saving model ...
Epoch: 0011 loss_train: 0.9338 f1_micro_train: 0.3530 f1_macro_train: 0.2843 auc_train: 0.7243 f1_micro_val: 0.4228 f1_macro_val: 0.3092 auc_val: 0.7344
 Saving model ...
Epoch: 0012 loss_train: 0.9182 f1_micro_train: 0.4236 f1_macro_train: 0.3094 auc_train: 0.7362 f1_micro_val: 0.4604 f1_macro_val: 0.3359 auc_val: 0.7400
Epoch: 0013 loss_train: 0.9004 f1_micro_train: 0.4656 f1_macro_train: 0.3404 auc_train: 0.7428 f1_micro_val: 0.3951 f1_macro_val: 0.2566 auc_val: 0.7424
Epoch: 0014 loss_train: 0.8848 f1_micro_train: 0.3933 f1_macro_train: 0.2582 auc_train: 0.7458 f1_micro_val: 0.3780 f1_macro_val: 0.2492 auc_val: 0.7421
Epoch: 0015 loss_train: 0.8727 f1_micro_train: 0.3639 f1_macro_train: 0.2296 auc_train: 0.7457 f1_micro_val: 0.3713 f1_macro_val: 0.2458 auc_val: 0.7399
Epoch: 0016 loss_train: 0.8646 f1_micro_train: 0.3565 f1_macro_train: 0.2311 auc_train: 0.7431 f1_micro_val: 0.3641 f1_macro_val: 0.2406 auc_val: 0.7401
Epoch: 0017 loss_train: 0.8599 f1_micro_train: 0.3581 f1_macro_train: 0.2303 auc_train: 0.7421 f1_micro_val: 0.3730 f1_macro_val: 0.2539 auc_val: 0.7425
Epoch: 0018 loss_train: 0.8553 f1_micro_train: 0.3664 f1_macro_train: 0.2445 auc_train: 0.7445 f1_micro_val: 0.3968 f1_macro_val: 0.2808 auc_val: 0.7478
Epoch: 0019 loss_train: 0.8469 f1_micro_train: 0.3953 f1_macro_train: 0.2763 auc_train: 0.7501 f1_micro_val: 0.4289 f1_macro_val: 0.3078 auc_val: 0.7548
Epoch: 0020 loss_train: 0.8341 f1_micro_train: 0.4275 f1_macro_train: 0.3033 auc_train: 0.7582 f1_micro_val: 0.4344 f1_macro_val: 0.2976 auc_val: 0.7636
Epoch: 0021 loss_train: 0.8189 f1_micro_train: 0.4344 f1_macro_train: 0.2922 auc_train: 0.7680 f1_micro_val: 0.4261 f1_macro_val: 0.2782 auc_val: 0.7745
Epoch: 0022 loss_train: 0.8051 f1_micro_train: 0.4295 f1_macro_train: 0.2774 auc_train: 0.7782 f1_micro_val: 0.4449 f1_macro_val: 0.3175 auc_val: 0.7819
 Saving model ...
Epoch: 0023 loss_train: 0.7938 f1_micro_train: 0.4489 f1_macro_train: 0.3163 auc_train: 0.7870 f1_micro_val: 0.4920 f1_macro_val: 0.3828 auc_val: 0.7882
 Saving model ...
Epoch: 0024 loss_train: 0.7830 f1_micro_train: 0.4948 f1_macro_train: 0.3860 auc_train: 0.7947 f1_micro_val: 0.5340 f1_macro_val: 0.4351 auc_val: 0.7945
 Saving model ...
Epoch: 0025 loss_train: 0.7717 f1_micro_train: 0.5375 f1_macro_train: 0.4386 auc_train: 0.8016 f1_micro_val: 0.5573 f1_macro_val: 0.4668 auc_val: 0.8005
 Saving model ...
Epoch: 0026 loss_train: 0.7603 f1_micro_train: 0.5618 f1_macro_train: 0.4710 auc_train: 0.8080 f1_micro_val: 0.5623 f1_macro_val: 0.4773 auc_val: 0.8054
 Saving model ...
Epoch: 0027 loss_train: 0.7510 f1_micro_train: 0.5760 f1_macro_train: 0.4897 auc_train: 0.8131 f1_micro_val: 0.5667 f1_macro_val: 0.4825 auc_val: 0.8092
Epoch: 0028 loss_train: 0.7423 f1_micro_train: 0.5780 f1_macro_train: 0.4928 auc_train: 0.8174 f1_micro_val: 0.5656 f1_macro_val: 0.4798 auc_val: 0.8133
Epoch: 0029 loss_train: 0.7314 f1_micro_train: 0.5705 f1_macro_train: 0.4842 auc_train: 0.8215 f1_micro_val: 0.5578 f1_macro_val: 0.4654 auc_val: 0.8172
Epoch: 0030 loss_train: 0.7187 f1_micro_train: 0.5547 f1_macro_train: 0.4631 auc_train: 0.8257 f1_micro_val: 0.5423 f1_macro_val: 0.4493 auc_val: 0.8212
Epoch: 0031 loss_train: 0.7068 f1_micro_train: 0.5412 f1_macro_train: 0.4454 auc_train: 0.8298 f1_micro_val: 0.5462 f1_macro_val: 0.4535 auc_val: 0.8246
Epoch: 0032 loss_train: 0.6959 f1_micro_train: 0.5414 f1_macro_train: 0.4466 auc_train: 0.8331 f1_micro_val: 0.5628 f1_macro_val: 0.4762 auc_val: 0.8270
 Saving model ...
Epoch: 0033 loss_train: 0.6842 f1_micro_train: 0.5588 f1_macro_train: 0.4701 auc_train: 0.8353 f1_micro_val: 0.5849 f1_macro_val: 0.5032 auc_val: 0.8284
 Saving model ...
Epoch: 0034 loss_train: 0.6723 f1_micro_train: 0.5893 f1_macro_train: 0.5060 auc_train: 0.8367 f1_micro_val: 0.5905 f1_macro_val: 0.5117 auc_val: 0.8299
 Saving model ...
Epoch: 0035 loss_train: 0.6618 f1_micro_train: 0.6041 f1_macro_train: 0.5226 auc_train: 0.8380 f1_micro_val: 0.5971 f1_macro_val: 0.5202 auc_val: 0.8322
 Saving model ...
Epoch: 0036 loss_train: 0.6524 f1_micro_train: 0.6073 f1_macro_train: 0.5279 auc_train: 0.8401 f1_micro_val: 0.6104 f1_macro_val: 0.5318 auc_val: 0.8354
 Saving model ...
Epoch: 0037 loss_train: 0.6418 f1_micro_train: 0.6158 f1_macro_train: 0.5352 auc_train: 0.8432 f1_micro_val: 0.6182 f1_macro_val: 0.5372 auc_val: 0.8390
 Saving model ...
Epoch: 0038 loss_train: 0.6301 f1_micro_train: 0.6250 f1_macro_train: 0.5415 auc_train: 0.8468 f1_micro_val: 0.6204 f1_macro_val: 0.5376 auc_val: 0.8421
 Saving model ...
Epoch: 0039 loss_train: 0.6196 f1_micro_train: 0.6264 f1_macro_train: 0.5406 auc_train: 0.8499 f1_micro_val: 0.6237 f1_macro_val: 0.5382 auc_val: 0.8448
Epoch: 0040 loss_train: 0.6098 f1_micro_train: 0.6250 f1_macro_train: 0.5380 auc_train: 0.8525 f1_micro_val: 0.6220 f1_macro_val: 0.5362 auc_val: 0.8475
 Saving model ...
Epoch: 0041 loss_train: 0.5992 f1_micro_train: 0.6294 f1_macro_train: 0.5414 auc_train: 0.8549 f1_micro_val: 0.6348 f1_macro_val: 0.5486 auc_val: 0.8499
 Saving model ...
Epoch: 0042 loss_train: 0.5886 f1_micro_train: 0.6398 f1_macro_train: 0.5513 auc_train: 0.8572 f1_micro_val: 0.6425 f1_macro_val: 0.5562 auc_val: 0.8521
 Saving model ...
Epoch: 0043 loss_train: 0.5797 f1_micro_train: 0.6495 f1_macro_train: 0.5609 auc_train: 0.8593 f1_micro_val: 0.6464 f1_macro_val: 0.5596 auc_val: 0.8542
 Saving model ...
Epoch: 0044 loss_train: 0.5708 f1_micro_train: 0.6539 f1_macro_train: 0.5643 auc_train: 0.8613 f1_micro_val: 0.6530 f1_macro_val: 0.5645 auc_val: 0.8565
 Saving model ...
Epoch: 0045 loss_train: 0.5609 f1_micro_train: 0.6577 f1_macro_train: 0.5667 auc_train: 0.8635 f1_micro_val: 0.6597 f1_macro_val: 0.5692 auc_val: 0.8588
 Saving model ...
Epoch: 0046 loss_train: 0.5519 f1_micro_train: 0.6608 f1_macro_train: 0.5692 auc_train: 0.8660 f1_micro_val: 0.6652 f1_macro_val: 0.5740 auc_val: 0.8614
 Saving model ...
Epoch: 0047 loss_train: 0.5437 f1_micro_train: 0.6642 f1_macro_train: 0.5722 auc_train: 0.8687 f1_micro_val: 0.6663 f1_macro_val: 0.5746 auc_val: 0.8644
 Saving model ...
Epoch: 0048 loss_train: 0.5348 f1_micro_train: 0.6709 f1_macro_train: 0.5781 auc_train: 0.8717 f1_micro_val: 0.6685 f1_macro_val: 0.5773 auc_val: 0.8672
 Saving model ...
Epoch: 0049 loss_train: 0.5259 f1_micro_train: 0.6746 f1_macro_train: 0.5812 auc_train: 0.8747 f1_micro_val: 0.6696 f1_macro_val: 0.5780 auc_val: 0.8699
 Saving model ...
Epoch: 0050 loss_train: 0.5177 f1_micro_train: 0.6786 f1_macro_train: 0.5844 auc_train: 0.8774 f1_micro_val: 0.6735 f1_macro_val: 0.5813 auc_val: 0.8726
 Saving model ...
Epoch: 0051 loss_train: 0.5090 f1_micro_train: 0.6839 f1_macro_train: 0.5890 auc_train: 0.8802 f1_micro_val: 0.6818 f1_macro_val: 0.5882 auc_val: 0.8754
 Saving model ...
Epoch: 0052 loss_train: 0.5004 f1_micro_train: 0.6885 f1_macro_train: 0.5928 auc_train: 0.8828 f1_micro_val: 0.6857 f1_macro_val: 0.5942 auc_val: 0.8779
 Saving model ...
Epoch: 0053 loss_train: 0.4923 f1_micro_train: 0.6916 f1_macro_train: 0.5992 auc_train: 0.8853 f1_micro_val: 0.6951 f1_macro_val: 0.6102 auc_val: 0.8806
 Saving model ...
Epoch: 0054 loss_train: 0.4838 f1_micro_train: 0.6984 f1_macro_train: 0.6097 auc_train: 0.8879 f1_micro_val: 0.7034 f1_macro_val: 0.6165 auc_val: 0.8835
 Saving model ...
Epoch: 0055 loss_train: 0.4755 f1_micro_train: 0.7062 f1_macro_train: 0.6211 auc_train: 0.8907 f1_micro_val: 0.7078 f1_macro_val: 0.6216 auc_val: 0.8864
 Saving model ...
Epoch: 0056 loss_train: 0.4669 f1_micro_train: 0.7113 f1_macro_train: 0.6264 auc_train: 0.8934 f1_micro_val: 0.7122 f1_macro_val: 0.6243 auc_val: 0.8891
 Saving model ...
Epoch: 0057 loss_train: 0.4583 f1_micro_train: 0.7133 f1_macro_train: 0.6247 auc_train: 0.8960 f1_micro_val: 0.7161 f1_macro_val: 0.6278 auc_val: 0.8916
 Saving model ...
Epoch: 0058 loss_train: 0.4501 f1_micro_train: 0.7173 f1_macro_train: 0.6264 auc_train: 0.8985 f1_micro_val: 0.7205 f1_macro_val: 0.6310 auc_val: 0.8944
 Saving model ...
Epoch: 0059 loss_train: 0.4416 f1_micro_train: 0.7220 f1_macro_train: 0.6317 auc_train: 0.9012 f1_micro_val: 0.7266 f1_macro_val: 0.6374 auc_val: 0.8974
 Saving model ...
Epoch: 0060 loss_train: 0.4333 f1_micro_train: 0.7282 f1_macro_train: 0.6423 auc_train: 0.9039 f1_micro_val: 0.7294 f1_macro_val: 0.6475 auc_val: 0.9000
 Saving model ...
Epoch: 0061 loss_train: 0.4249 f1_micro_train: 0.7311 f1_macro_train: 0.6494 auc_train: 0.9063 f1_micro_val: 0.7366 f1_macro_val: 0.6613 auc_val: 0.9025
 Saving model ...
Epoch: 0062 loss_train: 0.4165 f1_micro_train: 0.7348 f1_macro_train: 0.6568 auc_train: 0.9087 f1_micro_val: 0.7405 f1_macro_val: 0.6679 auc_val: 0.9053
 Saving model ...
Epoch: 0063 loss_train: 0.4080 f1_micro_train: 0.7396 f1_macro_train: 0.6678 auc_train: 0.9114 f1_micro_val: 0.7427 f1_macro_val: 0.6727 auc_val: 0.9082
 Saving model ...
Epoch: 0064 loss_train: 0.3998 f1_micro_train: 0.7451 f1_macro_train: 0.6757 auc_train: 0.9141 f1_micro_val: 0.7488 f1_macro_val: 0.6776 auc_val: 0.9106
Epoch: 0065 loss_train: 0.3921 f1_micro_train: 0.7485 f1_macro_train: 0.6778 auc_train: 0.9164 f1_micro_val: 0.7493 f1_macro_val: 0.6751 auc_val: 0.9129
Epoch: 0066 loss_train: 0.3845 f1_micro_train: 0.7509 f1_macro_train: 0.6793 auc_train: 0.9187 f1_micro_val: 0.7521 f1_macro_val: 0.6776 auc_val: 0.9154
 Saving model ...
Epoch: 0067 loss_train: 0.3773 f1_micro_train: 0.7547 f1_macro_train: 0.6827 auc_train: 0.9210 f1_micro_val: 0.7571 f1_macro_val: 0.6865 auc_val: 0.9177
 Saving model ...
Epoch: 0068 loss_train: 0.3701 f1_micro_train: 0.7575 f1_macro_train: 0.6855 auc_train: 0.9231 f1_micro_val: 0.7587 f1_macro_val: 0.6887 auc_val: 0.9199
 Saving model ...
Epoch: 0069 loss_train: 0.3631 f1_micro_train: 0.7607 f1_macro_train: 0.6926 auc_train: 0.9253 f1_micro_val: 0.7598 f1_macro_val: 0.6955 auc_val: 0.9220
 Saving model ...
Epoch: 0070 loss_train: 0.3563 f1_micro_train: 0.7645 f1_macro_train: 0.7017 auc_train: 0.9273 f1_micro_val: 0.7626 f1_macro_val: 0.7039 auc_val: 0.9238
 Saving model ...
Epoch: 0071 loss_train: 0.3497 f1_micro_train: 0.7686 f1_macro_train: 0.7100 auc_train: 0.9290 f1_micro_val: 0.7659 f1_macro_val: 0.7091 auc_val: 0.9259
 Saving model ...
Epoch: 0072 loss_train: 0.3433 f1_micro_train: 0.7726 f1_macro_train: 0.7163 auc_train: 0.9310 f1_micro_val: 0.7709 f1_macro_val: 0.7147 auc_val: 0.9280
Epoch: 0073 loss_train: 0.3373 f1_micro_train: 0.7754 f1_macro_train: 0.7205 auc_train: 0.9330 f1_micro_val: 0.7709 f1_macro_val: 0.7132 auc_val: 0.9297
 Saving model ...
Epoch: 0074 loss_train: 0.3313 f1_micro_train: 0.7776 f1_macro_train: 0.7222 auc_train: 0.9346 f1_micro_val: 0.7720 f1_macro_val: 0.7151 auc_val: 0.9313
 Saving model ...
Epoch: 0075 loss_train: 0.3258 f1_micro_train: 0.7803 f1_macro_train: 0.7258 auc_train: 0.9361 f1_micro_val: 0.7764 f1_macro_val: 0.7225 auc_val: 0.9331
 Saving model ...
Epoch: 0076 loss_train: 0.3205 f1_micro_train: 0.7836 f1_macro_train: 0.7323 auc_train: 0.9378 f1_micro_val: 0.7820 f1_macro_val: 0.7311 auc_val: 0.9346
 Saving model ...
Epoch: 0077 loss_train: 0.3155 f1_micro_train: 0.7863 f1_macro_train: 0.7375 auc_train: 0.9392 f1_micro_val: 0.7825 f1_macro_val: 0.7332 auc_val: 0.9360
 Saving model ...
Epoch: 0078 loss_train: 0.3106 f1_micro_train: 0.7897 f1_macro_train: 0.7440 auc_train: 0.9406 f1_micro_val: 0.7847 f1_macro_val: 0.7385 auc_val: 0.9374
 Saving model ...
Epoch: 0079 loss_train: 0.3061 f1_micro_train: 0.7925 f1_macro_train: 0.7491 auc_train: 0.9419 f1_micro_val: 0.7858 f1_macro_val: 0.7396 auc_val: 0.9386
 Saving model ...
Epoch: 0080 loss_train: 0.3017 f1_micro_train: 0.7934 f1_macro_train: 0.7495 auc_train: 0.9432 f1_micro_val: 0.7892 f1_macro_val: 0.7432 auc_val: 0.9399
 Saving model ...
Epoch: 0081 loss_train: 0.2975 f1_micro_train: 0.7974 f1_macro_train: 0.7545 auc_train: 0.9444 f1_micro_val: 0.7925 f1_macro_val: 0.7493 auc_val: 0.9411
 Saving model ...
Epoch: 0082 loss_train: 0.2936 f1_micro_train: 0.7991 f1_macro_train: 0.7574 auc_train: 0.9456 f1_micro_val: 0.7914 f1_macro_val: 0.7495 auc_val: 0.9422
 Saving model ...
Epoch: 0083 loss_train: 0.2897 f1_micro_train: 0.8005 f1_macro_train: 0.7605 auc_train: 0.9466 f1_micro_val: 0.7969 f1_macro_val: 0.7581 auc_val: 0.9433
 Saving model ...
Epoch: 0084 loss_train: 0.2861 f1_micro_train: 0.8041 f1_macro_train: 0.7671 auc_train: 0.9478 f1_micro_val: 0.7969 f1_macro_val: 0.7592 auc_val: 0.9443
 Saving model ...
Epoch: 0085 loss_train: 0.2826 f1_micro_train: 0.8065 f1_macro_train: 0.7710 auc_train: 0.9487 f1_micro_val: 0.8008 f1_macro_val: 0.7649 auc_val: 0.9454
 Saving model ...
Epoch: 0086 loss_train: 0.2793 f1_micro_train: 0.8077 f1_macro_train: 0.7734 auc_train: 0.9497 f1_micro_val: 0.8019 f1_macro_val: 0.7653 auc_val: 0.9464
 Saving model ...
Epoch: 0087 loss_train: 0.2760 f1_micro_train: 0.8092 f1_macro_train: 0.7745 auc_train: 0.9506 f1_micro_val: 0.8030 f1_macro_val: 0.7659 auc_val: 0.9473
 Saving model ...
Epoch: 0088 loss_train: 0.2730 f1_micro_train: 0.8103 f1_macro_train: 0.7755 auc_train: 0.9516 f1_micro_val: 0.8058 f1_macro_val: 0.7715 auc_val: 0.9483
 Saving model ...
Epoch: 0089 loss_train: 0.2700 f1_micro_train: 0.8130 f1_macro_train: 0.7807 auc_train: 0.9524 f1_micro_val: 0.8080 f1_macro_val: 0.7746 auc_val: 0.9491
 Saving model ...
Epoch: 0090 loss_train: 0.2672 f1_micro_train: 0.8145 f1_macro_train: 0.7832 auc_train: 0.9532 f1_micro_val: 0.8118 f1_macro_val: 0.7791 auc_val: 0.9500
Epoch: 0091 loss_train: 0.2644 f1_micro_train: 0.8164 f1_macro_train: 0.7855 auc_train: 0.9541 f1_micro_val: 0.8113 f1_macro_val: 0.7777 auc_val: 0.9508
 Saving model ...
Epoch: 0092 loss_train: 0.2617 f1_micro_train: 0.8168 f1_macro_train: 0.7850 auc_train: 0.9548 f1_micro_val: 0.8135 f1_macro_val: 0.7804 auc_val: 0.9517
 Saving model ...
Epoch: 0093 loss_train: 0.2591 f1_micro_train: 0.8176 f1_macro_train: 0.7862 auc_train: 0.9556 f1_micro_val: 0.8157 f1_macro_val: 0.7828 auc_val: 0.9525
 Saving model ...
Epoch: 0094 loss_train: 0.2566 f1_micro_train: 0.8194 f1_macro_train: 0.7884 auc_train: 0.9563 f1_micro_val: 0.8196 f1_macro_val: 0.7889 auc_val: 0.9532
 Saving model ...
Epoch: 0095 loss_train: 0.2542 f1_micro_train: 0.8216 f1_macro_train: 0.7913 auc_train: 0.9570 f1_micro_val: 0.8196 f1_macro_val: 0.7893 auc_val: 0.9540
 Saving model ...
Epoch: 0096 loss_train: 0.2518 f1_micro_train: 0.8226 f1_macro_train: 0.7931 auc_train: 0.9577 f1_micro_val: 0.8213 f1_macro_val: 0.7912 auc_val: 0.9547
 Saving model ...
Epoch: 0097 loss_train: 0.2495 f1_micro_train: 0.8238 f1_macro_train: 0.7941 auc_train: 0.9583 f1_micro_val: 0.8229 f1_macro_val: 0.7927 auc_val: 0.9554
Epoch: 0098 loss_train: 0.2473 f1_micro_train: 0.8246 f1_macro_train: 0.7943 auc_train: 0.9589 f1_micro_val: 0.8224 f1_macro_val: 0.7923 auc_val: 0.9561
 Saving model ...
Epoch: 0099 loss_train: 0.2452 f1_micro_train: 0.8259 f1_macro_train: 0.7963 auc_train: 0.9595 f1_micro_val: 0.8246 f1_macro_val: 0.7958 auc_val: 0.9567
Epoch: 0100 loss_train: 0.2431 f1_micro_train: 0.8277 f1_macro_train: 0.7992 auc_train: 0.9601 f1_micro_val: 0.8246 f1_macro_val: 0.7958 auc_val: 0.9573
Epoch: 0101 loss_train: 0.2411 f1_micro_train: 0.8284 f1_macro_train: 0.7996 auc_train: 0.9607 f1_micro_val: 0.8251 f1_macro_val: 0.7958 auc_val: 0.9579
Epoch: 0102 loss_train: 0.2391 f1_micro_train: 0.8293 f1_macro_train: 0.8008 auc_train: 0.9612 f1_micro_val: 0.8240 f1_macro_val: 0.7955 auc_val: 0.9584
 Saving model ...
Epoch: 0103 loss_train: 0.2373 f1_micro_train: 0.8295 f1_macro_train: 0.8004 auc_train: 0.9617 f1_micro_val: 0.8273 f1_macro_val: 0.7998 auc_val: 0.9590
Epoch: 0104 loss_train: 0.2355 f1_micro_train: 0.8313 f1_macro_train: 0.8039 auc_train: 0.9622 f1_micro_val: 0.8262 f1_macro_val: 0.7979 auc_val: 0.9593
 Saving model ...
Epoch: 0105 loss_train: 0.2338 f1_micro_train: 0.8318 f1_macro_train: 0.8038 auc_train: 0.9627 f1_micro_val: 0.8284 f1_macro_val: 0.8013 auc_val: 0.9601
Epoch: 0106 loss_train: 0.2323 f1_micro_train: 0.8339 f1_macro_train: 0.8082 auc_train: 0.9631 f1_micro_val: 0.8279 f1_macro_val: 0.7998 auc_val: 0.9599
 Saving model ...
Epoch: 0107 loss_train: 0.2311 f1_micro_train: 0.8333 f1_macro_train: 0.8051 auc_train: 0.9636 f1_micro_val: 0.8318 f1_macro_val: 0.8062 auc_val: 0.9609
Epoch: 0108 loss_train: 0.2298 f1_micro_train: 0.8361 f1_macro_train: 0.8116 auc_train: 0.9639 f1_micro_val: 0.8312 f1_macro_val: 0.8053 auc_val: 0.9606
 Saving model ...
Epoch: 0109 loss_train: 0.2284 f1_micro_train: 0.8329 f1_macro_train: 0.8048 auc_train: 0.9644 f1_micro_val: 0.8323 f1_macro_val: 0.8075 auc_val: 0.9616
 Saving model ...
Epoch: 0110 loss_train: 0.2263 f1_micro_train: 0.8385 f1_macro_train: 0.8146 auc_train: 0.9648 f1_micro_val: 0.8340 f1_macro_val: 0.8092 auc_val: 0.9616
 Saving model ...
Epoch: 0111 loss_train: 0.2240 f1_micro_train: 0.8385 f1_macro_train: 0.8123 auc_train: 0.9653 f1_micro_val: 0.8345 f1_macro_val: 0.8093 auc_val: 0.9620
Epoch: 0112 loss_train: 0.2223 f1_micro_train: 0.8396 f1_macro_train: 0.8144 auc_train: 0.9657 f1_micro_val: 0.8323 f1_macro_val: 0.8077 auc_val: 0.9625
 Saving model ...
Epoch: 0113 loss_train: 0.2212 f1_micro_train: 0.8412 f1_macro_train: 0.8174 auc_train: 0.9660 f1_micro_val: 0.8373 f1_macro_val: 0.8138 auc_val: 0.9624
Epoch: 0114 loss_train: 0.2201 f1_micro_train: 0.8404 f1_macro_train: 0.8145 auc_train: 0.9665 f1_micro_val: 0.8367 f1_macro_val: 0.8132 auc_val: 0.9631
 Saving model ...
Epoch: 0115 loss_train: 0.2185 f1_micro_train: 0.8424 f1_macro_train: 0.8193 auc_train: 0.9667 f1_micro_val: 0.8390 f1_macro_val: 0.8151 auc_val: 0.9631
 Saving model ...
Epoch: 0116 loss_train: 0.2167 f1_micro_train: 0.8439 f1_macro_train: 0.8197 auc_train: 0.9672 f1_micro_val: 0.8412 f1_macro_val: 0.8176 auc_val: 0.9635
Epoch: 0117 loss_train: 0.2152 f1_micro_train: 0.8454 f1_macro_train: 0.8219 auc_train: 0.9676 f1_micro_val: 0.8401 f1_macro_val: 0.8171 auc_val: 0.9638
Epoch: 0118 loss_train: 0.2140 f1_micro_train: 0.8453 f1_macro_train: 0.8225 auc_train: 0.9678 f1_micro_val: 0.8406 f1_macro_val: 0.8175 auc_val: 0.9638
 Saving model ...
Epoch: 0119 loss_train: 0.2129 f1_micro_train: 0.8452 f1_macro_train: 0.8212 auc_train: 0.9682 f1_micro_val: 0.8412 f1_macro_val: 0.8184 auc_val: 0.9643
Epoch: 0120 loss_train: 0.2114 f1_micro_train: 0.8481 f1_macro_train: 0.8259 auc_train: 0.9685 f1_micro_val: 0.8401 f1_macro_val: 0.8163 auc_val: 0.9644
 Saving model ...
Epoch: 0121 loss_train: 0.2098 f1_micro_train: 0.8485 f1_macro_train: 0.8255 auc_train: 0.9689 f1_micro_val: 0.8423 f1_macro_val: 0.8202 auc_val: 0.9647
 Saving model ...
Epoch: 0122 loss_train: 0.2085 f1_micro_train: 0.8497 f1_macro_train: 0.8274 auc_train: 0.9692 f1_micro_val: 0.8428 f1_macro_val: 0.8215 auc_val: 0.9650
Epoch: 0123 loss_train: 0.2074 f1_micro_train: 0.8520 f1_macro_train: 0.8310 auc_train: 0.9695 f1_micro_val: 0.8390 f1_macro_val: 0.8163 auc_val: 0.9651
 Saving model ...
Epoch: 0124 loss_train: 0.2062 f1_micro_train: 0.8504 f1_macro_train: 0.8277 auc_train: 0.9698 f1_micro_val: 0.8467 f1_macro_val: 0.8258 auc_val: 0.9654
Epoch: 0125 loss_train: 0.2049 f1_micro_train: 0.8524 f1_macro_train: 0.8311 auc_train: 0.9701 f1_micro_val: 0.8434 f1_macro_val: 0.8221 auc_val: 0.9655
Epoch: 0126 loss_train: 0.2036 f1_micro_train: 0.8531 f1_macro_train: 0.8317 auc_train: 0.9704 f1_micro_val: 0.8423 f1_macro_val: 0.8206 auc_val: 0.9657
 Saving model ...
Epoch: 0127 loss_train: 0.2025 f1_micro_train: 0.8528 f1_macro_train: 0.8311 auc_train: 0.9707 f1_micro_val: 0.8484 f1_macro_val: 0.8284 auc_val: 0.9660
Epoch: 0128 loss_train: 0.2015 f1_micro_train: 0.8557 f1_macro_train: 0.8347 auc_train: 0.9709 f1_micro_val: 0.8434 f1_macro_val: 0.8211 auc_val: 0.9661
 Saving model ...
Epoch: 0129 loss_train: 0.2003 f1_micro_train: 0.8539 f1_macro_train: 0.8324 auc_train: 0.9713 f1_micro_val: 0.8489 f1_macro_val: 0.8293 auc_val: 0.9664
Epoch: 0130 loss_train: 0.1991 f1_micro_train: 0.8565 f1_macro_train: 0.8355 auc_train: 0.9715 f1_micro_val: 0.8484 f1_macro_val: 0.8279 auc_val: 0.9666
Epoch: 0131 loss_train: 0.1980 f1_micro_train: 0.8562 f1_macro_train: 0.8350 auc_train: 0.9718 f1_micro_val: 0.8467 f1_macro_val: 0.8253 auc_val: 0.9667
 Saving model ...
Epoch: 0132 loss_train: 0.1970 f1_micro_train: 0.8571 f1_macro_train: 0.8362 auc_train: 0.9720 f1_micro_val: 0.8522 f1_macro_val: 0.8327 auc_val: 0.9670
Epoch: 0133 loss_train: 0.1960 f1_micro_train: 0.8588 f1_macro_train: 0.8384 auc_train: 0.9722 f1_micro_val: 0.8478 f1_macro_val: 0.8266 auc_val: 0.9671
Epoch: 0134 loss_train: 0.1950 f1_micro_train: 0.8581 f1_macro_train: 0.8372 auc_train: 0.9725 f1_micro_val: 0.8517 f1_macro_val: 0.8326 auc_val: 0.9674
Epoch: 0135 loss_train: 0.1939 f1_micro_train: 0.8601 f1_macro_train: 0.8402 auc_train: 0.9727 f1_micro_val: 0.8517 f1_macro_val: 0.8320 auc_val: 0.9675
Epoch: 0136 loss_train: 0.1929 f1_micro_train: 0.8599 f1_macro_train: 0.8395 auc_train: 0.9730 f1_micro_val: 0.8500 f1_macro_val: 0.8299 auc_val: 0.9677
 Saving model ...
Epoch: 0137 loss_train: 0.1920 f1_micro_train: 0.8600 f1_macro_train: 0.8397 auc_train: 0.9732 f1_micro_val: 0.8533 f1_macro_val: 0.8343 auc_val: 0.9679
Epoch: 0138 loss_train: 0.1910 f1_micro_train: 0.8616 f1_macro_train: 0.8420 auc_train: 0.9734 f1_micro_val: 0.8511 f1_macro_val: 0.8309 auc_val: 0.9680
Epoch: 0139 loss_train: 0.1900 f1_micro_train: 0.8607 f1_macro_train: 0.8404 auc_train: 0.9737 f1_micro_val: 0.8533 f1_macro_val: 0.8340 auc_val: 0.9683
Epoch: 0140 loss_train: 0.1890 f1_micro_train: 0.8625 f1_macro_train: 0.8431 auc_train: 0.9739 f1_micro_val: 0.8522 f1_macro_val: 0.8319 auc_val: 0.9684
Epoch: 0141 loss_train: 0.1881 f1_micro_train: 0.8627 f1_macro_train: 0.8432 auc_train: 0.9741 f1_micro_val: 0.8539 f1_macro_val: 0.8339 auc_val: 0.9686
 Saving model ...
Epoch: 0142 loss_train: 0.1872 f1_micro_train: 0.8630 f1_macro_train: 0.8435 auc_train: 0.9743 f1_micro_val: 0.8556 f1_macro_val: 0.8369 auc_val: 0.9688
Epoch: 0143 loss_train: 0.1863 f1_micro_train: 0.8642 f1_macro_train: 0.8454 auc_train: 0.9745 f1_micro_val: 0.8550 f1_macro_val: 0.8352 auc_val: 0.9689
 Saving model ...
Epoch: 0144 loss_train: 0.1853 f1_micro_train: 0.8641 f1_macro_train: 0.8447 auc_train: 0.9747 f1_micro_val: 0.8572 f1_macro_val: 0.8390 auc_val: 0.9691
Epoch: 0145 loss_train: 0.1844 f1_micro_train: 0.8654 f1_macro_train: 0.8467 auc_train: 0.9749 f1_micro_val: 0.8556 f1_macro_val: 0.8362 auc_val: 0.9693
Epoch: 0146 loss_train: 0.1835 f1_micro_train: 0.8660 f1_macro_train: 0.8470 auc_train: 0.9751 f1_micro_val: 0.8561 f1_macro_val: 0.8370 auc_val: 0.9695
 Saving model ...
Epoch: 0147 loss_train: 0.1826 f1_micro_train: 0.8667 f1_macro_train: 0.8480 auc_train: 0.9753 f1_micro_val: 0.8589 f1_macro_val: 0.8407 auc_val: 0.9696
Epoch: 0148 loss_train: 0.1817 f1_micro_train: 0.8673 f1_macro_train: 0.8487 auc_train: 0.9755 f1_micro_val: 0.8572 f1_macro_val: 0.8380 auc_val: 0.9698
 Saving model ...
Epoch: 0149 loss_train: 0.1808 f1_micro_train: 0.8682 f1_macro_train: 0.8496 auc_train: 0.9757 f1_micro_val: 0.8594 f1_macro_val: 0.8415 auc_val: 0.9700
Epoch: 0150 loss_train: 0.1800 f1_micro_train: 0.8690 f1_macro_train: 0.8510 auc_train: 0.9759 f1_micro_val: 0.8589 f1_macro_val: 0.8398 auc_val: 0.9702
 Saving model ...
Epoch: 0151 loss_train: 0.1791 f1_micro_train: 0.8688 f1_macro_train: 0.8502 auc_train: 0.9761 f1_micro_val: 0.8600 f1_macro_val: 0.8426 auc_val: 0.9704
 Saving model ...
Epoch: 0152 loss_train: 0.1782 f1_micro_train: 0.8704 f1_macro_train: 0.8525 auc_train: 0.9763 f1_micro_val: 0.8616 f1_macro_val: 0.8429 auc_val: 0.9706
 Saving model ...
Epoch: 0153 loss_train: 0.1774 f1_micro_train: 0.8709 f1_macro_train: 0.8527 auc_train: 0.9765 f1_micro_val: 0.8622 f1_macro_val: 0.8451 auc_val: 0.9707
 Saving model ...
Epoch: 0154 loss_train: 0.1765 f1_micro_train: 0.8710 f1_macro_train: 0.8533 auc_train: 0.9767 f1_micro_val: 0.8639 f1_macro_val: 0.8460 auc_val: 0.9709
Epoch: 0155 loss_train: 0.1757 f1_micro_train: 0.8718 f1_macro_train: 0.8540 auc_train: 0.9768 f1_micro_val: 0.8628 f1_macro_val: 0.8449 auc_val: 0.9711
Epoch: 0156 loss_train: 0.1748 f1_micro_train: 0.8718 f1_macro_train: 0.8543 auc_train: 0.9770 f1_micro_val: 0.8639 f1_macro_val: 0.8458 auc_val: 0.9713
 Saving model ...
Epoch: 0157 loss_train: 0.1740 f1_micro_train: 0.8722 f1_macro_train: 0.8546 auc_train: 0.9772 f1_micro_val: 0.8644 f1_macro_val: 0.8473 auc_val: 0.9715
Epoch: 0158 loss_train: 0.1732 f1_micro_train: 0.8728 f1_macro_train: 0.8556 auc_train: 0.9774 f1_micro_val: 0.8644 f1_macro_val: 0.8471 auc_val: 0.9717
 Saving model ...
Epoch: 0159 loss_train: 0.1724 f1_micro_train: 0.8731 f1_macro_train: 0.8556 auc_train: 0.9776 f1_micro_val: 0.8650 f1_macro_val: 0.8479 auc_val: 0.9719
 Saving model ...
Epoch: 0160 loss_train: 0.1716 f1_micro_train: 0.8735 f1_macro_train: 0.8564 auc_train: 0.9777 f1_micro_val: 0.8655 f1_macro_val: 0.8483 auc_val: 0.9720
 Saving model ...
Epoch: 0161 loss_train: 0.1708 f1_micro_train: 0.8737 f1_macro_train: 0.8565 auc_train: 0.9779 f1_micro_val: 0.8661 f1_macro_val: 0.8490 auc_val: 0.9722
 Saving model ...
Epoch: 0162 loss_train: 0.1700 f1_micro_train: 0.8744 f1_macro_train: 0.8574 auc_train: 0.9781 f1_micro_val: 0.8661 f1_macro_val: 0.8491 auc_val: 0.9724
 Saving model ...
Epoch: 0163 loss_train: 0.1692 f1_micro_train: 0.8750 f1_macro_train: 0.8580 auc_train: 0.9783 f1_micro_val: 0.8661 f1_macro_val: 0.8491 auc_val: 0.9726
 Saving model ...
Epoch: 0164 loss_train: 0.1684 f1_micro_train: 0.8756 f1_macro_train: 0.8585 auc_train: 0.9784 f1_micro_val: 0.8672 f1_macro_val: 0.8497 auc_val: 0.9728
Epoch: 0165 loss_train: 0.1676 f1_micro_train: 0.8761 f1_macro_train: 0.8593 auc_train: 0.9786 f1_micro_val: 0.8666 f1_macro_val: 0.8491 auc_val: 0.9729
 Saving model ...
Epoch: 0166 loss_train: 0.1669 f1_micro_train: 0.8765 f1_macro_train: 0.8598 auc_train: 0.9788 f1_micro_val: 0.8688 f1_macro_val: 0.8513 auc_val: 0.9731
 Saving model ...
Epoch: 0167 loss_train: 0.1661 f1_micro_train: 0.8767 f1_macro_train: 0.8598 auc_train: 0.9789 f1_micro_val: 0.8688 f1_macro_val: 0.8514 auc_val: 0.9732
Epoch: 0168 loss_train: 0.1654 f1_micro_train: 0.8772 f1_macro_train: 0.8612 auc_train: 0.9791 f1_micro_val: 0.8688 f1_macro_val: 0.8500 auc_val: 0.9734
 Saving model ...
Epoch: 0169 loss_train: 0.1648 f1_micro_train: 0.8782 f1_macro_train: 0.8608 auc_train: 0.9793 f1_micro_val: 0.8700 f1_macro_val: 0.8548 auc_val: 0.9735
Epoch: 0170 loss_train: 0.1645 f1_micro_train: 0.8786 f1_macro_train: 0.8637 auc_train: 0.9794 f1_micro_val: 0.8605 f1_macro_val: 0.8361 auc_val: 0.9737
Epoch: 0171 loss_train: 0.1651 f1_micro_train: 0.8755 f1_macro_train: 0.8551 auc_train: 0.9796 f1_micro_val: 0.8666 f1_macro_val: 0.8539 auc_val: 0.9737
Epoch: 0172 loss_train: 0.1666 f1_micro_train: 0.8770 f1_macro_train: 0.8644 auc_train: 0.9795 f1_micro_val: 0.8567 f1_macro_val: 0.8240 auc_val: 0.9736
 Saving model ...
Epoch: 0173 loss_train: 0.1702 f1_micro_train: 0.8680 f1_macro_train: 0.8406 auc_train: 0.9797 f1_micro_val: 0.8672 f1_macro_val: 0.8553 auc_val: 0.9738
Epoch: 0174 loss_train: 0.1689 f1_micro_train: 0.8766 f1_macro_train: 0.8649 auc_train: 0.9794 f1_micro_val: 0.8539 f1_macro_val: 0.8255 auc_val: 0.9738
 Saving model ...
Epoch: 0175 loss_train: 0.1667 f1_micro_train: 0.8723 f1_macro_train: 0.8487 auc_train: 0.9800 f1_micro_val: 0.8744 f1_macro_val: 0.8599 auc_val: 0.9742
Epoch: 0176 loss_train: 0.1618 f1_micro_train: 0.8798 f1_macro_train: 0.8652 auc_train: 0.9800 f1_micro_val: 0.8711 f1_macro_val: 0.8566 auc_val: 0.9744
Epoch: 0177 loss_train: 0.1600 f1_micro_train: 0.8808 f1_macro_train: 0.8663 auc_train: 0.9804 f1_micro_val: 0.8650 f1_macro_val: 0.8405 auc_val: 0.9745
 Saving model ...
Epoch: 0178 loss_train: 0.1611 f1_micro_train: 0.8771 f1_macro_train: 0.8563 auc_train: 0.9805 f1_micro_val: 0.8749 f1_macro_val: 0.8619 auc_val: 0.9746
Epoch: 0179 loss_train: 0.1614 f1_micro_train: 0.8800 f1_macro_train: 0.8671 auc_train: 0.9803 f1_micro_val: 0.8611 f1_macro_val: 0.8392 auc_val: 0.9745
Epoch: 0180 loss_train: 0.1604 f1_micro_train: 0.8794 f1_macro_train: 0.8613 auc_train: 0.9807 f1_micro_val: 0.8733 f1_macro_val: 0.8572 auc_val: 0.9749
Epoch: 0181 loss_train: 0.1577 f1_micro_train: 0.8820 f1_macro_train: 0.8666 auc_train: 0.9807 f1_micro_val: 0.8733 f1_macro_val: 0.8584 auc_val: 0.9750
Epoch: 0182 loss_train: 0.1564 f1_micro_train: 0.8829 f1_macro_train: 0.8686 auc_train: 0.9811 f1_micro_val: 0.8633 f1_macro_val: 0.8405 auc_val: 0.9751
 Saving model ...
Epoch: 0183 loss_train: 0.1569 f1_micro_train: 0.8806 f1_macro_train: 0.8621 auc_train: 0.9812 f1_micro_val: 0.8777 f1_macro_val: 0.8639 auc_val: 0.9752
Epoch: 0184 loss_train: 0.1571 f1_micro_train: 0.8825 f1_macro_train: 0.8690 auc_train: 0.9810 f1_micro_val: 0.8644 f1_macro_val: 0.8445 auc_val: 0.9752
Epoch: 0185 loss_train: 0.1559 f1_micro_train: 0.8829 f1_macro_train: 0.8660 auc_train: 0.9814 f1_micro_val: 0.8749 f1_macro_val: 0.8585 auc_val: 0.9755
Epoch: 0186 loss_train: 0.1536 f1_micro_train: 0.8843 f1_macro_train: 0.8688 auc_train: 0.9815 f1_micro_val: 0.8749 f1_macro_val: 0.8599 auc_val: 0.9756
Epoch: 0187 loss_train: 0.1530 f1_micro_train: 0.8848 f1_macro_train: 0.8704 auc_train: 0.9816 f1_micro_val: 0.8639 f1_macro_val: 0.8427 auc_val: 0.9756
Epoch: 0188 loss_train: 0.1537 f1_micro_train: 0.8835 f1_macro_train: 0.8662 auc_train: 0.9818 f1_micro_val: 0.8755 f1_macro_val: 0.8606 auc_val: 0.9758
Epoch: 0189 loss_train: 0.1533 f1_micro_train: 0.8848 f1_macro_train: 0.8711 auc_train: 0.9817 f1_micro_val: 0.8683 f1_macro_val: 0.8485 auc_val: 0.9759
Epoch: 0190 loss_train: 0.1517 f1_micro_train: 0.8848 f1_macro_train: 0.8682 auc_train: 0.9821 f1_micro_val: 0.8749 f1_macro_val: 0.8595 auc_val: 0.9761
Epoch: 0191 loss_train: 0.1503 f1_micro_train: 0.8870 f1_macro_train: 0.8725 auc_train: 0.9822 f1_micro_val: 0.8755 f1_macro_val: 0.8599 auc_val: 0.9762
Epoch: 0192 loss_train: 0.1500 f1_micro_train: 0.8878 f1_macro_train: 0.8732 auc_train: 0.9822 f1_micro_val: 0.8694 f1_macro_val: 0.8509 auc_val: 0.9762
Epoch: 0193 loss_train: 0.1501 f1_micro_train: 0.8851 f1_macro_train: 0.8689 auc_train: 0.9824 f1_micro_val: 0.8744 f1_macro_val: 0.8592 auc_val: 0.9764
Epoch: 0194 loss_train: 0.1494 f1_micro_train: 0.8885 f1_macro_train: 0.8751 auc_train: 0.9824 f1_micro_val: 0.8755 f1_macro_val: 0.8557 auc_val: 0.9765
Epoch: 0195 loss_train: 0.1485 f1_micro_train: 0.8861 f1_macro_train: 0.8692 auc_train: 0.9827 f1_micro_val: 0.8755 f1_macro_val: 0.8602 auc_val: 0.9766
Epoch: 0196 loss_train: 0.1478 f1_micro_train: 0.8885 f1_macro_train: 0.8750 auc_train: 0.9828 f1_micro_val: 0.8777 f1_macro_val: 0.8612 auc_val: 0.9768
Epoch: 0197 loss_train: 0.1473 f1_micro_train: 0.8884 f1_macro_train: 0.8724 auc_train: 0.9828 f1_micro_val: 0.8783 f1_macro_val: 0.8626 auc_val: 0.9768
Epoch: 0198 loss_train: 0.1464 f1_micro_train: 0.8898 f1_macro_train: 0.8757 auc_train: 0.9830 f1_micro_val: 0.8777 f1_macro_val: 0.8618 auc_val: 0.9769
Epoch: 0199 loss_train: 0.1456 f1_micro_train: 0.8907 f1_macro_train: 0.8764 auc_train: 0.9831 f1_micro_val: 0.8777 f1_macro_val: 0.8609 auc_val: 0.9770
 Saving model ...
Epoch: 0200 loss_train: 0.1451 f1_micro_train: 0.8899 f1_macro_train: 0.8746 auc_train: 0.9832 f1_micro_val: 0.8794 f1_macro_val: 0.8644 auc_val: 0.9770
Load model from epoch 199
Model Testing: f1_micro_test: 0.8791 f1_macro_test: 0.8640 auc_test: 0.9805

Process finished with exit code 0

'''

