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

'''

