## OPTIMA: Boosting Prompt Tuning with Domain Adaptation for Few-shot Learning

Very busy recently. Code will come soon within Dec 2022.

Note: This is a work in progress. 

Data and code for paper: [Improving the Sample Efficiency of Prompt Tuning with Domain Adaptation](https://arxiv.org/abs/2210.02952), Findings of the Association for Computational Linguistics: EMNLP 2022 

## Using OPTIMA

OPTIMA is essentially a combination of adversarial training and unsupervised domain adaptation.

#### 1. Pretrain Soft Prompt with OPTIMA

We need to pretrain a soft prompt that will be used for initialization. 


#### 2. Using Pretrained Soft Prompt as Initialization


## Reproduce Experiments
Note: create a new env before installing the following.
* Python version: Python 3.8.8
* packages used are included in requirements.txt: ```pip install -r requirements.txt```
* a picture of the whole experimental env are described in /env/pip-list.txt


### Run zero-shot experiments

1. Using labeled source-domain dataset only
   * E.g., we can pretrain soft prompts using *labeled* MRPC training set
   
        `python train.py --cfg src_full/qqp`
        
2. Using labeled source-domain datatset and unlabeled target-domain dataset and select the best model using MRPC validation set:
        
   * pretrain soft prompts with OPTIMA:

        `python train.py --cfg optima_full/qqp`
        
   * pretrain soft prompts with FreeAT:
        
        `python train.py --cfg free_full/qqp`

   * After pretraining, we obtain soft prompts pretrained on MRPC adapted to QQP. To guarantee the reliability of results, we may run for several seeds.

Configurations for other *pretraining* data pairs can be found in `/tgt_unlabeled_configs/optima_full/*.yml`

### Run few-shot experiments


1. Without pretraining
    * The naive Prompt Tuning:
    
        `python train.py --cfg tgt_sup_shot/qqp`
        
    * Fine-tuning the entire T5 without prepending any soft prompt:
        
        `python train.py --cfg tgt_sup_shot/qqp --tune --eval`
    
    * Fine-tuning the entire T5 together with a set of prepended soft prompt:
           
        `python train.py --cfg tgt_sup_shot/qqp --tune --src_soft_num 100`
    
2. Using pretrained soft prompts
    * choose one checkpoint of soft prompts pretrained on source-domain data only:
    
        `python train.py --cfg tgt_sup_shot/qqp --reload --ckpt src_full/qqp --seed 111`

    * choose one checkpoint of soft prompts pretrained with OPTIMA:
    
        `python train.py --cfg tgt_sup_shot/qqp --reload --ckpt optima_full/qqp --seed 111`


Configurations for other *few-shot learning* data pairs can be found in `/tgt_labeled_configs/tgt_sup_shot/*.yml`
