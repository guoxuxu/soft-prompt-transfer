# OPTIMA: Boosting Prompt Tuning with Domain Adaptation for Few-shot Learning

Code to reproduce the experiments from the paper: [Improving the Sample Efficiency of Prompt Tuning with Domain Adaptation](https://arxiv.org/abs/2210.02952), Findings of the Association for Computational Linguistics: EMNLP 2022 

## Table of Contents
* [Environment](#environment)
* [Preliminary Knowledge](#preliminary-knowledge)
* [The Hybrid Prompt Template](#the-hybrid-prompt-template)
* [Soft Prompt Initialization](#soft-prompt-initialization)
* [Model Configuration](#model-configuration)
* [Training A Soft Prompt with OPTIMA](#pre-training-a-soft-prompt-with-optima)
    * [Construct an Unsupervised Domain Adaptation Setting](#construct-an-unsupervised-domain-adaptation-setting)
    * [Minimizing Domain Gap with Transferable Adversarial Training](#minimizing-domain-gap-with-transferable-adversarial-training)
* [Applying a Pretrained Soft Prompt](#applying-a-pretrained-soft-prompt)
    * [Zero-shot Learning Setting](#zero-shot-learning-setting)
    * [Few-shot Learning Setting](#few-shot-learning-setting)
* [Released Soft Prompts](#released-soft-prompts) 
* [Extra Resources](#extra-resources)
* [How to Cite](#how-to-cite)
 
## Environment
It's better to first create a virtual environment before installing the following.
* Python version: Python 3.8.8
* Packages used are included in ```requirements.txt```, install them by running ```pip install -r requirements.txt```
* A picture of the whole experimental env are described in ```/env/pip-list.txt```
* We used an earlier version of [OpenPrompt]() to set up prompt tuning.

## Preliminary Knowledge
* (Soft) Prompt Tuning: [Paper](), [Code]()
    * Soft prompt refers to prompts whose embeddings are trainable. Hard prompts refers to untunable natual language prompts.
    * The large pretrained language models (PLMs) are frozen when we tune prompt embeddings. If you train PLMs as well, such practcie generally refers to prompt-based fine-tuning ([Paper](), [Code]()). 
* __However, the effectiveness of prompt tuning relies on large enough labeled training data. Prompt tuning lags far behind full-model tuning in few-shot learning setting. It easily falls in overfiting despite having far fewer trainable parameters than full-model tuning.__
    * One possible way is to pretrain soft prompts on largly available unlabeled corpus like how we pretrain large language models: [Paper](), [Code]()
    * Another solution is to leverage task-specific similar domains to perform domain adaptation: [Paper](), [Code]()
* A quick comparison of the two kinds of possible solutions:
    * The capacity of soft prompts is limited. It's hard to learn from huge online corpora like how large language models do.
    * Task-specific data is more straightforward for low-capacity soft prompts to learn task-related prompt information. However, it has to deal with the challenge of underlying domain gap between similar but different datasets.
* OPTIMA solves the challenge of domain gap via alternating two optimization steps:
    * Generate adversarial perturbations to make source domain data to be indistinguishable with target domain data. (Reference to Virtual Adversarial Training [Paper]() and [Code]())
    * Train soft prompts to guide large language models to make consistent predictions against such adversarial perturbations.  
* Note: 1) Virtual Adversarial Training is a semi-supervised learning algorithm for single domain data. It cannot be directly applied to unsupervised domain adaptation setting as there is a domain gap between labeled data and unlabeled data. 2) Perturbations generated on unlabeled target domain data is unreliable because there're no supervisory signals form the target domain.

## The Hybrid Prompt Template
Studies in this [paper](https://aclanthology.org/2022.acl-long.576.pdf) shows that hybrid prompts, i.e., using both soft and hard prompts, achieves better downstream task performance (Page 3, Table 1). In this paper, we adopt hybrid template, in which the soft prompts are prepended to the hard prompts as shown below. This is an example for QQP dataset:

| Virtual tokens for soft prompt                   | Input data wrapped by hard prompts            | Label to be predicted  |
|--------------------------------------------------|-----------------------------------------------|------------------------|
| P<sub>1</sub>, P<sub>2</sub>, ..., P<sub>m</sub> | <Sentence 1> and <Sentence 2> are equivalent? | [MASK]                 |

<Sentence 1> and <Sentence 2> are original sentence pairs provided in the dataset. P<sub>1</sub>, P<sub>2</sub>, ..., P<sub>m</sub> are soft prompts to be trained on a labeled dataset. A PLM is required to predict [MASK], e.g., to be "Yes" or "No" for this QQP dataset. Similar label words such as "True" and "False" are also applicable. The previous studies mentioned above also showcased the impact of choosing different label words.

A comprehensive hard prompt templates for each dataset are under ```./scripts```. Template ids are specified in configuration files under ```./tgt_unlabeled_configs```. This paper evaluated on six __sentence pair classification__ datasets: paraphrase detection (QQP and MRPC), natural language inference (MNLI, SNLI, SICK, CB).

## Soft Prompt Initialization
The prompt tuning [paper](https://aclanthology.org/2021.emnlp-main.243/) provided a study on choosing the initialization method for soft prompts (Page 5, Figure 3b). They found that generally using random vocab token embeddings is helpful enough. We adopt this setting in this paper. We init the soft prompt embeddings using the first N alphabetic token embeddings of T5. The exact tokens are printed in the file ```./data/tokens.txt```

## Model Configuration
This paper is based on T5-large, which has ~770M parameters with 24 transformer layers, 1024 hidden states. A comprehensive studies on how prompt tuning takes effect with the size of large PLMs can be found in the [survey](https://arxiv.org/pdf/2203.06904.pdf) (Page ). A general conclusion is that it generally requires T5-large and above.  

## Pre-Training A Soft Prompt with OPTIMA
### 1. Construct an Unsupervised Domain Adaptation Setting
We suspect that purely pretraining a soft prompt on the source domain may lead to overfitting shortcut features that are inherent to the source-domain datasets. To encourage soft prompts to learn domain or dataset-agnostic knowledge and present consistent performance on similar domains or datasets, we propose to pretrain soft prompts under an unsupervised domain adaptation (UDA) setting.

**UDA setting**: prepare a source-domain dataset and a target-domain dataset, the source domain is labeled but the target domain is unlabeled. The objective of UDA is to train a model on the labeled source-domain dataset and use the unlabeled target-domain data to encourage the model to generalize to the target domain. No labels for validation either. Performance comparison will be made on a held-out labeled test set.

This paper constructed the following UDA settings for pretraining soft prompts:

|               | MRPC &rarr; QQP | QQP &rarr; MRPC | SNLI &rarr; MNLI | SNLI &rarr; SICK | SNLI &rarr; CB | MNLI &rarr; SNLI | MNLI &rarr; SICK | MNLI &rarr; CB |  
|---------------|-----------------|--------------|----------------|-----------------|----------------|---------------|---------------|---------------|
| Prompt Tuning | 48.4<sub> 4.9   | 53.1<sub> 11.4 | 33.4<sub> 1.6  | 61.5<sub> 7.8   | 38.3<sub> 13.6 | 34.6<sub> 2.4 | /             | /             |
| PPT           | 55.6<sub> 4.9   | 55.9<sub> 11.5 | 34.4<sub> 1.4  |  54.6<sub> 14.0 | 46.7<sub> 12.6 | 34.7<sub> 2.8 | /             | /             |
| OPTIMA        | 71.2<sub> 1.7   | 69.1<sub> 1.7 | 78.4<sub> 0.6  |  73.3<sub> 6.8  | 64.8<sub> 1.1  | 82.1<sub> 0.8 | 74.8<sub> 4.4 | 71.2<sub> 3.1 |

This table shows a comparison of few-shot performance among naive prompt tuning, pretraining prompt on open-domain corpora, and our proposed OPTIMA approach which pretrains prompt on task-specific related domains. Results show that pretraining data coming from general domains do not guarantee consistent performance improvement across benchmarks. In contrast, gaining knowledge from task-specific datasets benefit the most for target datasets.    

Note: 1) PPT use T5-XXL as the PLM. The pretrained prompt weights are downloaded following the [paper github repo](https://github.com/thu-coai/PPT). Prompt Tuning and OPTIMA use T5-Large. 2) We conduct 8-shot evaluation. We randomly sample 16 times on the original training dataset and obtain 16 different 8-shot training set for few-shot evaluation. The results are averaged and the standard deviation is indicated as the subscript of each result.

### 2. Bridging the Domain Gap with OPTIMA
#### 1. Pretraining soft prompts with Adversarial Training
We want to train such soft prompts that, when plugged into PLMs, can make consistent predictions against small perturbations over the original input. We expect that training soft prompts in this way can enhance their generalization performance (evaluated on both zero-shot and few-shot learning settings). 

For this purpose, we implement two popular adversarial training algorithms for prompt tuning: [Adversarial Training](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf) ([Free AT](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf), [FreeLB for NLP](https://arxiv.org/pdf/1909.11764.pdf)) and [Virtual Adversarial Training](https://arxiv.org/pdf/1412.6572.pdf). They are regularization terms that reflect a priori knowledge or belief regarding the PLMs. The priori is that the outputs of the PLMs are smooth with respect to the inputs. Training soft prompts with adversarial training terms encourages the soft prompts to prefer output distributions that are smooth with respect to the input data. Different from random perturbations, adversarial perturbations are always point to the directions of gradients that are perpendicular to the decision boundary. See Equations 7-9 in our paper.

```
python train.py --cfg free_full/qqp
```

```
python train.py --cfg vat_full/qqp
```

#### 2. How to deal with the domain gap between source and target domains during pretraining?
Adversarial training enhances generalization for in-domain learning. However, it cannot guarantee that the model still makes consistent predictions under domain divergence. Because the decision boundary that works in the source domain may no longer be suitable for the target domain.

Our motivation is that we can generate perturbations that make source domain to be similar as target domain data, such that the decision boundary produced on the perturbed source-domain data can be directly applied to the target-domain data. Such perturbations are obtained by adopting a domain discriminator, which is simply a MLP classifier, to work against the optimization of adversarial perturbations. See Equations 4-6 in our paper. The use of domain discriminator to form a GAN-like setup is very broad. See my previous [paper](https://aclanthology.org/2021.naacl-main.425/) for supervised setting. 

Assume that we obtained such perturbations that bridge the domain gap, we train soft prompts conditioned on the frozen T5 against these perturbations. Soft prompts and adversarial perturbations are alternately optimized throughout the training epochs. See Algorithm 1 in our paper. 

```
python train.py --cfg optima_full/qqp
```
For comparison, directly apllying prompt tuning on the source-domain datasets can be done by
```
python train.py --cfg src_full/qqp
```

## Applying a Pretrained Soft Prompt

### Zero-shot Learning Setting
To apply those pretrained soft prompts on unseen datasets, we first load the pretrained checkpoints and set to test mode.
E.g., we apply the soft prompts pretrained on the MRPC dataset on QQP dataset. 
```
python train.py --optima_full/qqp --best 111 --test
```
"--best 111" means load the best checkpoint under random seed 111.


### Few-shot Learning Setting
For few-shot learning setting, we are provided with 8-shot labeled examples for each dataset. We will first load a pretrained soft prompt checkpoint as warm start. 
Configurations for other *few-shot learning* data pairs can be found in `/tgt_labeled_configs/tgt_sup_shot/*.yml`

``` 
python train.py --cfg tgt_sup_shot/qqp --reload --ckpt optima_full/qqp --seed 111
```
For comparison, we can evaluate the performance of naive prompt tuning, prompt-based fine-tuning, and full-model tuning without soft prompts as follows:

Without pretraining：
* The naive Prompt Tuning:

    `python train.py --cfg tgt_sup_shot/qqp`
    
* Fine-tuning the entire T5 without prepending any soft prompt:
    
    `python train.py --cfg tgt_sup_shot/qqp --tune --eval`

* Fine-tuning the entire T5 together with a set of prepended soft prompt:
       
    `python train.py --cfg tgt_sup_shot/qqp --tune --src_soft_num 100`
    

## Released Soft Prompts
We provide the pretrained soft prompts using OPTIMA under folder ```checkpoints/```

## Extra Resources
* Presentation
    * [Video](https://www.youtube.com/watch?v=43MH9POr2r4&ab_channel=BoyangAlbertLi)
    * [Slides]()


## How to Cite
If you make use of this code or idea, please cite:
```
@inproceedings{guo-etal-2022-improving,
    title = "Improving the Sample Efficiency of Prompt Tuning with Domain Adaptation",
    author = "Guo, Xu  and
      Li, Boyang  and
      Yu, Han",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.258",
    pages = "3523--3537",
}
```