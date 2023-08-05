# OPTIMA: Boosting Prompt Tuning with Domain Adaptation for Few-shot Learning

Code to reproduce the experiments from the paper: [Improving the Sample Efficiency of Prompt Tuning with Domain Adaptation](https://arxiv.org/abs/2210.02952), Findings of
the Association for Computational Linguistics: EMNLP 2022

## Table of Contents


* [Preliminary Knowledge](#preliminary-knowledge)
* [Released Checkpoints](#released-checkpoints)
  * [How to Use](#how-to-use)
* [Reproduce Experiments](#reproduce-experiments)
  * [Environment](#environment)
  * [Data Preparation](data-preparation)
  * [The Hybrid Prompt Template](#the-hybrid-prompt-template)
  * [Model Configuration](#model-configuration)
  * [Training Soft Prompts with OPTIMA](#pre-training-a-soft-prompt-with-optima)
      * [Construct an Unsupervised Domain Adaptation Setting](#construct-an-unsupervised-domain-adaptation-setting)
      * [Minimizing Domain Gap with Transferable Adversarial Training](#minimizing-domain-gap-with-transferable-adversarial-training)
  * [Inference with Pretrained Soft Prompts](#applying-a-pretrained-soft-prompt)
      * [Zero-shot Learning Setting](#zero-shot-learning-setting)
      * [Few-shot Learning Setting](#few-shot-learning-setting)

* [How to Cite](#how-to-cite)
* [Extra Resources](#extra-resources)

## Preliminary Knowledge

* (Soft) Prompt Tuning: [Paper](https://aclanthology.org/2021.emnlp-main.243.pdf), [Code](https://github.com/google-research/prompt-tuning)
    * **Conceptual understaning:** Soft prompt refers to a sequence of tokens whose embeddings are trainable. We train the soft prompt on a labeled training set such that
      it can steer a pretrained language model (PLM) to make correct predictions. The length of the sequence and the initialization for the embeddings are to be
      determined with a held-out validation set (Studied in the prompt tuning paper). In contrast, hard prompt refers to a sequence of human-readable natural languages
      whose embeddings are fixed. We can treat hard prompt as additional task information used to augment the original input data. It can be a question: is it positive or
      negative? A hint: the sentiment of the review is _ ? Or, a few examples from the dataset. The [GPT-3 paper](https://arxiv.org/pdf/2005.14165.pdf) talked a lot about
      how to design hard prompts to steer GPT-2 in zero-shot and few-shot setting.
    * **Procedural standard：** The weights of PLMs are typically fixed when we tune the weights of soft prompt embeddings. If you tune PLMs together with soft prompts, it
      typically refers to prompt-based fine-tuning ([Paper](https://aclanthology.org/2021.acl-long.295/), [Code](https://github.com/princeton-nlp/LM-BFF)).
* __However, the effectiveness of prompt tuning relies on large enough labeled training data. Prompt tuning lags far behind full-model tuning in few-shot learning
  setting. It easily falls in overfiting despite having far fewer trainable parameters than full-model tuning.__
    * One possible way is to pretrain soft prompts on largly available unlabeled corpus like how we pretrain large language
      models: [Paper](https://aclanthology.org/2022.acl-long.576/), [Code](https://github.com/thu-coai/ppt)
    * Another solution is to leverage task-specific similar domains for pretraining: [Paper](https://aclanthology.org/2022.acl-long.346/),  [Code](https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/spot)
* A quick comparison of the two kinds of possible solutions:
    * The capacity of soft prompts is limited. It's hard to learn from huge online corpora like how large language models do.
    * Task-specific data is more straightforward for low-capacity soft prompts to learn task-related prompt information. However, it has to deal with the challenge of
      underlying domain gap between similar but different datasets.
* OPTIMA solves the challenge of domain gap via alternating two optimization steps:
    * Generate adversarial perturbations to make source domain data to be indistinguishable with target domain data. (Reference to Virtual Adversarial Training [Paper](https://openreview.net/pdf?id=r1X3g2_xl)
      and [Code](https://github.com/takerum/vat_tf/blob/master/vat.py))
    * Train soft prompts to guide large language models to make consistent predictions against such adversarial perturbations.
* Note: 1) Virtual Adversarial Training is a semi-supervised learning algorithm for single domain data. It cannot be directly applied to unsupervised domain adaptation
  setting as there is a domain gap between labeled data and unlabeled data. 2) Perturbations generated on unlabeled target domain data is unreliable because there're no
  supervisory signals form the target domain.


## Released Checkpoints
We release the soft prompts pretrained using OPTIMA in folder ```checkpoints/```
* soft prompts trained for QQP task： ```qqp_prompt.pt``` 
* soft prompts trained for MRPC task: ```mrpc_prompt.pt```
* soft prompts trained for SNLI task: ```snli_prompt.pt```
* soft prompts trained for MNLI task: ```mnli_prompt.pt```
* soft prompts trained for SICK task
  * using MNLI dataset as the source domain: ```mnli/msick_prompt.pt```
  * using SNLI dataset as the source domain: ```snli/ssick_prompt.pt```
* soft prompts trained for CB task
  * using MNLI dataset as the source domain: ```mnli/mcb_prompt.pt```
  * using SNLI dataset as the source domain: ```snli/scb_prompt.pt```

### How to Use
load the soft prompt into your embedding table, see more in the ```process_batch``` function at ```layers/soft_template.py:74```
   * ```pretrained_prompts = torch.load(**.pt)```
   * ```inputs_embeds = self.raw_embedding(batch['input_ids'])```
   * ```inputs_embeds = torch.cat([pretrained_prompts, inputs_embeds], 1)```
   * do remember to extend your attention mask together.

## Reproduce Experiments
### Environment

It's better to first create a virtual environment before installing the packages.

* Python version: Python 3.8.8
* Packages used are included in ```requirements.txt```, install them by running ```pip install -r requirements.txt```
* We used an earlier version of [OpenPrompt](https://github.com/thunlp/OpenPrompt) to set up prompt tuning.

### Data Preparation
For pretraining, we use the full labeled dataset for the source domain and unlabeled dataset for the target domain.
For few-shot learning, we follow [LM-BFF](https://github.com/princeton-nlp/LM-BFF/tree/main/data) to make few-shot data splits. 
        
        cd data
        bash download_data.sh
        cd ..
        python generate_k_shot.py

Modify the```data_dir``` argument accordingly.

### The Hybrid Prompt Template

Studies in this [paper](https://aclanthology.org/2022.acl-long.576.pdf) shows that hybrid prompts, i.e., using both soft and hard prompts, achieves better downstream task
performance (Page 3, Table 1). In this paper, we adopt hybrid template, in which the soft prompts are prepended to the hard prompts as shown below. This is an example for
QQP dataset:

| Virtual tokens for soft prompt                   | Input data wrapped by hard prompts            | Label to be predicted  |
|--------------------------------------------------|-----------------------------------------------|------------------------|
| P<sub>1</sub>, P<sub>2</sub>, ..., P<sub>m</sub> | <Sentence 1> and <Sentence 2> are equivalent? | [MASK]                 |

<Sentence 1> and <Sentence 2> are original sentence pairs provided in the dataset. P<sub>1</sub>, P<sub>2</sub>, ..., P<sub>m</sub> are soft prompts to be trained on a
labeled dataset. A PLM is required to predict [MASK], e.g., to be "Yes" or "No" for this QQP dataset. Similar label words such as "True" and "False" are also applicable.
The previous studies mentioned above also showcased the impact of choosing different label words.

A comprehensive hard prompt templates for each dataset are under ```./scripts```. Template ids are specified in configuration files under ```./tgt_unlabeled_configs```.
This paper evaluated on six __sentence pair classification__ datasets: paraphrase detection (QQP and MRPC), natural language inference (MNLI, SNLI, SICK, CB).

### Soft Prompts Initialization

The prompt tuning [paper](https://aclanthology.org/2021.emnlp-main.243/) provided a study on choosing the initialization method for soft prompts (Page 5, Figure 3b). They
found that generally using random vocab token embeddings is helpful enough. We adopt this setting in this paper. We initialize the soft prompt embeddings using the first
N alphabetic token embeddings of T5. The exact tokens are printed in the file ```./data/tokens.txt```

### Model Configuration

This paper is based on T5-large, which has ~770M parameters with 24 transformer layers, 1024 hidden states. A comprehensive studies on how prompt tuning takes effect with
the size of large PLMs can be found in the [survey](https://arxiv.org/pdf/2203.06904.pdf) (Page ). A general conclusion is that it generally requires T5-large and above.

### Pre-Training Soft Prompts with OPTIMA

#### 1. Construct an Unsupervised Domain Adaptation Setting

We suspect that purely pretraining a soft prompt on the source domain may lead to overfitting shortcut features that are inherent to the source-domain datasets. To
encourage soft prompts to learn domain or dataset-agnostic knowledge and present consistent performance on similar domains or datasets, we propose to pretrain soft
prompts under an unsupervised domain adaptation (UDA) setting.

**UDA setting**: prepare a source-domain dataset and a target-domain dataset, the source domain is labeled but the target domain is unlabeled. The objective of UDA is to
train a model on the labeled source-domain dataset and use the unlabeled target-domain data to encourage the model to generalize to the target domain. No labels for
validation either. Performance comparison will be made on a held-out labeled test set.

This paper constructed the following UDA settings for pretraining soft prompts:

|               | MRPC &rarr; QQP | QQP &rarr; MRPC | SNLI &rarr; MNLI | SNLI &rarr; SICK | SNLI &rarr; CB | MNLI &rarr; SNLI | MNLI &rarr; SICK | MNLI &rarr; CB |  
|---------------|-----------------|--------------|----------------|-----------------|----------------|---------------|---------------|---------------|
| Prompt Tuning | 48.4<sub> 4.9   | 53.1<sub> 11.4 | 33.4<sub> 1.6  | 61.5<sub> 7.8   | 38.3<sub> 13.6 | 34.6<sub> 2.4 | /             | /             |
| PPT           | 55.6<sub> 4.9   | 55.9<sub> 11.5 | 34.4<sub> 1.4  |  54.6<sub> 14.0 | 46.7<sub> 12.6 | 34.7<sub> 2.8 | /             | /             |
| OPTIMA        | 71.2<sub> 1.7   | 69.1<sub> 1.7 | 78.4<sub> 0.6  |  73.3<sub> 6.8  | 64.8<sub> 1.1  | 82.1<sub> 0.8 | 74.8<sub> 4.4 | 71.2<sub> 3.1 |

This table shows a comparison of few-shot performance among naive prompt tuning, pretraining prompt on open-domain corpora, and our proposed OPTIMA approach which
pretrains prompt on task-specific related domains. Results show that pretraining data coming from general domains do not guarantee consistent performance improvement
across benchmarks. In contrast, gaining knowledge from task-specific datasets benefit the most for target datasets.

Note: 1) PPT use T5-XXL as the PLM. The pretrained prompt weights are downloaded following the [paper github repo](https://github.com/thu-coai/PPT). Prompt Tuning and
OPTIMA use T5-Large. 2) We conduct 8-shot evaluation. We randomly sample 16 times on the original training dataset and obtain 16 different 8-shot training set for
few-shot evaluation. The results are averaged and the standard deviation is indicated as the subscript of each result.

#### 2. Bridging the Domain Gap with OPTIMA

##### 1. Pretraining soft prompts with Adversarial Training

We want to train such soft prompts that, when plugged into PLMs, can make consistent predictions against small perturbations over the original input. We expect that
training soft prompts in this way can enhance their generalization performance (evaluated on both zero-shot and few-shot learning settings).

For this purpose, we implement two popular adversarial training algorithms for prompt
tuning: [Adversarial Training](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf) ([Free AT](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf)
, [FreeLB for NLP](https://arxiv.org/pdf/1909.11764.pdf)) and [Virtual Adversarial Training](https://arxiv.org/pdf/1412.6572.pdf). They are regularization terms that
reflect a priori knowledge or belief regarding the PLMs. The priori is that the outputs of the PLMs are smooth with respect to the inputs. Training soft prompts with
adversarial training terms encourages the soft prompts to prefer output distributions that are smooth with respect to the input data. Different from random perturbations,
adversarial perturbations are always point to the directions of gradients that are perpendicular to the decision boundary. See Equations 7-9 in our paper.

adversarial training:
```
python train.py --config free_full/qqp
```
virtual adversarial training:
```
python train.py --config vat_full/qqp
```
The hyperparameter ```adv_max_norm``` can be used to stablize training by normalizing the magnitude of perturbations. In my experiments, I tried 0.1 and 0.3 but the performance gains are not consistent. 

##### 2. How to deal with the domain gap between source and target domains during pretraining?

Adversarial training enhances generalization for in-domain learning. However, it cannot guarantee that the model still makes consistent predictions under domain
divergence. Because the decision boundary that works in the source domain may no longer be suitable for the target domain.

Our motivation is that we can generate perturbations that make source domain to be similar as target domain data, such that the decision boundary produced on the
perturbed source-domain data can be directly applied to the target-domain data. Such perturbations are obtained by adopting a domain discriminator, which is simply a MLP
classifier, to work against the optimization of adversarial perturbations. See Equations 4-6 in our paper. The use of domain discriminator to form a GAN-like setup is
very broad. See my previous [paper](https://aclanthology.org/2021.naacl-main.425/) for supervised setting.

Assume that we obtained such perturbations that bridge the domain gap, we train soft prompts conditioned on the frozen T5 against these perturbations. Soft prompts and
adversarial perturbations are alternately optimized throughout the training epochs. See Algorithm 1 in our paper.

```
python train.py --config optima_full/qqp
```

For comparison, directly apllying prompt tuning on the source-domain datasets can be done by

```
python train.py --config src_full/qqp
```

### Inference with Pretrained Soft Prompts

#### Zero-shot Learning Setting

To apply those pretrained soft prompts on unseen datasets, we first load the pretrained checkpoints and set to test mode. E.g., we apply the soft prompts pretrained on
the MRPC dataset on QQP dataset.

```
python train.py --optima_full/qqp --test
```

print test results across random seeds:
```
python train.py --optima_full/qqp --summarize_seeds --print_test
```

#### Few-shot Learning Setting

For few-shot learning setting, we are provided with 8-shot labeled examples for each dataset. We will first load a pretrained soft prompt checkpoint as warm start.
Configurations for other *few-shot learning* data pairs can be found in `/tgt_labeled_configs/tgt_sup_shot/*.yml`

``` 
python train.py --config tgt_sup_shot/qqp --reload --ckpt optima_full/qqp --load_seed 111
```
test
``` 
python train.py --config tgt_sup_shot/qqp --reload --ckpt optima_full/qqp --load_seed 111 --test
```
print test results under 16 randomly sampled few-shot training sets for each pretrained checkpoint.

``` 
python train.py --config tgt_sup_shot/qqp --reload --ckpt optima_full/qqp --load_seed 111 --summarize_seeds --print_test
```


For comparison, we can evaluate the performance of naive prompt tuning, prompt-based fine-tuning, and full-model tuning without soft prompts as follows:

Without pretraining：

* The naive Prompt Tuning:

  `python train.py --config tgt_sup_shot/qqp`

* Fine-tuning the entire T5 without prepending any soft prompt:

  `python train.py --config tgt_sup_shot/qqp --tune --eval`

* Fine-tuning the entire T5 together with a set of prepended soft prompt:

  `python train.py --config tgt_sup_shot/qqp --tune --src_soft_num 100`



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

## Extra Resources

* Presentation
    * [Video](https://www.youtube.com/watch?v=43MH9POr2r4&ab_channel=BoyangAlbertLi)
    * [Slides]()
