## Improving the Sample Efficiency of Prompt Tuning with Domain Adaptation

*Findings of the Association for Computational Linguistics: EMNLP 2022*


We propose a method named - bOosting Prompt TunIng with doMain Adaptation (OPTIMA) for few-shot learning.

code will come with an extended work.

---


#### Here are the reviews I received from EMNLP 2022, which might be a good lesson to learn.

```
============================================================================
                            META-REVIEW
============================================================================ 

Comments: By reading the paper, the reviews and the author responses, I tend to vote an acceptance. The 2.5/3 reviewers might be correct with their raised concerns, however being generic technique itself cannot be a solid reason to reject a paper.

============================================================================
                            REVIEWER #1
============================================================================

What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
---------------------------------------------------------------------------
The paper studies the domain adaptation of the prompt tuning paradigm for pretrained language models (PLM). It is the first work which studies the domain adaptation for the PLM prompt tuning. It proposes two techniques, namely decision boundary smoothing (which is achieved by using an adversarial pertubation method) and domain discriminator (which could make sure the quality of the adaptation), and the whole algorithm works under the adversarial learning framework. The experiment shows that the proposed techniques achieve excellent transferibility and sample-efficiency compared to strong competitors.
---------------------------------------------------------------------------


Reasons to accept
---------------------------------------------------------------------------
1. The paper studies an important and hot topic. 
2. The result is of significance to the deployment of PLM in practice and guides the usages of its prompt tuning. 
3. The paper is technically sound and is a solid work.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                         Reproducibility: 4
                        Ethical Concerns: No
     Overall Recommendation - Long Paper: 4.5


============================================================================
                            REVIEWER #2
============================================================================

What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
---------------------------------------------------------------------------
This work studies prompt tuning and explores domain adaptation for prompt tuning. The method regularizes the decision boundary to be smooth around regions where source and target data distributions are similar. The results are validated in paraphrase detection and natural language inference.
---------------------------------------------------------------------------


Reasons to accept
---------------------------------------------------------------------------
- This paper is well written and easy to follow. The models and the experimental results are well presented. 
- The improvement over the existing methods is clear. 
- The proposed OPTIMA Algorithm is technically sound to me.
---------------------------------------------------------------------------


Reasons to reject
---------------------------------------------------------------------------
- This work seems to be a simple extension to the self-supervised learning methods. The new contributions seem minor.
- The framework can also be applied for generic domain adaption, but not limited prompt-based domain adaptation. 
- In the conclusion section, the authors should discuss some possible future work directions in the revision.
- Discussions about limitations can be included in the paper. 
- From the results in Table 4, the performance improvement is minor.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                         Reproducibility: 3
                        Ethical Concerns: No
     Overall Recommendation - Long Paper: 2.5


============================================================================
                            REVIEWER #3
============================================================================

What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
---------------------------------------------------------------------------
The paper explores the problem of unsupervised domain adaptation (UDA), where labeled data is given for the source domain, while unlabeled data is given for the target domain. The paper proposes a prompt tuning algorithm applied with a variant of the Virtual Adversarial Training algorithm (where an adversarial loss term is taken into consideration during the computation of perturbation). The paper is primarily interested in exploring the integration of adversarial training with prompt tuning in the context of UDA. 

Strengths:
- The paper is well-written and the problem is well-defined.
- The paper explores unsupervised domain adaptation in NLP, which has been relatively less covered in the past.
- Experiments are well-designed and the results show that the proposed method outperforms all baselines.

Weaknesses:
- The paper makes very little mention of prior work in unsupervised domain adaptation in NLP.
- The proposed approach is not prompt tuning-specific. The relationship between the proposed variation of the VAT algorithm and prompt tuning is not fully explored.
- The premise of using prompt tuning to solve UDA is unconvincing. There are other adaptation approaches for language models not covered in the paper, such as adapter, LoRA, Ladder Side-Tuning, HyperPrompt, linear probing, and so on. Other approaches are also well-known for parameter-efficient adaptation of the language models while maintaining good generalizability. The proposed adversarial training is not limited to prompt tuning and can be easily extended to other approaches.
---------------------------------------------------------------------------


Reasons to accept
---------------------------------------------------------------------------
Some may find the findings of using prompt tuning to solve UDA interesting.
---------------------------------------------------------------------------


Reasons to reject
---------------------------------------------------------------------------
The novelty of the proposed approach is unclear. The proposed algorithm is a minor variation of the VAT algorithm, and it is not prompt tuning-specific either. Furthermore, there is little motivation to use prompt tuning as the only approach to adapting language models for domain adaptation.
---------------------------------------------------------------------------


Questions for the Author(s)
---------------------------------------------------------------------------
Has there been any similar work that uses the adversarial training approach for UDA x NLP?
---------------------------------------------------------------------------


Missing References
---------------------------------------------------------------------------
Ramponi, Alan, and Barbara Plank. "Neural Unsupervised Domain Adaptation in NLP—A Survey." Proceedings of the 28th International Conference on Computational Linguistics. 2020.
---------------------------------------------------------------------------


---------------------------------------------------------------------------
Reviewer's Scores
---------------------------------------------------------------------------
                         Reproducibility: 4
                        Ethical Concerns: No
     Overall Recommendation - Long Paper: 3
```
