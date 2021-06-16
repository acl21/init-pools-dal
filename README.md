# On Initial Pools for Deep Active Learning

This is the official code implementation of our paper, _On Initial Pools for Deep Active Learning_, accepted at Pre-registration Workshop at NeurIPS 2020. The final results paper is accepted to be included in the Proceedings of Machine Learning Research (PMLR), a sister publication to the Journal of Machine Learning Research (JMLR). You can find the paper [here](https://arxiv.org/abs/2011.14696). 


## Abstract
Active Learning (AL) techniques aim to minimize the training data required to train a model for a given task. Pool-based AL techniques start with a small initial labeled pool and then iteratively pick batches of the most informative samples for labeling. Generally, the initial pool is sampled randomly and labeled to seed the AL iterations. While recent studies have focused on evaluating the robustness of various query functions in AL, little to no attention has been given to the design of the initial labeled pool for deep active learning. Given the recent successes of learning representations in self-supervised/unsupervised ways, we study if an intelligently sampled initial labeled pool can improve deep AL performance. We investigate the effect of intelligently sampled initial labeled pools, including the use of self-supervised and unsupervised strategies, on deep AL methods. The setup, hypotheses, methodology, and implementation details were evaluated by peer review before experiments were conducted. Experimental results could not conclusively prove that intelligently sampled initial pools are better for AL than random initial pools in the long run, although a Variational Autoencoder-based initial pool sampling strategy showed interesting trends that merit deeper investigation.


## Conclusion
In this paper, we proposed two kinds of strategies -- self-supervision based and clustering based -- for intelligently sampling initial pools before the use of active learning (AL) methods for deep neural network models. Our motivation was to study if there exist good initial pools that contribute to better model generalization and better deep AL performance in the long run. Our proposed methods and experiments conducted on four image classification datasets couldn't conclusively prove the existence of such good initial pools. However, a surprising outcome of this study was how initial pools sampled with a simple VAE task contributed to improved AL performance, better than more complex SimCLR and SCAN tasks. Even though VAE-based initial pools worked better than random initial pools only on one dataset (CIFAR-100), ablation studies on low budget CIFAR-10 settings as well as on Long-Tail CIFAR-10 point towards potential in VAE-sampled initial pools. Are images that are hard to reconstruct for VAEs good for generalization? Can better generative models like GANs do better than VAEs? We leave this for future work. While our methods and findings could not conclusively prove our hypothesis that AL methods can benefit from more intelligently chosen initial pools, we are optimistic about the potential this research direction holds. 


## Citing this Repository
If you find this repo helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing us:
```
@inproceedings{Chandra2021OnIP,
author = {Chandra, Akshay L and Desai, Sai Vikas and Devaguptapu, Chaitanya and Balasubramanian, Vineeth N},
title = {On Initial Pools for Deep Active Learning},
publisher = {Pre-registration Workshop, NeurIPS},
year = {2020},
booktitle = {Proceedings of Machine Learning Research (PMLR)},
series = {Volume 148},
numpages = {19}
url = {https://arxiv.org/abs/2011.14696},
}
```

### About Pre-registration Workshop
Pre-registration separates the generation and confirmation of hypotheses. Before spending a lot of time on running confirmatory experiments, this workshop allows researchers to get their ideas peer-reviewed first through paper proposals. Only after the paper is accepted, you run the experiments and report your results.  

What does science get from this?
* A healthy mix of positive and negative results
* Reasonable ideas that don’t work still get published, avoiding wasteful replications
* Papers are evaluated on the basis of scientific interest, not whether they achieve the best results

Checkout their workshops here:
* [NeurIPS 2020](https://preregister.science/neurips2020.html)
* [ICCV 2019](http://preregister.vision/)
* [CVPR 2017](http://negative.vision/)

_Note: The repository is a clone of the toolkit _Deep Active Learning Toolkit for Image Classification in PyTorch_, available [here](https://github.com/acl21/deep-active-learning-pytorch). This codebase only supports single-machine single-gpu training. If you have any questions about the code, please email at research [at] akshaychandra [dot] com._ 

## Acknowledgement
We thank DST, Govt of India, for partly supporting this work through the IMPRINT program (IMP/2019/000250). We also thank the members of Lab1055, IIT Hyderabad for engaging and fruitful discussions. Last but not the least, we thank all our anonymous reviewers for their insightful comments and suggestions, which helped improve the quality of this work.