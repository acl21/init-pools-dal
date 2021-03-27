# On Initial Pools for Deep Active Learning

This is the official code implementation of our paper, _On Initial Pools for Deep Active Learning_, you can find the paper [here](https://arxiv.org/abs/2011.14696).  Results will be updated here soon. 

## Introduction

The repository is a clone of the toolkit _Deep Active Learning Toolkit for Image Classification in PyTorch_, available [here](https://github.com/acl21/deep-active-learning-pytorch). The codebase currently only supports single-machine single-gpu training. We will soon scale it to single-machine multi-gpu training, powered by the PyTorch distributed package.
<!-- The codebase supports efficient single-machine multi-gpu training, powered by the PyTorch distributed package, and provides implementations of standard models including [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946), and [RegNet](https://arxiv.org/abs/2003.13678). -->

## Using the toolkit

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief installation instructions and basic usage examples.

## Active Learning Methods Supported
* Uncertainty Sampling
  * Least Confidence
  * Min-Margin
  * Max-Entropy
  * Deep Bayesian Active Learning (DBAL) [1]
* Diversity Sampling 
  * Coreset (greedy) [2]
  * Variational Adversarial Active Learning (VAAL) [3]
* Query-by-Committee Sampling
  * Ensemble Variation Ratio (Ens-varR) [4]


## Datasets Supported
* [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) (Download the zip file [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip))


## Model Zoo

We provide a large set of baseline results as proof of repository's efficiency. (coming soon)


## Citing this repository

If you find this repo helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing us and the owners of the original toolkit:

```
@article{Chandra2020OnIP,
  title={On Initial Pools for Deep Active Learning},
  author={A. Chandra and Sai Vikas Desai and Chaitanya Devaguptapu and V. Balasubramanian},
  journal={Pre-registration Workshop NeurIPS},
  year={2020}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## References

[1] Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1183–1192. JMLR. org, 2017.

[2] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

[3] Sinha, Samarth et al. Variational Adversarial Active Learning. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) (2019): 5971-5980.

[4] William H. Beluch, Tim Genewein, Andreas Nürnberger, and Jan M. Köhler. The power of ensembles for active learning in image classification. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9368–9377, 2018.
