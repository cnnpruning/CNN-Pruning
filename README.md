# CNN Pruning
This repo contains implementations of kernel-level pruning and run-time pruning of CNNs.

## Convolution Kernel Pruning/Kernel-Level Pruning

This contains a pytorch implementation of the ImageNet experiments of kernel-level pruning.

### Implementation
We prune only the kernel in the convolutional layer. We use the mask implementation, where during pruning, we set the weights that are pruned to be 0. During training, we make sure that we don't update those pruned parameters.

### Baseline
We get the base model of VGG-16 and ResNet-50 from Pytorch [Model Zoo](https://pytorch.org/docs/stable/torchvision/models.html).

### Prune
```
python kernel_prune.py --arch vgg16_bn --pretrained --percent 0.3 --save [PATH TO SAVE RESULTS] [IMAGENET]
python kernel_prune.py --arch resnet50 --pretrained --percent 0.3 --save [PATH TO SAVE RESULTS] [IMAGENET]
```

### Finetune
```
python main_finetune.py --arch vgg16_bn --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
python main_finetune.py --arch resnet50 --resume [PATH TO THE PRUNED MODEL] --save [PATH TO SAVE RESULTS] [IMAGENET]
```

### References
* Pytorch documentation https://pytorch.org/docs/stable/index.html
* Rethinking-network-pruning Project https://github.com/Eric-mingjie/rethinking-network-pruning/

## Convolution Run-time Pruning

Feature Boosting and Suppression (FBS) is a method that exploits run-time dynamic information flow in CNNs to dynamically prune channel-wise parameters.

Intuitively, we can imagine that the flow of information of each output channel can be amplified or restricted under the control of a “valve”. 
This allows salient information to flow freely while we stop all information from unimportant channels and skip their computation. 
Unlike static pruning, the valves use features from the previous layer to predict the saliency of output channels. 
FBS introduces tiny auxiliary connections to existing convolutional layers. 
The minimal overhead added to the existing model is thus negligible when compared to the potential speed up provided by the dynamic sparsity.

### Running with FBS in Mayo

Prerequisites

Before setting up Mayo, you will need to have [Git][git], [Git-LFS][git-lfs], [Python 3.6.5 or above][python3] and [TensorFlow 1.11 or above][tensorflow] installed.

Setting up Mayo
```bash
$ cd mayo
$ pip3 install -r requirements.txt
```
Training A FBS

We suggest first
fine-tune a pretrained model with 100% density,
and gradually decrease the density
and fine-tune the resulting model
for minimal accuracy drops.
The command is as follows:
```bash
$ ./my \
models/gate/{model}.yaml \
datasets/{dataset}.yaml \
trainers/mobilenet.yaml \
_gate.density={density} \
system.checkpoint.load={pretrained_model} \
train.learning_rate._initial={initial_lr} \ # this adjusts the initial learning rate
reset-num-epochs train
```
Please see [`mayo/docs/fbs.md`] for futher instructions.
