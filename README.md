# Convolution Pruning
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
