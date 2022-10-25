# Pytorch-cifar100

practice on cifar100 using pytorch

## Usage

### train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
$ CUDA_DEVICES=X,Y python train.py -net resnet18 -gpu
```

sometimes, you might want to use warmup training by set ```-warm``` to 1 or 2, to prevent network
diverge during early training phase.

The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
```
Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'(default in checkpoint folder).


### 5. test the model
Test the model using test.py
```bash
$ CUDA_VISIBLE_DEVICES=X,Y python test.py -net resnet18 -weights path_to_resnet18_weights_file -gpu
```

## Implementated NetWork

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- googlenet [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
- shufflenetv2 [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv2 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- residual attention network [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)
- senet [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- squeezenet [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4)
- nasnet [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012v4)
- wide residual network[Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- stochastic depth networks[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)


I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.1 divide by 5 at 60th, 120th, 160th epochs, train for 200
epochs with batchsize 128 and weight decay 5e-4, Nesterov momentum of 0.9. You could also use the hyperparameters from paper [Regularizing Neural Networks by Penalizing Confident Output Distributions](https://arxiv.org/abs/1701.06548v1) and [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896v2), which is initial lr = 0.1, lr divied by 10 at 150th and 225th epochs, and training for 300 epochs with batchsize 128, this is more commonly used. You could decrese the batchsize to 64 or whatever suits you, if you dont have enough gpu memory.