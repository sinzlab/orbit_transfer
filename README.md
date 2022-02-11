# Can Functional Transfer Methods Capture Simple Inductive Biases? -- Code

This repository contains the implementation of all methods that we discuss in our publication *Can Functional Transfer Methods Capture Simple Inductive Biases?*

## :gear: Usage:

The experiments from that paper that use the methods from this repository can be found in [orbit_transfer_recipes](https://github.com/sinzlab/orbit_transfer_recipes).

The experiments further require installation of:
- [nntransfer](https://github.com/sinzlab/nntransfer)
- [nnfabrik](https://github.com/sinzlab/nnfabrik)
- [neuralpredictors](https://github.com/sinzlab/neuralpredictors)
- [pytorch_warmup](https://github.com/ArneNx/pytorch_warmup)

## :star: Features:

### Models:

`models/cnn.py`: Modular architecture of a plain CNN model 

`models/group_cnn.py`: Implementation of a group-equivariant CNN model

`models/group_equivariant_layers.py`: Layers used in the G-CNN model

`models/learned_equiv.py`: Implementation of the Orbit model

`models/mlp.py`: Modular architecture of a plain MLP model

`models/vit.py`: Simplified implementation of a small VIT model 

### Transfer methods:

We implemented all transfer methods that we discussed in the paper as main-loop-modules (see [nntransfer](https://github.com/sinzlab/nntransfer)). 
This includes: attention transfer, knowledge distillation, representational distance learning and orbit transfer.

### Simple MNIST-1D experiments:

A less modular and flexible implementation of all transfer methods and corresponding training can be found in `trainer/simple_train.py` and `trainer/forward_methods.py`. 
Fitting models are given in `models/mnist_1d.py`.


## :bug: Report bugs 

In case you find a bug, please create an issue or contact any of the contributors.
