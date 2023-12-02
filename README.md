# CIFAR100 with DeepSpeed and DALI

The goal of this project is to train a transformer architecture on the CIFAR100 task using Microsoft DeepSpeed and Nvidia DALI.

I would like to use this as a scalable training script for exploring modifications to transformer architecture with faster iteration speed.

## Ideas to try:

* Standard transformer with FFN mlp_size=256
* Pairs of FFN layers instead of one per transformer
* FFF with depth=7 (256 parameters)
* Fan-out=4 using softmax instead of 2 with depth=3 (256 parameters)
* Adjust training rate of root to be smaller than leafs
* Full dense multiply and select output on CPU for training speed
* Mix or chain CSC-I
* Add 8 dense rows
