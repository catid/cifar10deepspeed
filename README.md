# CIFAR100 with DeepSpeed and DALI

The goal of this project is to train a transformer architecture on the CIFAR100 task using Microsoft DeepSpeed and Nvidia DALI.

I would like to use this as a scalable training script for exploring modifications to transformer architecture with faster iteration speed.

## Setup

```bash
git clone https://github.com/catid/cifar100deepspeed
cd cifar100deepspeed

conda create -n train python=3.10
conda activate train

# Update this from https://pytorch.org/get-started/locally/
pip3 install --upgrade torch torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu118

# Update this from https://github.com/NVIDIA/DALI#installing-dali
pip install --upgrade nvidia-dali-cuda110 --extra-index-url https://developer.download.nvidia.com/compute/redist

pip install -U -r requirements.txt

# Extract the dataset and produce labels
gdown 'https://drive.google.com/uc?id=1MYQyvXFoxakvQBWnGd1NF5iUO23VATj1'
unzip cifar100.zip
python prepare_dataset.py
```


## Train

```bash
./launch_local_train.sh
```


## Evaluate

```bash
python evaluate.py
```


## Set up training cluster

If using just a single computer for training you can skip this section.

Edit the `hostfile` to specify the list of nodes in the training cluster.  They must be accessible over SSH without a password: Use `ssh-copy-id myname@hostname` to set this up.

The dataset must be at the same path on each computer participating in the training cluster.  I'd recommend just repeating these preparation steps on each computer in the cluster rather than using a network drive, since the dataset is small.

```bash
./launch_distributed_train.sh
```


## Dataset details

I used this project to download CIFAR100 as image files: https://github.com/knjcode/cifar2png

The image files are hosted on my Google Drive here: https://drive.google.com/file/d/1MYQyvXFoxakvQBWnGd1NF5iUO23VATj1/view?usp=sharing


## Ideas to try:

* Standard transformer with FFN mlp_size=256
* Pairs of FFN layers instead of one per transformer
* FFF with depth=7 (256 parameters)
* Fan-out=4 using softmax instead of 2 with depth=3 (256 parameters)
* Adjust training rate of root to be smaller than leafs
* Full dense multiply and select output on CPU for training speed
* Mix or chain CSC-I
* Add 8 dense rows
