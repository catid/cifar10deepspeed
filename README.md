# CIFAR-10 with DeepSpeed and DALI

The goal of this project is to train a transformer architecture on the CIFAR-10 task using Microsoft DeepSpeed and Nvidia DALI for faster training.

I would like to use this as a scalable training script for exploring modifications to transformer architecture with faster iteration speed.

As compared to the `vision-transformers-cifar10` repo: This repo uses about half as many epochs, each running about 2x faster, so about 4x faster overall, to train the same model with 3% higher accuracy scores.

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

Make sure your Nvidia drivers are installed properly by running `nvidia-smi`.  If not, you may need to run something like `sudo apt install nvidia-driver-535-server` or newer.

Install CUDA toolkit:

```bash
# Here's how to do it on Ubuntu:
sudo apt install nvidia-cuda-toolkit
```

Make sure you can use `nvcc`:

```bash
nvcc --version
```

Setup the software from this repo:

```bash
git clone https://github.com/catid/cifar10deepspeed
cd cifar10deepspeed

conda create -n train python=3.10
conda activate train

# Update this from https://pytorch.org/get-started/locally/
pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu118

# Update this from https://github.com/NVIDIA/DALI#installing-dali
pip install --upgrade nvidia-dali-cuda110 --extra-index-url https://developer.download.nvidia.com/compute/redist

pip install -U -r requirements.txt

# Extract the dataset and produce labels
gdown 'https://drive.google.com/uc?id=1r0Rb7dfex7g3ovRacIvoEbTkXkrgmboe'

# Install the zip/unzip CLI tools
sudo apt install zip

unzip cifar10.zip
python prepare_dataset.py
```


## Train

```bash
conda activate train
./launch_local_train.sh
```

The training process will stop after 50 epochs without any improvement in validation loss, which ends up being about 175 epochs, which is about 6 minutes with my hardware.

If training is interrupted it will resume from the last checkpoint.  You can pass `--reset` to clear the last checkpoint and train from scratch, which you should do when changing models.

The training script will save the best model to disk during training as a `cifar10.pth` model file.  You can copy this file around to save it for your records to reproduce results.

In another window you can run tensorboard and then navigate to http://gpu1.lan:6006/ to watch the progress of training:

```bash
conda activate train
./tensorboard.sh
```

Tensorboard has a Dark Mode, refresh button, and auto-refresh in the upper right.  There's also a pin button for the ValAcc graph to put it on top.  If you restart training, you will need to restart the tensorboard server.


## Evaluate

```bash
conda activate train
python evaluate.py

(train) ➜  cifar10deepspeed git:(main) ✗ python evaluate.py
2023-12-03 09:23:30,619 [INFO] Loading as FP16: True
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 6074.96it/s]
2023-12-03 09:23:32,723 [INFO] Test loss = 0.51888203125
2023-12-03 09:23:32,723 [INFO] Test accuracy: 84.62%
```

This will print the accuracy % on the test set.  As a sanity check it also reports the test loss of the model, which should match the epoch where it was sampled from during training.

You can actually run the evaluation script while the training script is running if you are impatient.


## Set up training cluster

If using just a single computer for training you can skip this section.

I found that for a 3x 3090 GPU setup with about ~2 Gbps Ethernet between them, it's actually faster to just use one machine for training rather than a cluster.  I haven't tested on my other machines yet, so not sure using a training cluster is ever useful for this problem.

Checkout the code at the same path on each computer.

Edit the `hostfile` to specify the list of nodes in the training cluster.  They must be accessible over SSH without a password: Use `ssh-copy-id myname@hostname` to set this up.

The dataset must be at the same path on each computer participating in the training cluster.  I'd recommend just repeating these preparation steps on each computer in the cluster rather than using a network drive, since the dataset is small.

```bash
./launch_distributed_train.sh
```


## Ideas to try:

* Standard transformer with FFN mlp_size=256
* Pairs of FFN layers instead of one per transformer
* FFF with depth=7 (256 parameters)
* Fan-out=4 using softmax instead of 2 with depth=3 (256 parameters)
* Adjust training rate of root to be smaller than leafs
* Full dense multiply and select output on CPU for training speed
* Mix or chain CSC-I
* Add 8 dense rows
