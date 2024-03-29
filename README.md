# CIFAR-10 with DeepSpeed and DALI

The goal of this project is to train a transformer model to solve the CIFAR-10 task using Microsoft DeepSpeed and Nvidia DALI for faster training.

I would like to use this as a scalable training script for exploring modifications to transformer architecture with faster iteration speed.

As compared to the `vision-transformers-cifar10` repo: This repo uses about half as many epochs, each running about 2x faster, so about 4x faster overall, to train the same model with 3% higher accuracy scores.

By bringing in the latest `x-transformers` model, we can use the same number of parameters as ViT-tiny to achieve 3% higher performance again: 89%.  By using 4x more parameters we can get over 90% without pre-training.

Experimental runs (git hash, results, all parameters) are recorded to a results text file so that you can run a batch of experiments and then use the results text file to reproduce the results and generate graphs.


## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

Make sure your Nvidia drivers are installed properly by running `nvidia-smi`.  The best way is to install the latest CUDA toolkit from https://developer.nvidia.com/cuda-downloads which includes the latest drivers.

I'm using CUDA 12.4.  If you have issues it might make sense to upgrade.

Make sure you can use `nvcc` and `nvidia-smi` and the CUDA version matches on both:

```bash
nvidia-smi
nvcc --version
```

You need `zip`: `sudo apt install zip`

Setup the software from this repo:

```bash
git clone https://github.com/catid/cifar10deepspeed
cd cifar10deepspeed

conda create -n train python=3.10 -y && conda activate train

pip install -U -r requirements.txt

# Work-around for https://github.com/Dao-AILab/flash-attention/issues/867
# Sadly this takes a while...
pip install --no-build-isolation flash-attn==2.3.0
```

Setup dataset:

```bash
# Extract the dataset and produce labels
gdown 'https://drive.google.com/uc?id=1r0Rb7dfex7g3ovRacIvoEbTkXkrgmboe'

unzip cifar10.zip
python prepare_dataset.py
```


## Train

```bash
conda activate train
./launch_local_train.sh --reset
```

The training process will stop after 50 epochs without any improvement in validation loss, which ends up being about 175 epochs, which is about 6 minutes with my hardware.

If training is interrupted it will resume from the last checkpoint.  You can pass `--reset` to clear the last checkpoint and train from scratch, which you should do when starting to train a new model.

The training script will save the best model to disk during training as a `cifar10.pth` model file.  You can copy this file around to save it for your records to reproduce results.

In another window you can run tensorboard and then navigate to http://gpu1.lan:6006/ to watch the progress of training:

```bash
conda activate train
./tensorboard.sh
```

Tensorboard has a Dark Mode, refresh button, and auto-refresh in the upper right.  There's also a pin button for the ValAcc graph to put it on top.  If you restart training, you will need to restart the tensorboard server.


## Batch experiments

To run a hyperparameter sweep for the baseline model:

```bash
conda activate train
./scripts/exp_baseline_mlp_dim.sh
```

This will produce results in `results.txt` you can graph by parsing the text file.  There is a `results_parser.py` that shows how to parse the `results.txt` file.

You can watch the (very slow) progress in another tmux window:

```bash
# In case the file is not created yet
touch results.txt

tail -f results.txt
```


## Weights & Biases

To enable W&B, run `wandb login` before training to set up your API key.  Pass `--wandb` to the training script.  You will also need to specify a `--name` for the experiment as well when using this option, so that it shows up properly in W&B.


## Graph results

You can combine all the `results.txt` files from the machines into one big text file and then graph the data all together:

```bash
python graph.py --results combined_results.txt --name vary_mlp_dim --series mlp_dim
```

This is a plot of the performance of the transformer model if you vary the MLP hidden dimension, showing error bars for variance across 3 training sessions:

![Accuracy Variation](docs/graph_acc_vary_mlp_dim_mlp_dim.png)

Note that there's a sweet spot at 256, after which it starts getting too big and it's memorizing the training set instead of generalizing.

It's also interesting that by dropping mlp_dim to just 16 from 256, it only loses 6% accuracy on the test set.


## Stand-alone evaluation

The training script inserts a `cifar10deepspeed` key into the PyTorch `.pth` model file dictionary containing the architecture, architecture parameters, and fp16 mode so that it can reload the correct model from the file.  So you should not need to specify the model during evaluation.

```bash
conda activate train
python evaluate.py

(train) ➜  cifar10deepspeed git:(main) ✗ python evaluate.py
2023-12-10 06:33:54,631 [INFO] Loaded model with arch=x_transformers config=patch_size=4,dim=512,depth=6,heads=8 fp16=True model size = 31571562 weights
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 7490.96it/s]
2023-12-10 06:33:56,008 [INFO] Test loss = 0.509036669921875
2023-12-10 06:33:56,008 [INFO] Test accuracy: 88.6%
```

This will print the accuracy % on the test set.  As a sanity check it also reports the test loss of the model, which should match the epoch where it was sampled from during training.

You can actually run the evaluation script while the training script is running if you are impatient.

Here's another run with a 6x smaller model that performs better:

```bash
(train) ➜  cifar10deepspeed git:(main) ✗ python evaluate.py
2023-12-10 19:34:59,060 [INFO] Loaded model with arch=x_transformers config=patch_size=4,dim=256,depth=4,heads=6 fp16=True model size = 5812842 weights
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 9448.25it/s]
2023-12-10 19:35:00,150 [INFO] Test loss = 0.46669970703125
2023-12-10 19:35:00,150 [INFO] Test accuracy: 88.95%
```

## Set up training cluster

If using just a single computer for training you can skip this section.

I found that for a 3x 3090 GPU setup with about ~2 Gbps Ethernet between them, it's actually faster to just use one machine for training rather than a cluster.  I haven't tested on my other machines yet, so not sure using a training cluster is ever useful for this problem.

Checkout the code at the same path on each computer.

Edit the `hostfile` to specify the list of nodes in the training cluster.  They must be accessible over SSH without a password: Use `ssh-copy-id myname@hostname` to set this up.

The dataset must be at the same path on each computer participating in the training cluster.  I'd recommend just repeating these preparation steps on each computer in the cluster rather than using a network drive, since the dataset is small.

```bash
./launch_distributed_train.sh --reset
```
