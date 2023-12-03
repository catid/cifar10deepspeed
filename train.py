import os
import shutil
import time
import random

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch._dynamo as dynamo

import deepspeed
from deepspeed import comm
from deepspeed import log_dist

import argparse

from model.vit_small import ViT

from dataloader import CifarDataLoader

from deepspeed.runtime.config import DeepSpeedConfig

from torch.utils.tensorboard import SummaryWriter



def select_model(args):
    return ViT()


def log_0(msg):
    log_dist(msg, ranks=[0])

def log_all(msg):
    log_dist(msg, ranks=[-1])


# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


def is_main_process():
    return comm.get_rank() == 0

def ref_forward_and_loss(criterion, data, labels, model_engine):
    # DeepSpeed: forward + backward + optimize
    outputs = model_engine(data)
    return criterion.forward(outputs, labels)

def train_one_epoch(opt_forward_and_loss, criterion, train_loader, model_engine):
    train_loss = 0.0

    model_engine.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (label, target, data) in enumerate(train_loader):
            data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)

            loss = opt_forward_and_loss(criterion, data, target, model_engine)

            model_engine.backward(loss)
            model_engine.step()

            train_loss += loss.item()

    return train_loss


def validation_one_epoch(opt_forward_and_loss, criterion, val_loader, model_engine):
    val_loss = 0.0

    model_engine.eval()

    with torch.set_grad_enabled(False):
        for batch_idx, (labels, data) in enumerate(val_loader):
            labels, data = labels.to(model_engine.local_rank), data.to(model_engine.local_rank)

            loss = opt_forward_and_loss(criterion, data, labels, model_engine)

            val_loss += loss.item()

            if batch_idx == 0:
                test_images = data[:2]
                output_labels = model_engine(test_images)
                examples = (test_images, labels[:2], output_labels[:2])

    return val_loss, examples

def dict_compare(dict1, dict2):
    # Check if the dictionaries have the same length
    if len(dict1) != len(dict2):
        return False

    # Check if the dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Check if the dictionaries have the same values for each key
    for key in dict1:
        if dict1[key] != dict2[key]:
            return False

    return True

def delete_folder_contents(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def get_absolute_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))

    return absolute_path

def main(args):
    params = {}
    params['learning_rate'] = 0.001

    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)

    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    # Model and optimizer
    model = select_model(args)

    torch._dynamo.config.verbose = False
    torch._dynamo.config.suppress_errors = True

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model.parameters())

    log_dir = f"{args.log_dir}/cifar100"
    os.makedirs(log_dir, exist_ok=True)
    if args.reset:
        log_0("Resetting training - deleting Tensorboard directory")
        delete_folder_contents(log_dir)
    tensorboard = SummaryWriter(log_dir=log_dir)

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}")

    num_loader_threads = os.cpu_count()//2
    seed = 0
    crop_w = 32
    crop_h = 32

    dataset_dir = get_absolute_path(args.dataset_dir)
    log_all(f"Loading dataset from: {dataset_dir}")

    train_loader = CifarDataLoader(
        batch_size=data_loader_batch_size,
        device_id=rank,
        num_threads=num_loader_threads,
        seed=seed,
        file_list=os.path.join(dataset_dir, "training_file_list.txt"),
        mode='training',
        crop_w=crop_w,
        crop_h=crop_h,
        shard_id=shard_id,
        num_shards=num_gpus)

    val_loader = CifarDataLoader(
        batch_size=data_loader_batch_size,
        device_id=rank,
        num_threads=num_loader_threads,
        seed=seed,
        file_list=os.path.join(dataset_dir, "validation_file_list.txt"),
        mode='validation',
        crop_w=crop_w,
        crop_h=crop_h,
        shard_id=shard_id,
        num_shards=num_gpus)

    # Loss functions

    criterion = nn.CrossEntropyLoss()
    criterion.cuda(rank)

    forward_and_loss = ref_forward_and_loss
    forward_and_loss = dynamo.optimize("eager")(forward_and_loss)

    # Initialize training

    best_val_loss = float("inf")
    avg_val_loss = float("inf")
    start_epoch = 0
    epochs_without_improvement = 0

    if args.reset:
        log_0("Resetting training - deleting output directory")
        if rank == 0:
            delete_folder_contents(args.output_dir)
    else:
        _, client_state = model_engine.load_checkpoint(load_dir=args.output_dir)
        if client_state is not None:
            start_epoch = client_state['epoch'] + 1
            if client_state['crop_w'] != crop_w or client_state['crop_h'] != crop_h or client_state['train_version'] != 1:
                log_all(f"Model checkpoint is incompatible with current training parameters. Please reset the training by deleting the output directory {args.output_dir} or running with --reset")
                exit(1)
            if not dict_compare(params, client_state['model_params']):
                log_all(f"Model params is incompatible with current training parameters. Please reset the training by deleting the output directory {args.output_dir} or running with --reset")
                exit(1)
            avg_val_loss = client_state['avg_val_loss']
            best_val_loss = avg_val_loss
            log_all(f"Loaded checkpoint at epoch {client_state['epoch']}")
        else:
            log_all("No checkpoint found - Starting training from scratch")

    # Training/validation loop

    for epoch in range(start_epoch, args.max_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(forward_and_loss, criterion, train_loader, model_engine)

        val_loss, examples = validation_one_epoch(forward_and_loss, criterion, val_loader, model_engine)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Sync variables between machines
        sum_train_loss = torch.tensor(train_loss).cuda(rank)
        sum_val_loss = torch.tensor(val_loss).cuda(rank)
        comm.all_reduce(tensor=sum_train_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_val_loss, op=comm.ReduceOp.SUM)

        total_train_items = len(train_loader) * num_gpus
        total_val_items = len(val_loader) * num_gpus
        comm.barrier()
        avg_train_loss = sum_train_loss.item() / total_train_items
        avg_val_loss = sum_val_loss.item() / total_val_items

        if is_main_process():
            events = [
                ("AvgTrainLoss", avg_train_loss, model_engine.global_samples),
                ("AvgValLoss", avg_val_loss, model_engine.global_samples)
            ]
            for event in events:
                tensorboard.add_scalar(*event)

            # Note: Tensorboard needs NCHW uint8
            input_images = examples[0]
            input_labels = examples[1]
            output_labels = examples[2]

            tensorboard.add_images('input', input_images, epoch)
            tensorboard.add_images('labels', input_labels, epoch)
            tensorboard.add_images('output', output_labels, epoch)

            log_0(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Check if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            log_0(f'New best validation loss: {best_val_loss:.4f}')

            client_state = {
                'train_version': 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'epoch': epoch,
                'crop_w': crop_w,
                'crop_h': crop_h,
                'fp16': fp16,
                'model_params': params
            }
            model_engine.save_checkpoint(save_dir=args.output_dir, client_state=client_state)
            log_all(f'Saved new best checkpoint')
        else:
            epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= args.patience:
                log_0(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss}")
                break

    if is_main_process():
        log_0(f'Training complete.  Final validation loss: {avg_val_loss}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset-dir", type=str, default=str("cifar100"), help="Path to the dataset directory (default: ./cifar100/)")
    parser.add_argument("--max-epochs", type=int, default=1000000, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=100, help="Patience for validation loss not decreasing before early stopping")
    parser.add_argument("--output-dir", type=str, default="output_model", help="Path to the output trained model")
    parser.add_argument("--log-dir", type=str, default="tb_logs", help="Path to the Tensorboard logs")
    parser.add_argument("--reset", action="store_true", help="Reset training from scratch")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
