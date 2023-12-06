import os
import shutil
import time
import random

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

import deepspeed
from deepspeed import comm
from deepspeed import log_dist

import argparse

from dataloader import CifarDataLoader

from deepspeed.runtime.config import DeepSpeedConfig

from torch.utils.tensorboard import SummaryWriter

from models.model_loader import select_model, params_to_string

# Prettify printing tensors when debugging
import lovely_tensors as lt
lt.monkey_patch()


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
    _, predicted = outputs.max(1)
    return criterion(outputs, labels), predicted

def train_one_epoch(opt_forward_and_loss, criterion, train_loader, model_engine, image_dtype):
    train_loss = 0.0

    model_engine.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (labels, images) in enumerate(train_loader):
            images = (images / 255.0).to(image_dtype) # Normalize NCHW uint8 input
            labels = labels.squeeze().to(torch.long)

            #log_all(f"train_one_epoch: batch_idx = {batch_idx} labels[:4] = {labels[:4]}")

            labels, images = labels.to(model_engine.local_rank), images.to(model_engine.local_rank)

            loss, _ = opt_forward_and_loss(criterion, images, labels, model_engine)

            model_engine.backward(loss)
            model_engine.step()

            train_loss += loss.item()

    return train_loss

def validation_one_epoch(opt_forward_and_loss, criterion, val_loader, model_engine, image_dtype):
    val_loss = 0.0
    correct = 0
    total = 0

    model_engine.eval()

    with torch.set_grad_enabled(False):
        for batch_idx, (labels, images) in enumerate(val_loader):
            images = (images / 255.0).to(image_dtype) # Normalize NCHW uint8 input
            labels = labels.squeeze().to(torch.long)

            #log_all(f"validation_one_epoch: batch_idx = {batch_idx} labels[:4] = {labels[:4]}")

            labels, images = labels.to(model_engine.local_rank), images.to(model_engine.local_rank)

            loss, predicted = opt_forward_and_loss(criterion, images, labels, model_engine)

            val_loss += loss.item()

            correct += torch.eq(predicted, labels).sum().item()
            total += predicted.size(0)

            if batch_idx == 0:
                test_images = images[:2]
                output_labels = model_engine(test_images)
                examples = (test_images, labels[:2], output_labels[:2])

    return val_loss, correct, total, examples

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
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def get_absolute_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))

    return absolute_path

import subprocess
from datetime import datetime

def record_experiment(args, params, best_train_loss, best_val_loss, best_val_acc,
                      end_epoch, dt, num_params):
    git_hash = "Git hash unavailable"
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        pass

    datetime_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = {}
    data["name"] = args.name
    data["notes"] = args.notes
    data["arch"] = args.arch
    data["params"] = params_to_string(params) # Note that args.params may not include any default values
    data["best_val_acc"] = best_val_acc
    data["best_train_loss"] = best_train_loss
    data["best_val_loss"] = best_val_loss
    data["end_epoch"] = end_epoch
    data["train_seconds"] = dt
    data["num_params"] = num_params
    data["git_hash"] = git_hash
    data["timestamp"] = datetime_string
    data["seed"] = args.seed
    data["optimizer"] = "AdamW"
    data["lr"] = args.lr
    data["weight_decay"] = args.weight_decay
    data["max_epochs"] = args.max_epochs

    record_lines = [f"\t{key.rjust(16)}: {value}" for key, value in data.items() if value]
    text = "Experiment:\n" + "\n".join(record_lines) + "\n\n"

    with open(args.result_file, "a") as file:
        file.write(text)

def synchronize_seed(args, rank, shard_id):
    if args.seed < 0:
        seed = get_true_random_32bit_positive_integer()
    else:
        seed = args.seed

    if shard_id == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    seed_tensor = seed_tensor.cuda(rank)

    comm.broadcast(tensor=seed_tensor, src=0)

    seed = int(seed_tensor.item()) + shard_id
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={shard_id}")
    return seed


def main(args):
    t0 = time.time()

    params = {}
    #params['learning_rate'] = 0.001

    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    # Model and optimizer
    params, model = select_model(args)

    log_0(f"Selected model with arch={args.arch}, params={params_to_string(params)}")

    all_parameters = list(model.parameters())
    # General parameters don't contain the special _optim key
    optimizer_params = [p for p in all_parameters if not hasattr(p, "_optim")]

    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model.parameters())

    log_dir = f"{args.log_dir}/cifar10"
    os.makedirs(log_dir, exist_ok=True)

    log_0(f"Arguments: {args}")

    comm.barrier()

    if args.reset:
        log_0("Resetting training - deleting Tensorboard directory")
        if is_main_process():
            delete_folder_contents(log_dir)

    comm.barrier()

    tensorboard = SummaryWriter(log_dir=log_dir)

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    if fp16:
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    seed = synchronize_seed(args, rank, shard_id)

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}, seed={seed}")

    num_loader_threads = os.cpu_count()//2
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

    # The time this takes to compile sadly offsets the time it saves
    # Without dynamic=True it actually takes *longer* overall
    forward_and_loss = torch.compile(forward_and_loss, dynamic=True, fullgraph=False)

    # Initialize training

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_acc = float("inf")
    avg_val_loss = float("inf")
    start_epoch = 0
    end_epoch = 0
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
        end_epoch = epoch
        start_time = time.time()

        train_loss = train_one_epoch(forward_and_loss, criterion, train_loader, model_engine, image_dtype)

        val_loss, correct, total, examples = validation_one_epoch(forward_and_loss, criterion, val_loader, model_engine, image_dtype)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Sync variables between machines
        sum_train_loss = torch.tensor(train_loss).cuda(rank)
        sum_val_loss = torch.tensor(val_loss).cuda(rank)
        sum_correct = torch.tensor(correct).cuda(rank)
        sum_total = torch.tensor(total).cuda(rank)
        comm.all_reduce(tensor=sum_train_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_val_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_correct, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_total, op=comm.ReduceOp.SUM)

        total_train_items = len(train_loader) * num_gpus
        total_val_items = len(val_loader) * num_gpus
        comm.barrier()
        avg_train_loss = sum_train_loss.item() / total_train_items
        avg_val_loss = sum_val_loss.item() / total_val_items
        val_acc = 100. * sum_correct / sum_total

        if is_main_process():
            events = [
                ("AvgTrainLoss", avg_train_loss, model_engine.global_samples),
                ("ValAcc", val_acc, model_engine.global_samples),
                ("AvgValLoss", avg_val_loss, model_engine.global_samples),
            ]
            for event in events:
                tensorboard.add_scalar(*event)

            # Note: Tensorboard needs NCHW uint8
            input_images = examples[0]
            input_labels = examples[1]
            output_labels = examples[2]

            tensorboard.add_images('input', input_images, global_step=epoch)
            tensorboard.add_scalar('label0', input_labels[0], global_step=epoch)
            tensorboard.add_scalar('label1', input_labels[1], global_step=epoch)
            tensorboard.add_histogram('output0', output_labels[0], global_step=epoch)
            tensorboard.add_histogram('output1', output_labels[1], global_step=epoch)

            log_0(f"Epoch {epoch + 1} - TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.2f}%, Time={epoch_time:.2f} sec")

        # Check if validation loss has improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_train_loss = avg_train_loss
            epochs_without_improvement = 0

            log_0(f'New best validation loss: {best_val_loss:.4f}  Validation accuracy: {best_val_acc:.2f}%')

            client_state = {
                'train_version': 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
                'crop_w': crop_w,
                'crop_h': crop_h,
                'fp16': fp16,
                'model_params': params
            }
            model_engine.save_checkpoint(save_dir=args.output_dir, client_state=client_state)

            if is_main_process():
                # Write output .pth file
                saved_state_dict = model_engine.state_dict()
                fixed_state_dict = {key.replace("module.", ""): value for key, value in saved_state_dict.items()}
                torch.save(fixed_state_dict, args.output_model)
                log_0(f"Wrote model to {args.output_model} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= args.patience:
                log_0(f"Early stopping at epoch {epoch} due to epochs_without_improvement={epochs_without_improvement}")
                break

    if is_main_process():
        log_0(f'Training complete.  Best model was written to {args.output_model}  Final best validation loss: {best_val_loss}, best validation accuracy: {best_val_acc:.2f}%')

        t1 = time.time()
        dt = t1 - t0

        num_params = sum(p.numel() for p in model.parameters())

        record_experiment(args, params, best_train_loss, best_val_loss, best_val_acc, end_epoch, dt, num_params)

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--arch", type=str, default="vit_tiny", help="Model architecture defined in models/model_loader.py")
    parser.add_argument("--params", type=str, default="", help="Model architecture parameters defined in models/model_loader.py")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset-dir", type=str, default=str("cifar10"), help="Path to the dataset directory (default: ./cifar10/)")
    parser.add_argument("--patience", type=int, default=50, help="Patience for validation loss not decreasing before early stopping")
    parser.add_argument("--output-dir", type=str, default="output_model", help="Path to the output trained model")
    parser.add_argument("--log-dir", type=str, default="tb_logs", help="Path to the Tensorboard logs")
    parser.add_argument("--reset", action="store_true", help="Reset training from scratch")
    parser.add_argument("--output-model", type=str, default="cifar10.pth", help="Output model file name")
    parser.add_argument("--result-file", type=str, default="results.txt", help="Append the experiment results to a file")
    parser.add_argument("--notes", type=str, default="", help="Provide any additional notes about the experiment to record")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for random numbers.  Set to -1 to pick a random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay for training")
    parser.add_argument("--max-epochs", type=int, default=400, help="Maximum epochs to train")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
