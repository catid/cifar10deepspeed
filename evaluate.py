# Pretty logging

import logging

import colorama
from colorama import Fore, Style

colorama.init()

class ColoredFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        if not record.exc_info:
            record.msg = f"{ColoredFormatter.level_colors[record.levelno]}{record.msg}{Style.RESET_ALL}"
        return super(ColoredFormatter, self).format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Model

from models.model_loader import select_model

import torch
import torch.nn as nn

def load_model(args, model_path, fp16):
    model = select_model(args)
    model.load_state_dict(torch.load(model_path))
    if fp16:
        model.half()
    model.eval()

    #for name, param in model.named_parameters():
    #    logging.info(f"Name: {name}, Type: {param.dtype}, Size: {param.size()}")

    return model

# Evaluation

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                # The first part is the file path, and the second part is the label
                file_path, label = parts[0], int(parts[1])
                data.append((file_path, label))
    return data

def evaluate(model, dataset_dir, fp16=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    validation_file = os.path.join(dataset_dir, "validation_file_list.txt")

    data = read_data_file(validation_file)

    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    for file_path, label in tqdm(data, "Evaluating"):
        input_image = Image.open(file_path).convert("RGB")

        input_image = np.array(input_image)
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        if fp16:
            input_tensor = input_tensor.to(torch.float16)
        else:
            input_tensor = input_tensor.to(torch.float32)
        input_tensor = input_tensor.to(device)

        label_tensor = torch.tensor([label]).to(torch.long).to(device)

        with torch.no_grad():
            result_tensor = model(input_tensor)
            loss = criterion(result_tensor, label_tensor)

        _, predicted = result_tensor.max(1)
        if predicted == label:
            correct += 1
        test_loss += loss.item()
        total += 1

    logger.info(f"Test loss = {test_loss/total}")
    logger.info(f"Test accuracy: {100.*correct/total}%")

# Entrypoint

import argparse

def main(args):
    fp16 = not args.fp32
    logger.info(f"Loading as FP16: {fp16}")

    model = load_model(args, args.model, fp16)

    evaluate(model, args.dataset_dir, fp16=fp16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model", type=str, default="cifar100.pth", help="Path to the model file produced by export_trained_model.py")
    parser.add_argument("--arch", type=str, default="vit_tiny", help="Model architecture (must match model file)")
    parser.add_argument('--fp32', action='store_true', help='Use FP32 network instead of FP16 (only if you trained in fp32 instead)')
    parser.add_argument("--dataset-dir", type=str, default=str("cifar100"), help="Path to the dataset directory (default: ./cifar100/)")

    args = parser.parse_args()

    main(args)