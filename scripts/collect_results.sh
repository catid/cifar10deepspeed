#!/bin/bash

set -x

# Modify for your setup: You have to use ssh-copy-id to avoid needing passwords

ssh gpu1.lan cat ~/sources/cifar10deepspeed/results.txt > combined_results.txt
#ssh gpu2.lan cat ~/sources/cifar10deepspeed/results.txt >> combined_results.txt
ssh gpu3.lan cat ~/sources/cifar10deepspeed/results.txt >> combined_results.txt
ssh gpu4.lan cat ~/sources/cifar10deepspeed/results.txt >> combined_results.txt
ssh gpu5.lan cat ~/sources/cifar10deepspeed/results.txt >> combined_results.txt
ssh gpu6.lan cat ~/sources/cifar10deepspeed/results.txt >> combined_results.txt
