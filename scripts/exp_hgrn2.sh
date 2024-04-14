#!/bin/bash

# This one compares AdamW and ScheduleFree using xformers architecture

notes="exp_hgrn2.sh" # no spaces allowed
iterations=1
arch="vit_hgrn2" # no spaces allowed

name="hgrn2" # no spaces allowed

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --lr="${lr}" --weight-decay="${weight_decay}" --notes="${notes}"
    done
}

weight_decay=0.0

lr=0.0005
launch_training
lr=0.001
launch_training
lr=0.0025
launch_training
lr=0.005
launch_training
lr=0.0075
launch_training
lr=0.01
launch_training
lr=0.015
launch_training
lr=0.02
launch_training
lr=0.025
launch_training
lr=0.03
launch_training

weight_decay=0.001

lr=0.0005
launch_training
lr=0.001
launch_training
lr=0.0025
launch_training
lr=0.005
launch_training
lr=0.0075
launch_training
lr=0.01
launch_training
lr=0.015
launch_training
lr=0.02
launch_training
lr=0.025
launch_training
lr=0.03
launch_training

weight_decay=0.01

lr=0.0005
launch_training
lr=0.001
launch_training
lr=0.0025
launch_training
lr=0.005
launch_training
lr=0.0075
launch_training
lr=0.01
launch_training
lr=0.015
launch_training
lr=0.02
launch_training
lr=0.025
launch_training
lr=0.03
launch_training

weight_decay=0.1

lr=0.0005
launch_training
lr=0.001
launch_training
lr=0.0025
launch_training
lr=0.005
launch_training
lr=0.0075
launch_training
lr=0.01
launch_training
lr=0.015
launch_training
lr=0.02
launch_training
lr=0.025
launch_training
lr=0.03
launch_training
