#!/bin/bash

# This one compares AdamW and ScheduleFree using xformers architecture

notes="exp_schedulefree.sh" # no spaces allowed
iterations=1
arch="vit_xformers" # no spaces allowed

name="xformers_schedulefree" # no spaces allowed
optimizer="ScheduleFree"

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --lr="${lr}" --weight-decay="${weight_decay}" --optimizer="${optimizer}" --notes="${notes}"
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



name="xformers_adamw" # no spaces allowed
optimizer="AdamW"



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
