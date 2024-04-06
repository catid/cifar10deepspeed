#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_schedulefree.sh" # no spaces allowed
name="scheduled" # no spaces allowed
iterations=2

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --lr="${lr}" --params="${params}" --notes="${notes}"
    done
}

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
