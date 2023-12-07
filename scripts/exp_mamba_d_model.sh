#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_mamba_d_model.sh" # no spaces allowed
name="vary_d_model" # no spaces allowed
arch="mamba" # no spaces allowed
iterations=3

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --params="${params}" --notes="${notes}"
    done
}

params="d_model=320,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=256,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=192,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=128,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=64,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=32,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=16,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training

params="d_model=8,d_state=16,d_conv=4,expand=2,n_layers=4" # no spaces allowed
launch_training
