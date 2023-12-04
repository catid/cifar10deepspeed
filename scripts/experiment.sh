#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_baseline_mlp_dim.sh" # no spaces allowed
name="vary_mlp_dim" # no spaces allowed
arch="vit_tiny" # no spaces allowed
iterations=3

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --params="${params}" --notes="${notes}"
    done
}

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=24" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=12" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=8" # no spaces allowed
launch_training
