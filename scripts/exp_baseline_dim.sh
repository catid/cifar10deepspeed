#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_baseline_dim.sh" # no spaces allowed
name="vary_dim" # no spaces allowed
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


params="patch_size=4,dim=640,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=384,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=256,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=160,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=128,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=80,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=64,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=32,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=16,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training
