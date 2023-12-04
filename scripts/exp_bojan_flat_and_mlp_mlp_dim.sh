#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_bojan_flat_and_mlp_mlp_dim.sh" # no spaces allowed
name="vary_mlp_dim" # no spaces allowed
arch="vit_bojan_flat_and_mlp" # no spaces allowed
iterations=3

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --params="${params}" --notes="${notes}"
    done
}

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=512,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=384,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=256,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=192,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=128,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=80,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=64,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=32,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=24,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=16,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=12,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=8,mlp_size=8" # no spaces allowed
launch_training
