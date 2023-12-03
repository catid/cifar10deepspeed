#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_baseline_depth.sh" # no spaces allowed
name="vary_depth" # no spaces allowed
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

params="patch_size=4,dim=512,depth=1,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=2,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=3,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=5,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=6,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=7,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=8,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=9,heads=6,mlp_dim=256" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=10,heads=6,mlp_dim=256" # no spaces allowed
launch_training
