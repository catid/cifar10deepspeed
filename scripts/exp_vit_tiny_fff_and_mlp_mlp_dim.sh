#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_vit_tiny_fff_and_mlp_mlp_dim.sh" # no spaces allowed
name="vary_mlp_dim" # no spaces allowed
arch="vit_tiny_fff_and_mlp" # no spaces allowed
iterations=3

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --params="${params}" --notes="${notes}"
    done
}

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=8,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=7,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=6,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=5,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=4,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=3,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=2,fff_count=1,mlp_size=8" # no spaces allowed
launch_training

# count=2

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=8,fff_count=2,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=7,fff_count=2,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=6,fff_count=2,mlp_size=8" # no spaces allowed
launch_training

# count=3

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=7,fff_count=3,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=6,fff_count=3,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=5,fff_count=3,mlp_size=8" # no spaces allowed
launch_training

# count=4

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=6,fff_count=4,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=5,fff_count=4,mlp_size=8" # no spaces allowed
launch_training

params="patch_size=4,dim=512,depth=4,heads=6,fff_depth=4,fff_count=4,mlp_size=8" # no spaces allowed
launch_training
