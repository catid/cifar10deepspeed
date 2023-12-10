#!/bin/bash

# See models/model_loader.py for different architectures

notes="exp_x_transformers.sh" # no spaces allowed
arch="x_transformers" # no spaces allowed
name="grid_search" # no spaces allowed
iterations=3

# Function to launch training
launch_training() {
    # Loop to run the command multiple times
    for ((i=1; i<=iterations; i++)); do
        echo "Launching training (Iteration $i)..."
        ./launch_local_train.sh --reset --name="${name}" --arch="${arch}" --params="${params}" --notes="${notes}"
    done
}

params="dim=512,depth=6,heads=8" # no spaces allowed
launch_training
params="dim=512,depth=5,heads=8" # no spaces allowed
launch_training
params="dim=512,depth=5,heads=7" # no spaces allowed
launch_training
params="dim=512,depth=4,heads=7" # no spaces allowed
launch_training
params="dim=512,depth=4,heads=6" # no spaces allowed
launch_training

params="dim=384,depth=6,heads=8" # no spaces allowed
launch_training
params="dim=384,depth=5,heads=8" # no spaces allowed
launch_training
params="dim=384,depth=5,heads=7" # no spaces allowed
launch_training
params="dim=384,depth=4,heads=7" # no spaces allowed
launch_training
params="dim=384,depth=4,heads=6" # no spaces allowed
launch_training

params="dim=256,depth=6,heads=8" # no spaces allowed
launch_training
params="dim=256,depth=5,heads=8" # no spaces allowed
launch_training
params="dim=256,depth=5,heads=7" # no spaces allowed
launch_training
params="dim=256,depth=4,heads=7" # no spaces allowed
launch_training
params="dim=256,depth=4,heads=6" # no spaces allowed
launch_training

params="dim=192,depth=6,heads=8" # no spaces allowed
launch_training
params="dim=192,depth=5,heads=8" # no spaces allowed
launch_training
params="dim=192,depth=5,heads=7" # no spaces allowed
launch_training
params="dim=192,depth=4,heads=7" # no spaces allowed
launch_training
params="dim=192,depth=4,heads=6" # no spaces allowed
launch_training

params="dim=128,depth=6,heads=8" # no spaces allowed
launch_training
params="dim=128,depth=5,heads=8" # no spaces allowed
launch_training
params="dim=128,depth=5,heads=7" # no spaces allowed
launch_training
params="dim=128,depth=4,heads=7" # no spaces allowed
launch_training
params="dim=128,depth=4,heads=6" # no spaces allowed
launch_training
