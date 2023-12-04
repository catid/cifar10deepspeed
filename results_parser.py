from models.model_loader import get_model_params

# Replace params keys with a more useful dictionary
def experiments_upgrade_params(experiments):
    for e in experiments:
        e["params"] = get_model_params(e["arch"], e["params"])

def parse_results_file(file_path="results.txt", upgrade_params=True):
    experiments = []
    current_experiment = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Experiment:"):
                if current_experiment:
                    experiments.append(current_experiment)
                    current_experiment = {}
            else:
                if line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    current_experiment[key] = value

        if current_experiment:
            experiments.append(current_experiment)

    if upgrade_params:
        experiments_upgrade_params(experiments)

    return experiments
