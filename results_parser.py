from models.model_loader import get_model_params

# Replace params keys with a more useful dictionary
def experiments_upgrade_params(experiments):
    for e in experiments:
        # Insert dictionary keys at the same level as other stuff.
        # This makes it easier to use pandas for graphing the data.
        e.update(get_model_params(e["arch"], e["params"]))

        for key, value in e.items():
            try:
                # First, try converting to float
                float_value = float(value)

                # If the float value is equivalent to an integer, convert it to int
                if float_value.is_integer():
                    e[key] = int(float_value)
                else:
                    e[key] = float_value

            except ValueError:
                # If conversion fails, keep the original value
                pass

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
