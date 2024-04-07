import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def extract_experiments_from_text(text):
    """
    Parses the given text file into an array of objects with fields corresponding to the experiment details.
    
    Args:
    text (str): A string containing the experiment details.
    
    Returns:
    list of dicts: A list of dictionaries, each containing the details of one experiment.
    """
    # Define the regex pattern to capture the details of each experiment
    experiment_pattern = re.compile(
        r"Experiment:"
        r".*?name:\s*(?P<name>schedulefree)"
        r".*?arch:\s*(?P<arch>[^\n]+)"
        r".*?params:\s*(?P<params>[^\n]+)"
        r".*?best_val_acc:\s*(?P<best_val_acc>[\d.]+(?:e-?\d+)?)"
        r".*?best_train_loss:\s*(?P<best_train_loss>[\d.]+(?:e-?\d+)?)"
        r".*?best_val_loss:\s*(?P<best_val_loss>[\d.]+(?:e-?\d+)?)"
        r".*?end_epoch:\s*(?P<end_epoch>\d+)"
        r".*?train_seconds:\s*(?P<train_seconds>[\d.]+(?:e-?\d+)?)"
        r".*?num_params:\s*(?P<num_params>\d+)"
        r".*?git_hash:\s*(?P<git_hash>[^\n]+)"
        r".*?timestamp:\s*(?P<timestamp>[^\n]+)"
        r".*?seed:\s*(?P<seed>\d+)"
        r".*?lr:\s*(?P<lr>[\d.]+(?:e-?\d+)?)"
        r".*?weight_decay:\s*(?P<weight_decay>[\d.]+(?:e-?\d+)?)"
        r".*?max_epochs:\s*(?P<max_epochs>\d+)"
        r".*?optimizer:\s*(?P<optimizer>[^\n]+)"
        r".*?scheduler:\s*(?P<scheduler>[^\n]+)",
        re.DOTALL
    )

    # Find all matches in the text
    matches = experiment_pattern.finditer(text)

    # Create a list of dictionaries for each match
    experiments = []
    for match in matches:
        experiment_data = match.groupdict()
        experiments.append(experiment_data)

    return experiments


def parse_experiment_data(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    experiments = extract_experiments_from_text(content)

    # Not sure why this is here
    #experiments.pop()

    data = []
    for exp in experiments:
        name, best_val_acc, lr, weight_decay = exp["name"], float(exp["best_val_acc"]), float(exp["lr"]), float(exp["weight_decay"])
        print(f"name, best_val_acc, lr, weight_decay = {name, best_val_acc, lr, weight_decay}")
        data.append((lr, weight_decay, best_val_acc))

    return data

def identify_missing_data_points(data, lr_sorted, wd_sorted):
    existing_points = set((lr, wd) for lr, wd, _ in data)
    all_points = set((lr, wd) for lr in lr_sorted for wd in wd_sorted)
    missing_points = all_points - existing_points
    return missing_points

def plot_heatmap(data, output_filename="grid_search_result.png"):
    data_array = np.array(data)
    lr_sorted = sorted(set(data_array[:, 0]))
    wd_sorted = sorted(set(data_array[:, 1]))

    # Identify missing data points
    missing_points = identify_missing_data_points(data, lr_sorted, wd_sorted)
    print(f"Grid dimensions: {len(lr_sorted)} LR x {len(wd_sorted)} WD")
    print(f"Expecting {len(lr_sorted) * len(wd_sorted)} data points.  Got: {len(data)}")
    if missing_points:
        print(f"Missing data points (LR, WD): {missing_points}")

    # Create a matrix for the heatmap
    pivot_table = np.full((len(lr_sorted), len(wd_sorted)), np.nan)  # Use NaN for missing values
    for lr, wd, acc in data:
        lr_idx = lr_sorted.index(lr)
        wd_idx = wd_sorted.index(wd)
        pivot_table[lr_idx, wd_idx] = acc

    # Ensure proper formatting of the numbers
    formatted_lr = ['{:.2e}'.format(x) for x in lr_sorted]
    formatted_wd = ['{:.2e}'.format(x) for x in wd_sorted]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot_table, xticklabels=formatted_wd, yticklabels=formatted_lr, annot=True, cmap="viridis",
                     fmt=".2f", mask=np.isnan(pivot_table), cbar_kws={'label': 'Best Validation Accuracy'})

    # Set the axis labels
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Validation Accuracy for Different LR/WD')

    # Fix for the cut-off issues
    plt.tight_layout()

    # Save and show the figure
    plt.savefig(output_filename)
    #plt.show()

    print(f"Wrote plot to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Plot heatmap from experiment results.")
    parser.add_argument("filepath", type=str, help="Path to the text file containing experiment results.", default="combined_results.txt", nargs='?')
    args = parser.parse_args()

    data = parse_experiment_data(args.filepath)
    if not data:
        print("No data found")
        return
    plot_heatmap(data)

if __name__ == "__main__":
    main()
