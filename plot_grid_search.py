import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_experiment_data(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    pattern = re.compile(r"Experiment:.*?name: (lr.*?wd.*?)\n.*?lr: (0\.\d+).*?weight_decay: (0\.\d+).*?best_val_acc: (\d+\.\d+)", re.DOTALL)
    matches = pattern.findall(content)
    
    data = []
    for name, lr, weight_decay, best_val_acc in matches:
        lr, weight_decay, best_val_acc = map(float, [lr, weight_decay, best_val_acc])
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
    ax.set_title('Heatmap of Best Validation Accuracy')

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
    plot_heatmap(data)

if __name__ == "__main__":
    main()
