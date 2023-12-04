import argparse
import json

from results_parser import parse_results_file

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.reset_defaults()
plt.rcdefaults()

# Ensure Matplotlib uses a non-interactive backend
plt.switch_backend('Agg')


def render_error_bars_plot(data, png_file_path, x, y, series=None):
    df = pd.DataFrame(data)

    df[x] = pd.to_numeric(df[x], errors='coerce')

    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 8))
    if series:
        sns.pointplot(data=df, x=x, y=y, capsize=0.1, hue=series, palette='deep')
    else:
        sns.pointplot(data=df, x=x, y=y, capsize=0.1)

    plt.yticks(rotation=45)  # Rotate Y-axis labels

    plt.title(f"{x} vs {y}")  # Add a title
    plt.xlabel(f"{x}")  # X-axis label
    plt.ylabel(f"{y}")  # Y-axis label

    plt.tight_layout()  

    plt.savefig(png_file_path, dpi=300)

def render_scatter_plot(data, png_file_path, x, y, series=None):
    df = pd.DataFrame(data)

    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 8))
    if series:
        sns.scatterplot(data=df, x=x, y=y, hue=series, palette='deep')
    else:
        sns.scatterplot(data=df, x=x, y=y)

    plt.yticks(rotation=45)  # Rotate Y-axis labels

    plt.title(f"{x} vs {y}")  # Add a title
    plt.xlabel(f"{x}")  # X-axis label
    plt.ylabel(f"{y}")  # Y-axis label

    plt.tight_layout()  

    plt.savefig(png_file_path, dpi=300)

def main(args):
    experiments = parse_results_file(args.results, upgrade_params=True)

    if args.name:
        experiments = [e for e in experiments if e.get('name') == args.name]

    if not experiments:
        print("No data matches the filter criterion")
        return

    print(json.dumps(experiments, indent=4))

    filename = "graph"
    if args.name:
        filename += "_" + args.name
    if args.series:
        filename += "_" + args.series
    filename += ".png"

    if not args.series:
        render_scatter_plot(experiments, filename, x="num_params", y="best_val_acc")
    else:
        if args.name:
            render_error_bars_plot(experiments, filename, x="num_params", y="best_val_acc", series=args.series)
        else:
            render_scatter_plot(experiments, filename, x="num_params", y="best_val_acc", series=args.series)

    print("Rendered graphs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse experiment results from a file")
    parser.add_argument("-r", "--results", help="Path to the results file", default="results.txt", nargs="?")
    parser.add_argument("-n", "--name", help="Experiment name to filter on", default="", nargs="?")
    parser.add_argument("-s", "--series", help="Series name", default="", nargs="?")
    args = parser.parse_args()

    main(args)
