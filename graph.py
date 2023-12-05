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

    print(f"Saved point plot (with error bars) to: {png_file_path}")

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

    print(f"Saved scatter plot to: {png_file_path}")

def render_lineplot(data, png_file_path, x, y, series):
    df = pd.DataFrame(data)

    sns.set(style="darkgrid")

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x=x, y=y, hue=series, palette='deep')

    plt.yticks(rotation=45)  # Rotate Y-axis labels

    plt.title(f"{x} vs {y}")  # Add a title
    plt.xlabel(f"{x}")  # X-axis label
    plt.ylabel(f"{y}")  # Y-axis label

    plt.tight_layout()  

    plt.savefig(png_file_path, dpi=300)

    print(f"Saved scatter plot to: {png_file_path}")

def main(args):
    experiments = parse_results_file(args.results, upgrade_params=True)

    if args.name:
        experiments = [e for e in experiments if e.get('name') == args.name]

    if not experiments:
        print("No data matches the filter criterion")
        return

    print("Sample experiment result from file: ", json.dumps(experiments[0], indent=4))

    filename = f"graph_{args.type}"
    if args.name:
        filename += "_" + args.name
    if args.series:
        filename += "_" + args.series
    filename += ".png"

    if args.type == "acc":
        x="num_params"
        y="best_val_acc"
    elif args.type == "time":
        x="train_seconds"
        y="best_val_acc"

    if not args.series:
        render_scatter_plot(experiments, filename, x=x, y=y)
    else:
        if args.name and args.type != "time":
            render_lineplot(experiments, filename, x=x, y=y, series=args.series)
        else:
            render_scatter_plot(experiments, filename, x=x, y=y, series=args.series)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse experiment results from a file")
    parser.add_argument("-r", "--results", help="Path to the results file", default="results.txt", nargs="?")
    parser.add_argument("-n", "--name", help="Experiment name to filter on", default="", nargs="?")
    parser.add_argument("-s", "--series", help="Series name", default="", nargs="?")
    parser.add_argument("-t", "--type", help="Graph type", default="acc", nargs="?")
    args = parser.parse_args()

    main(args)
