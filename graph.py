import argparse
import json

def parse_results(file_path):
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

    return experiments

def main(args):
    data = parse_results(args.results)

    print(json.dumps(data, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse experiment results from a file")
    parser.add_argument("-r", "--results", help="Path to the results file", default="results.txt", nargs="?")
    args = parser.parse_args()

    main(args)
