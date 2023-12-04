import argparse
import json

from results_parser import parse_results_file

def main(args):
    experiments = parse_results_file(args.results, upgrade_params=True)

    print(json.dumps(experiments, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse experiment results from a file")
    parser.add_argument("-r", "--results", help="Path to the results file", default="results.txt", nargs="?")
    args = parser.parse_args()

    main(args)
