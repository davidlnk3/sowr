import argparse


def main() -> None:
    from .experiment import run
    from .llm import GENERATORS

    parser = argparse.ArgumentParser(prog="sowr", description="Run the baselines and the width x depth grid on a dataset.")
    parser.add_argument("dataset", help="directory with root.json and items.jsonl")
    parser.add_argument("--generator", default="gpt-6-luna", choices=list(GENERATORS))
    parser.add_argument("--widths", default="1,2,3", help="comma-separated, e.g. 1,2,3")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=10, help="number of items (10 = pilot)")
    parser.add_argument("--baseline-repeats", type=int, default=1, help="use 3 for the determinism check")
    parser.add_argument("--out", default="runs", help="results.jsonl and saved trees go here")
    args = parser.parse_args()

    run(args.dataset, args.generator, [int(w) for w in args.widths.split(",")], args.max_depth,
        args.limit, args.out, args.baseline_repeats)
