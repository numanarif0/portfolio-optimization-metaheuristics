import argparse
from experiment import run_experiment
import plots


def main():
    ap = argparse.ArgumentParser(description="Portfolio Optimization - Metaheuristic Comparison")
    ap.add_argument("--runs", type=int, default=30, help="number of independent runs")
    ap.add_argument("--iter", type=int, default=200, help="number of iterations")
    ap.add_argument("--pop", type=int, default=30, help="population size")
    ap.add_argument("--rf", type=float, default=0.02, help="risk-free rate (annual)")
    ap.add_argument("--cardinality", type=int, default=None, help="max number of assets (K)")
    args = ap.parse_args()

    run_experiment(n_runs=args.runs, n_pop=args.pop, max_iter=args.iter,
                   rf=args.rf, cardinality=args.cardinality)
    plots.make_all()


if __name__ == "__main__":
    main()
