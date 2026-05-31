import argparse
from experiment import run_experiment
import plots


def main():
    ap = argparse.ArgumentParser(description="Portfoy Optimizasyonu - Metasezgisel Karsilastirma")
    ap.add_argument("--runs", type=int, default=30, help="bagimsiz kosu sayisi")
    ap.add_argument("--iter", type=int, default=200, help="iterasyon sayisi")
    ap.add_argument("--pop", type=int, default=30, help="populasyon boyutu")
    ap.add_argument("--rf", type=float, default=0.02, help="risksiz getiri (yillik)")
    ap.add_argument("--cardinality", type=int, default=None, help="en fazla varlik sayisi (K)")
    args = ap.parse_args()

    run_experiment(n_runs=args.runs, n_pop=args.pop, max_iter=args.iter,
                   rf=args.rf, cardinality=args.cardinality)
    plots.make_all()


if __name__ == "__main__":
    main()
