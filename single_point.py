"""Decode one (distance, p) point using the hand-written pipeline in qec/decode.py.

Usage:
    python single_point.py --distance 5 --p 0.005 --shots 100000

Good for building intuition before running the full sweep, e.g. check that the
logical error rate drops when you raise the distance at p = 0.003, and rises
when you do the same at p = 0.012.
"""

import argparse
import time

from qec.circuits import surface_code_memory
from qec.decode import count_logical_errors, per_round_error_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--noise", default="circuit")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    circuit = surface_code_memory(a.distance, a.p, noise=a.noise)
    print(f"d={a.distance}, rounds={a.distance}, p={a.p}, noise={a.noise}")
    print(f"  qubits={circuit.num_qubits}, detectors={circuit.num_detectors}, "
          f"observables={circuit.num_observables}")

    t = time.time()
    res = count_logical_errors(circuit, a.shots, seed=a.seed)
    dt = time.time() - t
    pr = per_round_error_rate(res.logical_error_rate, a.distance)
    print(f"  {res.logical_errors}/{res.shots} logical errors "
          f"-> per shot {res.logical_error_rate:.2e}, per round {pr:.2e}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
