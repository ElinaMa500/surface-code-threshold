"""Sweep code distance and physical error rate, collect logical error statistics.

Usage:
    python run_threshold.py                       # circuit-level noise, default sweep
    python run_threshold.py --noise phenomenological
    python run_threshold.py --quick               # small run to check everything works

Results are appended to data/<noise>.csv. The run is resumable: rerunning with
the same output file skips work that is already done.
"""

from __future__ import annotations

import argparse
import os

# Each sinter worker is a separate process that imports numpy. By default
# OpenBLAS starts one thread per CPU core in every process, so N workers create
# N*N threads; on Windows this exhausts memory ("OpenBLAS error: Memory
# allocation still failed"). The simulation does no linear algebra, so one
# BLAS thread per process is enough. Must be set before numpy is imported;
# worker processes inherit it.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import sinter

from qec.circuits import NOISE_MODELS, surface_code_memory

# Physical error rates bracketing the expected threshold of each noise model.
DEFAULT_P = {
    "circuit": np.geomspace(0.003, 0.016, 9),
    "phenomenological": np.geomspace(0.025, 0.08, 9),
}


def make_tasks(distances, ps, noise):
    for d in distances:
        for p in ps:
            yield sinter.Task(
                circuit=surface_code_memory(distance=d, p=float(p), noise=noise),
                # json_metadata is how we label points when plotting later.
                json_metadata={"d": d, "p": float(p), "r": d, "noise": noise},
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise", choices=NOISE_MODELS, default="circuit")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--max-errors", type=int, default=1_000,
                        help="stop a point once this many logical errors are seen "
                             "(relative error of the estimate ~ 1/sqrt(max_errors))")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = parser.parse_args()

    if args.quick:
        args.max_shots, args.max_errors = 20_000, 100

    os.makedirs("data", exist_ok=True)
    out = f"data/{args.noise}.csv"
    tasks = list(make_tasks(args.distances, DEFAULT_P[args.noise], args.noise))

    sinter.collect(
        num_workers=args.workers,
        tasks=tasks,
        decoders=["pymatching"],
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        save_resume_filepath=out,
        print_progress=True,
    )
    print(f"\nSaved statistics to {out}")


if __name__ == "__main__":
    main()
