"""Plot logical error rates, estimate the threshold and the suppression factor Lambda.

Usage:
    python plot_threshold.py                    # reads data/circuit.csv
    python plot_threshold.py --noise phenomenological

Writes figures/<noise>_threshold.png with two panels:
  left   logical error rate per shot  -> threshold read from curve crossings
  right  logical error rate per round -> error suppression factor Lambda

Why two different quantities?
  * Threshold: the per-shot curves (with rounds = d) cross at a single point,
    because at threshold a bigger code is exactly as good as a smaller one over
    the whole experiment. Near threshold the per-shot rate approaches 0.5, and
    the per-round conversion (which assumes independent rounds) becomes very
    sensitive to noise and biased, so it is not used there.
  * Lambda: well below threshold the per-round rate is the physically meaningful
    figure of merit (a computation lasts a fixed time, not a fixed number of
    shots), and it removes the trivial effect of bigger codes running more rounds.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sinter

from qec.decode import per_round_error_rate

# Fixed categorical order (blue, orange, aqua, yellow); a distance never changes colour.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]  # second encoding, so colour is never the only cue
NOISE_LABEL = {"circuit": "circuit-level", "phenomenological": "phenomenological"}
MAX_SHOT_RATE_FOR_PER_ROUND = 0.35  # beyond this the per-round conversion is unreliable


def load(noise: str):
    """Return {d: list of dicts with p, shot (best/low/high), round (best/low/high)}."""
    stats = sinter.read_stats_from_csv_files(f"data/{noise}.csv")
    curves = defaultdict(list)
    for s in stats:
        d, p, r = s.json_metadata["d"], s.json_metadata["p"], s.json_metadata["r"]
        if s.errors == 0:
            continue  # no estimate possible
        # Interval of rates whose likelihood is within 1000x of the best fit.
        fit = sinter.fit_binomial(num_shots=s.shots - s.discards, num_hits=s.errors,
                                  max_likelihood_factor=1000)
        curves[d].append(dict(
            p=p,
            shot=(fit.best, fit.low, fit.high),
            round=tuple(per_round_error_rate(x, r) for x in (fit.best, fit.low, fit.high)),
        ))
    return {d: sorted(v, key=lambda x: x["p"]) for d, v in sorted(curves.items())}


def crossing(c_small, c_big, key="shot"):
    """p where two curves cross, interpolated linearly in log-log space.

    Below threshold the bigger code is better (lower curve), above it is worse,
    so we look for the sign change of log(eps_big) - log(eps_small).
    """
    a = {x["p"]: x[key][0] for x in c_small}
    b = {x["p"]: x[key][0] for x in c_big}
    ps = sorted(set(a) & set(b))
    diff = [np.log(b[p]) - np.log(a[p]) for p in ps]
    for i in range(len(ps) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            x0, x1 = np.log(ps[i]), np.log(ps[i + 1])
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp(x0 + t * (x1 - x0)))
    return None


def draw(ax, curves, key, title, pth=None):
    for i, (d, pts) in enumerate(curves.items()):
        if key == "round":
            pts = [x for x in pts if x["shot"][0] <= MAX_SHOT_RATE_FOR_PER_ROUND]
        p = np.array([x["p"] for x in pts])
        e, lo, hi = (np.array([x[key][j] for x in pts]) for j in range(3))
        c = COLORS[i % len(COLORS)]
        ax.fill_between(p, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(p, e, color=c, lw=2, marker=MARKERS[i % len(MARKERS)], ms=5,
                markeredgecolor="white", markeredgewidth=1, label=f"d = {d}")
    if pth:
        ax.axvline(pth, color="#888888", lw=1, ls="--")
        ax.annotate(f"p_th ≈ {pth * 100:.2f}%", (pth, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=9, color="#555555")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p")
    ax.set_title(title, fontsize=10, loc="left", color="#222222")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    ax.grid(True, which="minor", color="#f2f2f2", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#555555", labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", default="circuit")
    args = parser.parse_args()

    curves = load(args.noise)
    ds = list(curves)

    # ---- threshold from per-shot crossings ----
    print(f"Noise model: {args.noise}")
    found = []
    for da, db in zip(ds, ds[1:]):
        x = crossing(curves[da], curves[db], "shot")
        print(f"  per-shot crossing d={da} / d={db}: " + (f"p ~ {x:.4f}" if x else "not in sweep"))
        if x:
            found.append(x)
    pth = float(np.mean(found)) if found else None
    if pth:
        print(f"  threshold estimate: p_th ~ {pth:.4f} ({pth * 100:.2f}%)"
              f"  [spread {min(found) * 100:.2f}-{max(found) * 100:.2f}%]")

    # ---- Lambda from per-round rates at the lowest p ----
    p_low = min(set.intersection(*({x["p"] for x in curves[d]} for d in ds)))
    print(f"  error suppression per round at p = {p_low:.4f}:")
    for da, db in zip(ds, ds[1:]):
        ea = next(x["round"][0] for x in curves[da] if x["p"] == p_low)
        eb = next(x["round"][0] for x in curves[db] if x["p"] == p_low)
        print(f"    Lambda(d={da}->{db}) = {ea / eb:.2f}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150)
    draw(ax1, curves, "shot", "Logical error rate per shot (rounds = d)", pth)
    draw(ax2, curves, "round", "Logical error rate per round")
    ax1.set_ylabel("Logical error rate")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Rotated surface code memory · {NOISE_LABEL.get(args.noise, args.noise)} noise · MWPM",
                 x=0.01, ha="left", fontsize=11, color="#222222")
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = f"figures/{args.noise}_threshold.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
