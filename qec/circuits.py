"""Circuit construction for rotated surface code memory experiments.

A "memory experiment" asks: if I prepare a logical state, run `rounds` rounds of
stabilizer measurement while noise acts, then measure out, how often does the
decoder recover the right logical value?

We use Stim's built-in generator for the rotated surface code. It produces a
circuit containing:
  * DETECTOR annotations: parities of measurement results that are deterministic
    (always 0) in the absence of noise. A detector that fires (=1) is a
    "detection event". For the surface code a detector compares one stabilizer
    measurement with the same stabilizer one round earlier.
  * OBSERVABLE_INCLUDE annotations: which final data-qubit measurements form the
    logical operator we are protecting (here logical Z for memory_z).

Two noise models are offered so you can compare thresholds:

  "circuit"          Every operation is noisy (gates, resets, measurements, idling
                     data qubits). This is the realistic model; threshold ~0.7-1%.
  "phenomenological" Data qubits get depolarized once per round and measurement
                     results are flipped, but the syndrome-extraction gates
                     themselves are perfect. Threshold ~3%.

The gap between the two numbers is a good interview talking point: hook errors
and gate errors spreading from ancilla to data qubits are what cost you most of
the threshold.
"""

from __future__ import annotations

import stim

NOISE_MODELS = ("circuit", "phenomenological")


def surface_code_memory(
    distance: int,
    p: float,
    rounds: int | None = None,
    noise: str = "circuit",
    basis: str = "z",
) -> stim.Circuit:
    """Return a noisy rotated-surface-code memory circuit.

    Args:
        distance: code distance d. The code has d*d data qubits and
            d*d - 1 measure qubits. It corrects any floor((d-1)/2) errors.
        p: physical error rate.
        rounds: number of syndrome-extraction rounds. Defaults to d, the usual
            choice: with fewer rounds than d, time-like error chains (repeated
            measurement errors) are artificially short and the experiment
            overstates performance.
        noise: "circuit" or "phenomenological" (see module docstring).
        basis: "z" protects logical |0>, "x" protects logical |+>.
    """
    if noise not in NOISE_MODELS:
        raise ValueError(f"noise must be one of {NOISE_MODELS}, got {noise!r}")
    if rounds is None:
        rounds = distance

    if noise == "circuit":
        # Uniform depolarizing circuit-level noise (often called "SD6"):
        #   - 2-qubit depolarizing channel of strength p after every CNOT
        #     (and 1-qubit depolarizing after every single-qubit Clifford),
        #   - X/Z flip with prob p after every reset,
        #   - result flip with prob p before every measurement,
        #   - 1-qubit depolarizing on data qubits at the start of every round.
        noise_args = dict(
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
    else:  # phenomenological
        # Only data errors between rounds plus noisy measurement outcomes.
        noise_args = dict(
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
        )

    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        **noise_args,
    )
