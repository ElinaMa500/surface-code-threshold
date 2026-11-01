"""The full decoding pipeline, written out step by step (no sinter).

sinter (used in run_threshold.py) does exactly this, just in parallel and with
automatic stopping. Read this file first: it is the part you must be able to
explain line by line in an interview.

Pipeline:
    noisy circuit
      -> detector error model (DEM)      which faults flip which detectors
      -> matching graph                  nodes = detectors, edges = faults
      -> sample detection events + true observable flips
      -> MWPM decoder predicts observable flips from detection events
      -> logical error  <=>  prediction != truth
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymatching
import stim


@dataclass
class DecodeResult:
    shots: int
    logical_errors: int

    @property
    def logical_error_rate(self) -> float:
        return self.logical_errors / self.shots


def count_logical_errors(circuit: stim.Circuit, shots: int, seed: int | None = None) -> DecodeResult:
    # 1. Detector error model.
    #    Stim propagates every possible single fault in the circuit (each Pauli
    #    term of each noise channel) to the end and records which detectors and
    #    which observables it flips, together with its probability.
    #    decompose_errors=True splits Y-type faults (which flip both X- and
    #    Z-type detectors, i.e. would be hyperedges) into pairs of graphlike
    #    edges, because matching can only handle edges with <= 2 endpoints.
    dem = circuit.detector_error_model(decompose_errors=True)

    # 2. Matching graph. Edge weight = log((1-q)/q) for a fault of probability q,
    #    so the minimum-weight correction is the most likely one (approximately:
    #    MWPM ignores degeneracy and X/Z correlations).
    matcher = pymatching.Matching.from_detector_error_model(dem)

    # 3. Sample. Stim's frame simulator is fast because it only tracks the
    #    *difference* (Pauli frame) from a noiseless reference run.
    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(shots, separate_observables=True)

    # 4. Decode. Sparse blossom finds the minimum-weight set of edges whose
    #    boundary is exactly the set of fired detectors (detectors may also be
    #    matched to the code boundary). The prediction is the XOR of the
    #    observable bits along those edges.
    predictions = matcher.decode_batch(detection_events)

    # 5. A logical error is a shot where the decoder's guess about whether the
    #    logical observable was flipped disagrees with what really happened.
    #    Note: the decoder never "fixes" qubits here; in a memory experiment we
    #    only need to correct the final logical measurement in software (Pauli
    #    frame tracking).
    num_errors = int(np.sum(np.any(predictions != observable_flips, axis=1)))
    return DecodeResult(shots=shots, logical_errors=num_errors)


def per_round_error_rate(per_shot_rate: float, rounds: int) -> float:
    """Convert a per-shot logical error rate into a per-round rate.

    Model each round as an independent chance r of flipping the logical bit.
    After R rounds the bit is flipped iff an odd number of flips happened:
        P_shot = (1 - (1 - 2r)^R) / 2
    Solving for r gives the formula below. Comparing per-round rates is fairer
    than per-shot rates because larger distances also run more rounds.
    """
    if per_shot_rate >= 0.5:
        return 0.5
    return (1 - (1 - 2 * per_shot_rate) ** (1 / rounds)) / 2
