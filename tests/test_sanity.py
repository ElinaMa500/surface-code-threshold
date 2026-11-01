"""Sanity checks. Run with `python -m pytest tests` (or `python tests/test_sanity.py`).

Each test checks a property you should be able to justify from first principles.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qec.circuits import surface_code_memory  # noqa: E402
from qec.decode import count_logical_errors, per_round_error_rate  # noqa: E402


def test_noiseless_circuit_has_no_detection_events():
    # Detectors are defined to be deterministic without noise.
    c = surface_code_memory(distance=3, p=0.0)
    det, obs = c.compile_detector_sampler().sample(1000, separate_observables=True)
    assert not det.any() and not obs.any()


def test_circuit_level_distance_equals_code_distance():
    # The smallest set of single faults that flips the logical observable without
    # firing any detector has size d. If the syndrome-extraction CNOT order were
    # wrong, a "hook" error could make this smaller than d.
    for d in (3, 5, 7):
        c = surface_code_memory(distance=d, p=0.001)
        assert len(c.shortest_graphlike_error()) == d


def test_bigger_code_is_better_well_below_threshold():
    r3 = count_logical_errors(surface_code_memory(3, 0.002), 200_000, seed=1)
    r5 = count_logical_errors(surface_code_memory(5, 0.002), 200_000, seed=1)
    assert per_round_error_rate(r5.logical_error_rate, 5) < per_round_error_rate(r3.logical_error_rate, 3)


def test_per_round_conversion_round_trips():
    r, R = 0.01, 7
    shot = (1 - (1 - 2 * r) ** R) / 2
    assert np.isclose(per_round_error_rate(shot, R), r)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
