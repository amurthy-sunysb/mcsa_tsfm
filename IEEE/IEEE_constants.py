"""
Constants for the experiments pertaining to the IEEE broken-rotor-bar dataset.
"""

import os

IEEE_DATA_PATH = "../data/IEEE"
IEEE_SIGNALS_PATH = os.path.join(IEEE_DATA_PATH, "signals")
IEEE_LABELS_PATH = os.path.join(IEEE_DATA_PATH, "labels.csv")
SAMPLE_COL, ISA_COL, ISB_COL, ISC_COL = 0, 1, 2, 3

# The motor is off until ~2.0 s, then starts up (until ~3.0 s in rs_torque15_10); windows start at 3.2 s (50 kHz)
STEADY_STATE_START = 160_000
