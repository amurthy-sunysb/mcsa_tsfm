"""
Constants for the experiments pertaining to the IEEE broken-rotor-bar dataset.
"""

import os

IEEE_DATA_PATH = "../data/IEEE"
IEEE_SIGNALS_PATH = os.path.join(IEEE_DATA_PATH, "signals")
IEEE_LABELS_PATH = os.path.join(IEEE_DATA_PATH, "labels.csv")
SAMPLE_COL, ISA_COL, ISB_COL, ISC_COL = 0, 1, 2, 3
