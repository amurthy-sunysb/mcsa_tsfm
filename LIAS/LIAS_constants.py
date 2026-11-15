"""
Constants for the experiments pertaining to LIAS.
"""

LIAS_DATA_PATH = "../data/LIAS"
TIME_COL, ISA_COL, ISB_COL, ISC_COL = 0,1,2,3

# Which files (by number, e.g. 60 means ccs60.csv) belong to which motor condition and speed,
# taken from the dataset's README. The other 30 files are not described there, so we skip them.
MOTOR_CONDITIONS_RANGES = {
    "healthy_low": range(0, 18),         # 18 files
    "one_broken_low": range(60, 68),     # 8
    "two_broken_low": range(120, 128),   # 8

    "healthy_med": range(20, 38),        # 18
    "one_broken_med": range(80, 98),     # 18
    "two_broken_med": range(140, 148),   # 8

    "healthy_high": range(40, 58),       # 18
    "one_broken_high": range(100, 118),  # 18
    "two_broken_high": range(160, 178),  # 18
}