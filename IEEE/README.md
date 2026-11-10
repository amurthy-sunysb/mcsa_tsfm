# IEEE rotor dataset: how the data was made

This file is an edited copy of `readme.txt` from the dataset. It also uses the data-layout figure from `Set up.pdf` (page 2).

Source: Treml, Flauzino, Suetake and Maciejewski, *Experimental database for detecting and diagnosing rotor broken bar in a three-phase induction motor*, IEEE DataPort, 2020. DOI [10.21227/fmnm-bn95](https://doi.org/10.21227/fmnm-bn95).

For how to run the pipeline, see [docs/IEEE.md](../docs/IEEE.md).

## Test bench

The bench is at the School of Engineering of São Carlos (EESC), University of São Paulo (USP), Brazil. Two laboratories ran it: LAIPS and LACIME.

| Part | Detail |
|---|---|
| Motor | WEG W22, 1 cv, 220 V / 380 V, 3.02 A / 1.75 A, 4 poles, 60 Hz |
| Rated torque and speed | 4.1 N·m, 1715 rpm |
| Rotor | Squirrel cage, 34 bars |
| Supply | Balanced three-phase mains, 60 Hz. Direct start (no inverter). |
| Load | A direct-current generator. The operator changes the load with the voltage on its field winding (through a Variac and a rectifier). |
| Torque sensor | Transtec MT-103 torque wrench |
| Current sensors | Yokogawa 96033 current probes (up to 50 A RMS, 10 mV/A) |
| Voltage sensors | Yokogawa oscilloscope voltage probes, connected to the motor terminals |
| Vibration sensors | Five Vibrocontrol PU 2001 sensors (10 mV/mm/s, 5 – 2000 Hz). They give vibration **velocity**, not acceleration. |

The five vibration sensors are on the motor (non-drive end, drive end, housing, axial direction) and on the support base.

## How the faults were made

The team drilled holes in the rotor with a 6 mm drill. The hole is wider than one rotor bar. It is at the middle of the rotor length. The team tested a healthy rotor first. Then they replaced it with a rotor that has 1, 2, 3 or 4 adjacent broken bars.

## Tests

- The motor started directly from the mains. The supply was balanced, three-phase, 60 Hz.
- The load on the shaft was 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 and 4.0 N·m.
- The team did 10 repetitions at each load.

## File layout

There is one MATLAB file for each rotor:

| File | Rotor |
|---|---|
| `struct_rs_R1.mat` | Healthy |
| `struct_r1b_R1.mat` | 1 broken bar |
| `struct_r2b_R1.mat` | 2 broken bars |
| `struct_r3b_R1.mat` | 3 broken bars |
| `struct_r4b_R1.mat` | 4 broken bars |

Each file has one struct. The struct name is the rotor name (`rs`, `r1b`, …). In the struct, there is one field for each load (`torque05` … `torque40`). Each load field has 10 rows (the repetitions) and 12 columns (the signals):

| Signal | What it measures | Samples in one recording |
|---|---|---|
| `Va`, `Vb`, `Vc` | Voltage, phases A, B, C | 1,001,000 |
| `Ia`, `Ib`, `Ic` | Current, phases A, B, C | 1,001,000 |
| `Vib_acpi` | Radial vibration velocity, driven side | 153,504 |
| `Vib_carc` | Tangential vibration velocity, housing | 153,504 |
| `Vib_acpe` | Radial vibration velocity, non-driven side | 153,504 |
| `Vib_axial` | Axial vibration velocity, driven side | 153,504 |
| `Vib_base` | Tangential vibration velocity, base | 153,504 |
| `Trigger` | Documented as a constant 5 V | 153,504 |

The sample counts are from the figure in `Set up.pdf`. They are true for `struct_rs_R1.mat`, load `torque05`. In `struct_r1b_R1.mat`, the vibration and `Trigger` lengths change from recording to recording (148,319 to 153,740).

## Differences between the documents

| Item | `Set up.pdf` | `readme.txt` and the real files |
|---|---|---|
| Struct names | `struct_healthy`, `struct_r1b`, … | `struct_rs_R1`, `struct_r1b_R1`, … The struct inside the file is `rs`, `r1b`, … |
| `Trigger` | Constant 5 V | Not constant. In `struct_rs_R1.mat`, `torque05`, recording 1 it goes from −0.22 V to 6.24 V. |
| Recording time | The IEEE DataPort page says 18 s | The current signals have 1,001,000 samples at 50 kHz, which is 20.02 s |
