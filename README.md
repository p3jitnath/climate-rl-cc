# Robustness of RL-based State-Dependent Parameter Estimation Under Climate Change Scenarios

## 1. Overview

This repository investigates how robust reinforcement learning-based, state-dependent (tunable) parameter estimation is for idealised climate models under climate-change forcing scenarios. We use Deep Deterministic Policy Gradient (DDPG) agents to adapt tunable parameters in response to prescribed forcings.

The project investigates two key questions:
- Can RL-trained agents generalise to new climate regimes (e.g., warmer conditions)?
- How effective is RL-based parameter adjustment under climate change scenarios?

We test this on two idealised climate models:
- **Radiative Convective Equilibrium (RCE)**: Single-column model with prescribed SST forcing
- **Energy Balance Model (EBM)**: Latitudinal energy balance model with radiative forcing

## 2. Installation & Setup

### Prerequisites
- Python 3.11
- Miniconda

### Environment

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate venv
```

## 3. Usage

### RCE Model Experiments

Run baseline RCE inference (no climate change forcing):
```bash
python cc-rce17_v0.py --delta_sst 0.0
```

Run with +4 K SST forcing:
```bash
python cc-rce17_v0.py --delta_sst 4.0
```

### EBM Model Experiments

Run baseline EBM inference (no climate change):
```bash
python cc-ebm-v1.py --delta_a 0.0
```

Run with −4 W/m² radiative forcing:
```bash
python cc-ebm-v1.py --delta_a 4.0
```

### Visualisation

Launch Jupyter notebooks to visualise results:

```bash
jupyter lab notebooks/
```

## 5. Climate Change Scenarios

Climate change is applied as a **persistent boundary condition** each timestep:

**RCE Model:**
- Sea-surface temperature forcing: `Ts = Ts_baseline + ΔT`
- Default: `ΔT = +4.0 K` (warm world scenario)
- Applied to both CC (no RL) and CC + RL model trajectories
- Baseline model remains unperturbed for reference

**EBM Model:**
- Radiative forcing: `A(latitude) = A_ref(latitude) − ΔA`
- Default: `ΔA = 4.0 W/m²` (decreasing outgoing radiation effect)
- Applied to both CC (no RL) and CC + RL model trajectories
- Baseline model remains unperturbed for reference

## 6. Project Structure

```
climate-rl-cc/
├── cc-rce17_v0.py                                      # RCE inference script
├── cc-ebm-v1.py                                        # EBM inference script
├── environment.yml                                     # Conda environment
├── README.md
│
├── notebooks/
│   ├── CC1.1 - Plot RCE Climate Change Results.ipynb   # RCE visualisation
│   └── CC2.1 - Plot EBM Climate Change Results.ipynb   # EBM visualisation
│
├── datasets/                                           # NCEP reanalysis data
│   ├── air.mon.ltm.1981-2010.nc                        # RCE observations
│   └── skt.sfc.mon.1981-2010.ltm.nc                    # EBM observations
│
├── records/                                            # Training records
│   └── rce17-v0-cc_sst-4.0_*/                          # Run outputs
│   └── ebm-v1-cc_delta_a-4.0_*/                        # Run outputs
│
├── results/
│   ├── imgs/
│   │   ├── pdf/                                        # PDF plots
│   │   └── png/                                        # PNG plots
│   └── data/                                           # JSON data files
│
└── weights/                                            # Trained model weights
    ├── RadiativeConvectiveModel17-v0__ddpg*/
    └── EnergyBalanceModel-v1__ddpg_torch__*/
```
