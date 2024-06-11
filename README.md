# MD Data Analysis

This repository contains scripts used for processing and analyzing molecular dynamics (MD) simulations using AMBER trajectories.

## Contents

### Scripts

* **hydrogen_bonds_extract:** CPPTRAJ input file for extracting raw hydrogen bond data from trajectories.
* **hydrogen_bonds_analysis.py:** Python script for calculating hydrogen bond occupancy over trajectories.
* **residence_analisys.py:** Python script (requires CPPTRAJ) for extracting and calculating residence times over trajectories.
* **dihedral_dipole:** Scripts for extracting dipole and dihedral data and saving the data for plotting.

### perm tools folder

* **pf_cal_main.py:** Script for calculating permability from trajectories as described for the collective diffusion method by Zhu (Zhu F, Tajkhorshid E & Schulten K (2004) Collective diffusion model for water permeation through microscopic channels. Phys Rev Lett 93, 1–4)
* **perm_events_main.py:** Script for analyzing permeation events from trajectories.
* **hole_traj_MDA.py:** Script for calculating HOLE for each MIP protomer in a trajectory.

## Installation

1. Install Python 3.
2. Install required Python packages:

   ``` numpy scipy matplotlib pandas MDAnalysis pytraj natsort pickle 
   ```
3. Install CPPTRAJ (From ambertools (https://ambermd.org/index.php))


## Contact

For any questions or suggestions, please contact: biofisicadeacuaporinas@gmail.com
