# PlacentalVilliModelGeneration
Placental Microscale Model and Mesh Generation Pipeline

This repository contains the pipeline for automatically generating placental microscale models and their corresponding finite-element meshes, as presented in the paper: **“Multiscale computational modeling to quantify how spiral artery remodeling alters wall shear stress on placental villi”**.

The pipeline generates fully prepared simulation input files suitable for computational fluid dynamics (CFD) simulations in SimVascular.

## Overview

The workflow automates the creation of:

- Placental microstructure geometries
- Finite-element meshes
- Boundary surface files required for simulation setup

The final output is a **mesh-complete** directory containing all files necessary to run SimVascular simulations.
