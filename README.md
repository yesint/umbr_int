# umbr_int
Implementation of Umbrella Integration algorithm for unbiasing Umbrella Sampling simulations

The [method](https://doi.org/10.1063/1.2052648) is taken from: Johannes Kästner and Walter Thiel, "Bridging the gap between thermodynamic integration and umbrella sampling provides a novel analysis method: "Umbrella integration", J. Chem. Phys. 123, 144104 (2005).

This repository contains the pure Pyton implementation and an example toml input file.

The file umbr_int.f90 is my original implementation in fortran90 from 2007, which is left here for nostalgic reasons :)

Usage:
    python3 umbr_int.py inp.toml