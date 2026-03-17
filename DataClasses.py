"""
This file contains data classes for the marathon simulation project.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class SimConfig:
    target_dist: float
    num_sim: int
    dt: float
    max_steps: int

    const_v: np.ndarray|None
    t1: float|None
    t2: float|None

    F: float                    # Max thrust (m/s^2)
    E0: float                   # Initial energy (m^2/s^2)
    tau: float                  # Resistance coefficient (s)
    sigma: float                # Energy supply rate (m^2/s^3)
    gamma: float                # Fatigue constant (dimensionless)
    
    drag_coefficient: float     # Drag coefficient for a runner (dimensionless)
    frontal_area: float         # Frontal area of the runner (m^2)
    mass: float                 # Mass of the runner (kg)

    rho: float          # air density at sea level (kg/m^3)
    convection: float  # convection heat transfer coefficient (W/m^2K)
    alpha: float  # absorption coefficient for solar radiation (dimensionless)
    psi: float                  # weighting factor for the drop in aerobic power per temperature (dimensionless)