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

@dataclass
class Params:
    # contain bounds
    F: list[float]                      # Max thrust (m/s^2)
    E0: list[float]                     # Initial energy (m^2/s^2)
    tau: list[float]                    # Resistance coefficient (s)
    sigma: list[float]                  # Energy supply rate (m^2/s^3)
    gamma: list[float]                  # Fatigue constant (dimensionless)

    drag_coefficient: list[float]       # Drag coefficient for a runner (dimensionless)
    frontal_area: list[float]           # Frontal area of the runner (m^2)
    mass: list[float]                   # Mass of the runner (kg)

    rho: list[float]                    # air density at sea level (kg/m^3)
    convection: list[float]             # convection heat transfer coefficient (W/m^2K)
    alpha: list[float]                  # absorption coefficient for solar radiation (dimensionless)
    psi: list[float]                    # weighting factor for the drop in aerobic power per temperature (dimensionless)

    # F: float                    # Max thrust (m/s^2)
    # E0: float                   # Initial energy (m^2/s^2)
    # tau: float                  # Resistance coefficient (s)
    # sigma: float                # Energy supply rate (m^2/s^3)
    # gamma: float                # Fatigue constant (dimensionless)
    
    # drag_coefficient: float     # Drag coefficient for a runner (dimensionless)
    # frontal_area: float         # Frontal area of the runner (m^2)
    # mass: float                 # Mass of the runner (kg)

    # rho: float          # air density at sea level (kg/m^3)
    # convection: float  # convection heat transfer coefficient (W/m^2K)
    # alpha: float  # absorption coefficient for solar radiation (dimensionless)
    # psi: float                  # weighting factor for the drop in aerobic power per temperature (dimensionless)