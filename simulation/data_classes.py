"""f_maxile contains data classes for the marathon simulation project."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    """Configuration for the marathon simulation."""

    target_dist: float
    num_sim: int
    dt: float
    max_steps: int

    const_v: float|None = None
    t1: float|None = None
    t2: float|None = None

@dataclass
class Params:
    """Parameters for the marathon simulation. Each parameter is a list of two values representing the lower and upper bounds for sampling."""

    # contain bounds
    f_max: list[float]                  # Max thrust (m/s^2)
    e_init: list[float]                 # Initial energy (m^2/s^2)
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

@dataclass
class PacingContext:
    """Context class to hold the current state of the simulation for use in pacing strategies."""

    dt: float
    velocity: np.ndarray
    energy: np.ndarray
    theta: np.ndarray
    headwind: np.ndarray
    tau: np.ndarray
    mass: np.ndarray
    rho: np.ndarray
    drag_coefficient: np.ndarray
    frontal_area: np.ndarray
    f_max: np.ndarray
    g: float
