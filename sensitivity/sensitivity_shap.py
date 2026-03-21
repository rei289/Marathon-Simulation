"""
Use to run SHAP sensitivity analysis on the model
"""
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import qmc
import numpy as np

from simulation.monte_carlo_simulation import MonteCarloSimulation
from simulation.data_classes import SimConfig, Params


def generate_lhs_samples(n_samples: int, dimensions: int, variable_bounds: dict) -> np.ndarray:
    """
    Generates LHS samples scaled to specific variable ranges.
    
    :param n_samples: Number of samples to generate
    :param dimensions: Number of input variables
    :param variable_bounds: Dictionary with variable names as keys and [min, max] lists as values
    :return: Scaled LHS samples as a NumPy array
    """
    # initialize the LHS sampler
    sampler = qmc.LatinHypercube(d=dimensions)
    
    # generate samples in the unit hypercube [0, 1]
    unscaled_samples = sampler.random(n=n_samples)
    
    # map the samples to your actual variable ranges
    lower_bounds = [b[0] for b in variable_bounds.values()]
    upper_bounds = [b[1] for b in variable_bounds.values()]
    
    scaled_samples = qmc.scale(unscaled_samples, lower_bounds, upper_bounds)
    
    return scaled_samples

if __name__ == "__main__":
    # generate test samples using LHS
    variable_bounds = {
        "F": [9.0, 12.0],
        "E0": [1800.0, 2600.0],
        "tau": [0.8, 1.2],
        "sigma": [35.0, 55.0],
        "gamma": [3e-5, 8e-5],
        "drag_coefficient": [0.9, 1.1],
        "frontal_area": [0.4, 0.55],
        "mass": [60.0, 80.0],
        "rho": [1.225, 1.325],
        "convection": [10.0, 12.0],
        "alpha": [0.6, 0.8],
        "psi": [0.003, 0.007]
    }
    n_samples = 1000
    X = generate_lhs_samples(n_samples, len(variable_bounds), variable_bounds)
    y = []
    print(X.shape)