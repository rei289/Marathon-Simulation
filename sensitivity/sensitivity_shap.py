"""
Use to run SHAP sensitivity analysis on the model
"""
import numpy as np
import pandas as pd
import shap
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from simulation.data_classes import SimConfig
from simulation.monte_carlo_simulation import MonteCarloSimulation


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

def run_shap_analysis(X: pd.DataFrame, y: np.ndarray):
    """
    Runs SHAP analysis on the given input features and target variable.
    
    :param X: Input features as a NumPy array
    :param y: Target variable as a NumPy array
    """
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # train a Random Forest regressor on the training data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # create a SHAP explainer for the trained model
    explainer = shap.Explainer(model, X_train)
    
    # calculate SHAP values for the test set
    shap_values = explainer(X_test)
    
    # visualize the SHAP values (e.g., summary plot)
    shap.summary_plot(shap_values, X_test)

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
    n_samples = 10000
    X = generate_lhs_samples(n_samples, len(variable_bounds), variable_bounds)

    # convert the scaled samples to a DataFrame with appropriate column names
    df_input = pd.DataFrame(X, columns=variable_bounds.keys())

    # run the simulation for each sample and collect the outputs
    sim = MonteCarloSimulation(SimConfig(target_dist=4300, num_sim=n_samples, dt=0.1, max_steps=10000), df_input, csv_data=None, json_data=None)

    sim.run()

    y = sim.finish_time  # run the simulation for each sample and collect the outputs

    # run SHAP analysis on the collected data
    run_shap_analysis(df_input, y)