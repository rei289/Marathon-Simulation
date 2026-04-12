"""Use to run SHAP sensitivity analysis on the model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def generate_lhs_samples(n_samples: int, variable_bounds: dict[str, list[float]]) -> np.ndarray:
    """Use to generate LHS samples scaled to specific variable ranges.

    :param n_samples: Number of samples to generate
    :param variable_bounds: Dictionary with variable names as keys and [min, max] lists as values
    :return: Scaled LHS samples as a NumPy array
    """
    # initialize the LHS sampler
    sampler = qmc.LatinHypercube(d=len(variable_bounds))

    # generate samples in the unit hypercube [0, 1]
    unscaled_samples = sampler.random(n=n_samples)

    # map the samples to your actual variable ranges
    lower_bounds = [b[0] for b in variable_bounds.values()]
    upper_bounds = [b[1] for b in variable_bounds.values()]

    return qmc.scale(unscaled_samples, lower_bounds, upper_bounds)


def build_mixed_input_dataframe(
    n_samples: int,
    params: dict[str, list],
    categorical_columns: tuple[str, ...] = ("pacing_strat",),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a mixed input dataframe with LHS numeric columns and sampled categorical columns."""
    rng = np.random.default_rng(seed)

    numeric_bounds = {name: bounds for name, bounds in params.items() if name not in categorical_columns}
    df_numeric = pd.DataFrame(generate_lhs_samples(n_samples, numeric_bounds), columns=numeric_bounds.keys())

    df_categorical = pd.DataFrame(index=df_numeric.index)
    for column_name in categorical_columns:
        df_categorical[column_name] = rng.choice(params[column_name], size=n_samples)

    return pd.concat([df_numeric, df_categorical], axis=1)[list(params.keys())]


def prepare_shap_features(x: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns so tree models and SHAP can consume them."""
    categorical_columns = x.select_dtypes(include=["object", "category", "string"]).columns
    if len(categorical_columns) == 0:
        return x

    return pd.get_dummies(x, columns=list(categorical_columns), drop_first=False)

def run_shap_analysis(x: pd.DataFrame, y: np.ndarray) -> None:
    """Use to run SHAP analysis on the given input features and target variable.

    :param x: Input features as a DataFrame
    :param y: Target variable as a NumPy array
    """
    x_encoded = prepare_shap_features(x)

    x_numeric = x_encoded.apply(pd.to_numeric, errors="coerce")
    y_series = pd.to_numeric(pd.Series(np.asarray(y).ravel(), index=x_numeric.index), errors="coerce")

    x_array = x_numeric.to_numpy(dtype=np.float64)
    y_array = y_series.to_numpy(dtype=np.float64)

    valid_rows = np.isfinite(y_array) & np.isfinite(x_array).all(axis=1)
    x_clean = x_numeric.loc[valid_rows].astype(np.float64)
    y_clean = y_series.loc[valid_rows].to_numpy(dtype=np.float64)

    x_train, x_test, y_train, _ = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, feature_names=x_test.columns, show=False)

    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


