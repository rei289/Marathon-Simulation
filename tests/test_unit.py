"""Execute unit tests for the simulation."""

import numpy as np
import pytest

from src.simulation.data_classes import PacingContext, Params, SimConfig
from src.simulation.monte_carlo_simulation import MonteCarloSimulation, create_dataframes
from src.simulation.pacing_strategy import ConstantPaceStrategy, EvenEffortStrategy
from src.utilis.helper import get_param_info


@pytest.fixture
def sim_cfg() -> SimConfig:
    """Fixture to create a SimConfig instance for testing."""
    return SimConfig(
        target_dist=10000,
        num_sim=1,
        dt=0.1,
        max_steps=10000,
        const_v=5.0,
        t1=None,
        t2=None,
    )

@pytest.fixture
def params() -> Params:
    """Fixture to create a Params instance for testing."""
    return Params(
        f_max=[10.0],
        e_init=[2000.0],
        tau=[1.0],
        sigma=[25.0],
        gamma=[5e-5],
        drag_coefficient=[1.0],
        frontal_area=[0.5],
        mass=[70.0],
        rho=[1.225],
        convection=[10.0],
        alpha=[0.7],
        psi=[0.005],
    )

@pytest.fixture
def ctx() -> PacingContext:
    """Fixture to create a PacingContext instance for testing."""
    return PacingContext(
        dt=0.1,
        velocity=np.array([5.0]),
        energy=np.array([2000.0]),
        theta=np.array([0.0]),
        headwind=np.array([0.0]),
        tau=np.array([1.0]),
        mass=np.array([70.0]),
        rho=np.array([1.225]),
        drag_coefficient=np.array([1.0]),
        frontal_area=np.array([0.5]),
        f_max=np.array([10.0]),
        g=9.81,
    )

@pytest.fixture
def sim(sim_cfg: SimConfig, params: Params) -> MonteCarloSimulation:
    """Fixture to create a MonteCarloSimulation instance for testing."""
    strat = ConstantPaceStrategy(sim_cfg)

    df_input = create_dataframes(params, sim_cfg.num_sim)

    return MonteCarloSimulation(sim_cfg, strat, df_input=df_input, csv_data=None, json_data=None)

# overall simulation tests
def test_finish_time(sim: MonteCarloSimulation) -> None:
    """Test that the simulation runs without errors and produces a finish time."""
    sim.run()
    assert len(sim.finish_time) == 1
    assert np.all(sim.finish_time > 0.0 or np.isnan(sim.finish_time))

def test_time_positive(sim: MonteCarloSimulation) -> None:
    """Test that the finish time is positive."""
    sim.run()
    assert np.all(sim.time_elapsed >= 0.0)

def test_distance_positive(sim: MonteCarloSimulation) -> None:
    """Test that the distance covered is positive."""
    sim.run()
    assert np.all(sim.distance_covered >= 0.0)

def test_energy_non_negative(sim: MonteCarloSimulation) -> None:
    """Test that the energy is non-negative."""
    sim.run()
    assert np.all(sim.energy >= 0.0)

def test_velocity_non_negative(sim: MonteCarloSimulation) -> None:
    """Test that the velocity is non-negative."""
    sim.run()
    assert np.all(sim.velocity >= 0.0)

def test_energy_lower_than_initial(sim: MonteCarloSimulation) -> None:
    """Test that the energy is always lower than or equal to the initial energy."""
    sim.run()
    assert np.all(sim.energy <= sim.E0_values)


# pacing strategy tests
def test_constant_pace_strategy(sim_cfg: SimConfig, ctx: PacingContext) -> None:
    """Test that the ConstantPaceStrategy returns the correct target velocity."""
    strat = ConstantPaceStrategy(sim_cfg)
    target_velocity = strat.get_target_velocity(ctx)
    assert np.all(target_velocity == 5.0)

def test_even_effort_strategy(sim_cfg: SimConfig, ctx: PacingContext) -> None:
    """Test that the EvenEffortStrategy returns a target velocity that is not zero."""
    strat = EvenEffortStrategy(sim_cfg)
    target_velocity = strat.get_target_velocity(ctx)
    assert np.all(target_velocity > 0.0)

# boundary condition tests
def test_tau(sim: MonteCarloSimulation) -> None:
    """Test that the tau parameter is between min and max values."""
    sim.run()
    min_tau = get_param_info("tau")["min"]
    max_tau = get_param_info("tau")["max"]
    assert np.all(sim.tau_values >= min_tau)
    assert np.all(sim.tau_values <= max_tau)

def test_sigma(sim: MonteCarloSimulation) -> None:
    """Test that the sigma parameter is between min and max values."""
    sim.run()
    min_sigma = get_param_info("sigma")["min"]
    max_sigma = get_param_info("sigma")["max"]
    assert np.all(sim.sigma_values >= min_sigma)
    assert np.all(sim.sigma_values <= max_sigma)

def test_gamma(sim: MonteCarloSimulation) -> None:
    """Test that the gamma parameter is between min and max values."""
    sim.run()
    min_gamma = get_param_info("gamma")["min"]
    max_gamma = get_param_info("gamma")["max"]
    assert np.all(sim.gamma_values >= min_gamma)
    assert np.all(sim.gamma_values <= max_gamma)
