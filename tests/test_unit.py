"""Execute unit tests for the simulation."""

import numpy as np
import pytest

from simulation.data_classes import PacingContext, Params, SimConfig
from simulation.monte_carlo_simulation import MonteCarloSimulation, create_dataframes
from simulation.pacing_strategy import ConstantPaceStrategy, EvenEffortStrategy


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
        F=[10.0],
        E0=[2000.0],
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
def test_simulation_runs(sim: MonteCarloSimulation) -> None:
    """Test that the simulation runs without errors and produces a finish time."""
    sim.run()
    assert len(sim.finish_time) == 1
    assert sim.finish_time[0] > 0

def test_time_positive(sim: MonteCarloSimulation) -> None:
    """Test that the finish time is positive."""
    sim.run()
    assert np.all(sim.time_elapsed >= 0.0)


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
