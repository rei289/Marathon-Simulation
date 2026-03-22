"""Use to calculate target velocity at each time step based on a pacing strategy."""
from abc import ABC, abstractmethod

import numpy as np

from simulation.data_classes import SimConfig


class PacingStrategy(ABC):
    """Abstract base class for pacing strategies."""

    @abstractmethod
    def get_target_velocity(self) -> np.ndarray:
        """Calculate the target velocity for the current time step."""


class ConstantPaceStrategy(PacingStrategy):
    """Pacing strategy that maintains a constant velocity throughout the race."""

    def __init__(self, sim_cfg: SimConfig) -> None:
        """Initialize the strategy with the constant velocity from SimConfig."""
        self.target_velocity = np.full(sim_cfg.num_sim, sim_cfg.const_v)
        # if const_v is None, we throw an error since this strategy requires a constant velocity to be defined
        if sim_cfg.const_v is None:
            error_message = "ConstantPaceStrategy requires a constant velocity (const_v) to be defined in SimConfig."
            raise ValueError(error_message)

    def get_target_velocity(self) -> np.ndarray:
        """Return the constant target velocity."""
        return self.target_velocity
