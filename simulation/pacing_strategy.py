"""Use to calculate target velocity at each time step based on a pacing strategy."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from simulation.data_classes import SimConfig


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

class PacingStrategy(ABC):
    """Abstract base class for pacing strategies."""

    @abstractmethod
    def get_target_velocity(self, ctx: PacingContext) -> np.ndarray:
        """Calculate the target velocity for the current time step."""


class ConstantPaceStrategy(PacingStrategy):
    """Pacing strategy that maintains a constant velocity throughout the race."""

    def __init__(self, sim_cfg: SimConfig) -> None:
        """Initialize the strategy with the constant velocity from SimConfig."""
        self.pace_type = "constant velocity"
        self.target_velocity = np.full(sim_cfg.num_sim, sim_cfg.const_v)
        # if const_v is None, we throw an error since this strategy requires a constant velocity to be defined
        if sim_cfg.const_v is None:
            error_message = "ConstantPaceStrategy requires a constant velocity (const_v) to be defined in SimConfig."
            raise ValueError(error_message)

    def get_target_velocity(self, ctx: PacingContext) -> np.ndarray:
        """Return the constant target velocity."""
        return self.target_velocity


class EvenEffortStrategy(PacingStrategy):
    """Pacing strategy that maintains an even effort throughout the race, adjusting velocity based on remaining energy.

    We do this by starting with a target velocity and adjusting it to maintain the same force.
    """

    def __init__(self, sim_cfg: SimConfig) -> None:
        """Initialize the strategy with the SimConfig."""
        self.pace_type = "even effort"
        self.target_velocity = np.full(sim_cfg.num_sim, sim_cfg.const_v)
        self.sim_cfg = sim_cfg
        self.constant_force: np.ndarray | None = None

        if sim_cfg.const_v is None:
            error_message = "EvenEffortStrategy requires a constant velocity (const_v) to be defined in SimConfig as reference pace."
            raise ValueError(error_message)

    def _compute_constant_force(self, ctx: PacingContext) -> np.ndarray:
        """Calculate the constant force to maintain even effort based on initial conditions."""
        # calculate the initial resistive forces at the start
        # we ignore the effects of terrain and weather for this calculation since we want to maintain even effort regardless of conditions
        f_resistance = (0.5*ctx.rho*ctx.drag_coefficient*ctx.frontal_area*(self.target_velocity)**2)/ctx.mass
        f_const = f_resistance + (self.target_velocity / ctx.tau)  # dv=0 reference

        # check if this force exceeds the maximum force the runner can apply, if so we throw an error
        if np.any(f_const > ctx.f_max):
            error_message = (
                "The constant force required to maintain even effort exceeds the maximum force the runner can apply. \n"
                "Choose a lower constant velocity (const_v) in SimConfig."
            )
            raise ValueError(error_message)

        return f_const

    def get_target_velocity(self, ctx: PacingContext) -> np.ndarray:
        """Calculate target velocity based on remaining energy to maintain even effort."""
        # first we calculate the constant force if we haven't already
        if self.constant_force is None:
            self.constant_force = self._compute_constant_force(ctx)
        # calculate the target velocity based on amount of force we can apply based on external conditions
        a = -(ctx.rho*ctx.drag_coefficient*ctx.frontal_area)/(2*ctx.mass)
        b = (1/ctx.dt - (ctx.rho*ctx.drag_coefficient*ctx.frontal_area*ctx.headwind)/ctx.mass)
        c = -(ctx.velocity/ctx.dt \
              + ctx.g*np.sin(ctx.theta) \
              + ((ctx.headwind**2)*ctx.rho*ctx.drag_coefficient*ctx.frontal_area)/(2*ctx.mass)
              + self.constant_force)
        return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) # we take the positive root since velocity must be positive
