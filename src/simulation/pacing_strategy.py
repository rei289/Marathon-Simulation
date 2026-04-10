"""Use to calculate target velocity at each time step based on a pacing strategy."""

import numpy as np

from src.simulation.data_classes import PacingContext, SimConfig


class PacingStrategy:
    """Class for pacing strategies."""

    def __init__(self, sim_cfg: SimConfig) -> None:
        """Initialize the strategy with the SimConfig."""
        self.sim_cfg = sim_cfg
        self.constant_force: np.ndarray | None = None

    def constant_pace(self, ctx: PacingContext) -> np.ndarray:
        """Return the constant target velocity."""
        return ctx.const_v

    def even_effort_pace(self, ctx: PacingContext) -> np.ndarray:
        """Calculate target velocity based on remaining energy to maintain even effort."""
        # first we calculate the constant force if we haven't already
        if self.constant_force is None:
            self.constant_force = self._compute_constant_force(ctx)
        return self.constant_force

    def _compute_constant_force(self, ctx: PacingContext) -> np.ndarray:
        """Calculate the constant force to maintain even effort based on initial conditions."""
        # calculate the initial resistive forces at the start
        # we ignore the effects of terrain and weather for this calculation since we want to maintain even effort regardless of conditions
        f_resistance = (0.5*ctx.rho*ctx.drag_coefficient*ctx.frontal_area*(ctx.const_v)**2)/ctx.mass
        f_const = f_resistance + (ctx.const_v / ctx.tau)  # dv=0 reference

        # check if this force exceeds the maximum force the runner can apply, if so we throw an error
        if np.any(f_const > ctx.f_max):
            error_message = (
                "The constant force required to maintain even effort exceeds the maximum force the runner can apply. \n"
                "Choose a lower constant velocity (const_v) in SimConfig."
            )
            raise ValueError(error_message)

        return f_const
