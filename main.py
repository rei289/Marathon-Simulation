"""Smoke test for the Rust PyO3 simulation module."""

from __future__ import annotations

import os
import resource
import time

import psutil
import stride_sim_rust


def memory_usage() -> None:
    """Print the current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Current memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def main() -> None:
    """Run a smoke test of the Rust simulation module."""
    print(stride_sim_rust.module_info())

    config = stride_sim_rust.SimulationConfig(
        target_dist=43_000,
        num_sim=10_000,
        dt=0.1,
        max_steps=200_000,
        sample_rate=1.0,  # sample every 1 second
        result_path="rust_sim",

    )

    weather = stride_sim_rust.Weather(
        temperature=None,
        humidity=None,
        solar_radiation=None,
    )

    course = stride_sim_rust.CourseProfile(
        distance=None,
        grade=None,
        headwind=None,
    )

    runners = [
        stride_sim_rust.RunnerParams(
            runner_id=i,
            f_max=10.0,
            e_init=2200.0,
            tau=1.0,
            sigma=28.0,
            gamma=5e-5,
            drag_coefficient=1.0,
            frontal_area=0.48,
            mass=70.0,
            rho=1.225,
            convection=10.0,
            alpha=0.7,
            psi=0.005,
            const_v=4.0,
            pacing="constant",
        )
        for i in range(config.num_sim)
    ]

    config.result_path = "testing_random.parquet"

    start = time.time()
    stride_sim_rust.run_simulation(config, weather, course, runners)
    elapsed = time.time() - start

    print(f"Rust simulation wall time (s): {elapsed:.3f}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak memory: {peak / 1024:,.2f} MB")

if __name__ == "__main__":
    main()
