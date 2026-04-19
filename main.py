"""Smoke test for the Rust PyO3 simulation module."""

from __future__ import annotations

import os
import statistics
import time

import psutil


def memory_usage() -> None:
    """Print the current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Current memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def main() -> None:
    """Run a smoke test of the Rust simulation module."""
    try:
        import stride_sim_rust
    except ImportError as exc:
        error_message = (
            "Could not import stride_sim_rust. Build/install it first:\n"
            "  cd rust_sim\n"
            "  maturin develop\n"
        )
        raise SystemExit(error_message) from exc

    print(stride_sim_rust.module_info())

    config = stride_sim_rust.SimulationConfig(
        target_dist=4300,
        num_sim=10000,
        dt=0.1,
        max_steps=20000,
    )

    weather = stride_sim_rust.Weather(
        temperature=None,
        humidity=None,
        solar_radiation=None,
        # temperature=20.0,
        # humidity=0.50,
        # solar_radiation=800.0,
    )

    course = stride_sim_rust.CourseProfile(
        distance=None,
        grade=None,
        headwind=None,
        # distance_m=[0.0, 10_000.0, 20_000.0, 30_000.0, 42_195.0],
        # grade=[0.0, 0.0, 0.0, 0.0, 0.0],
        # headwind_mps=[0.0, 0.0, 0.0, 0.0, 0.0],
    )

    runners = [
        stride_sim_rust.RunnerParams(
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
        for _ in range(config.num_sim)
    ]

    start = time.time()
    result = stride_sim_rust.run_simulation(config, weather, course, runners)
    elapsed = time.time() - start

    summaries = result["summaries"]
    finished_times = [s["finish_time_s"] for s in summaries if s["finish_time_s"] is not None]

    print(f"Rust simulation wall time (s): {elapsed:.3f}")
    print(f"Total runners: {len(summaries)}")
    print(f"Finished runners: {len(finished_times)}")
    if finished_times:
        print(f"Average finish time (s): {statistics.mean(finished_times):.2f}")
        print(f"First 5 finish times (s): {finished_times[:5]}")
    else:
        print("No runners finished in this smoke test.")
    memory_usage()


if __name__ == "__main__":
    main()
