"""Smoke test for the Rust PyO3 simulation module."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import pandas as pd
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

    # read parquet file with pandas to get the number of simulations
    cols = ["distance_m", "grade_percent", "headwind_mps"]
    df = pd.read_parquet("running_simulation_data/01_runs/2026-04-06_13-50/streams.parquet", columns=cols, engine="pyarrow")
    json_content = Path("running_simulation_data/01_runs/2026-04-06_13-50/overall.json").read_text()
    overall_data = json.loads(json_content)

    df = df.iloc[1:]
    d_obs = df["distance_m"].to_numpy()
    grade = df["grade_percent"].to_numpy()
    headwind = df["headwind_mps"].to_numpy()
    run_data = {
        "distance": d_obs,
        "grade": grade,
        "headwind": headwind,
        "total_distance": overall_data["distance"],
        "temperature": overall_data["weather"]["temp"],
        "humidity": overall_data["weather"]["humidity"],
        "solar_radiation": overall_data["weather"]["solarradiation"],
    }

    # get the fitted parameters from the model fitting results
    json_content = Path("running_simulation_data/02_trainings/2026-04-06_13-50/model_coefficients.json").read_text()
    trained_coefficients = json.loads(json_content)

    config = stride_sim_rust.SimulationConfig(
        target_dist=run_data["total_distance"],
        num_sim=1000,
        dt=0.1,
        max_steps=200_000,
        sample_rate=1.0,  # sample every 1 second
        result_path=None,

    )

    weather = stride_sim_rust.Weather(
        temperature=run_data["temperature"],
        humidity=run_data["humidity"],
        solar_radiation=run_data["solar_radiation"],
    )

    course = stride_sim_rust.CourseProfile(
        distance=run_data["distance"],
        grade=run_data["grade"],
        headwind=run_data["headwind"],
    )

    runners = [
        stride_sim_rust.RunnerParams(
            runner_id=i,
            f_max=trained_coefficients["f_max"],
            e_init=trained_coefficients["e_init"],
            tau=trained_coefficients["tau"],
            sigma=trained_coefficients["sigma"],
            gamma=trained_coefficients["gamma"],
            drag_coefficient=trained_coefficients["drag_coefficient"],
            frontal_area=trained_coefficients["frontal_area"],
            mass=trained_coefficients["mass"],
            rho=trained_coefficients["rho"],
            convection=trained_coefficients["convection"],
            alpha=trained_coefficients["alpha"],
            psi=trained_coefficients["psi"],
            const_v=trained_coefficients["const_v"],
            pacing=trained_coefficients["pacing"],
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
