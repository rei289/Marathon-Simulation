"""Collection of plotting functions for visualizing simulation results."""
import matplotlib.pyplot as plt

from src.simulation.monte_carlo_simulation import MonteCarloSimulation


def spaghetti_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot all the results of the simulation (Note costs a lot of memory)."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sim.time_elapsed, sim.velocity, color="blue", alpha=0.05, label="Velocity (m/s)")
    plt.title(f"Monte Carlo: {sim.num_sim} Simulations Runner Velocity Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.subplot(2, 1, 2)
    plt.plot(sim.time_elapsed, sim.energy, color="red", alpha=0.05, label="Energy (J)")
    plt.title(f"Monte Carlo: {sim.num_sim} Simulations Runner Energy Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.tight_layout()
    plt.show()

def histogram_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot a histogram of the finish times of the simulations."""
    plt.figure(figsize=(10, 6))
    plt.hist(sim.finish_time, bins=30, color="green", alpha=0.7)
    plt.title("Distribution of Finish Times")
    plt.xlabel("Finish Time (s)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

def elevation_headwind_plots(sim: MonteCarloSimulation) -> None:
    """Use this function to plot the elevation and headwind profiles of the course."""
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(sim.time_elapsed, sim.elevation_profile, label="Elevation Profile (radians)")
    ax[0].set_title("Elevation Profile Over Time")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Grade (radians)")
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(sim.time_elapsed, sim.headwind_profile, label="Headwind Profile (m/s)", color="orange")
    ax[1].set_title("Headwind Profile Over Time")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Headwind Speed (m/s)")
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.show()

def distance_covered_plot(sim: MonteCarloSimulation) -> None:
    """Use this function to plot the distance covered over time for all simulations."""
    # plotting distance covered
    plt.figure(figsize=(12, 4))
    plt.plot(sim.time_elapsed, sim.distance_covered[:, 0], label="Distance Covered (m)")
    plt.title("Distance Covered Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()
