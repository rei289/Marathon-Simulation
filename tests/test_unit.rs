use stride_sim_rust::monte_carlo_simulation::{
    MonteCarloSimulation, SimulationConfig, SimulationInput, Weather, CourseProfile, RunnerParams, PacingStrategy
};
use tempfile::tempdir;

use uom::si::f64::*;
use uom::si::length::meter;
use uom::si::time::second;
use uom::si::velocity::meter_per_second;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::available_energy::joule_per_kilogram;
use uom::si::specific_power::watt_per_kilogram;
use uom::si::area::square_meter;
use uom::si::mass::kilogram;
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::heat_transfer::watt_per_square_meter_kelvin;
use uom::si::thermodynamic_temperature::degree_celsius;
use uom::si::heat_flux_density::watt_per_square_meter;
use uom::si::frequency::hertz;

#[test]
fn simulation_writes_output_without_error() {
    let temp_dir = tempdir().expect("failed to create temp dir");
    let result_path = temp_dir.path().join("test-results.parquet");
    let config = SimulationConfig {
        target_dist: Length::new::<meter>(1000.0),
        num_sim: 2,
        dt: Time::new::<second>(1.0),
        max_steps: 2000,
        sample_rate: Time::new::<second>(5.0),
        result_path: result_path.to_string_lossy().to_string(),
    };

    let runners = (0..config.num_sim).map(|i| RunnerParams {
        runner_id: i as u32,
        f_max: Acceleration::new::<meter_per_second_squared>(3.0),
        e_init: AvailableEnergy::new::<joule_per_kilogram>(50000.0),
        tau: Time::new::<second>(200.0),
        sigma: SpecificPower::new::<watt_per_kilogram>(10.0),
        k: Frequency::new::<hertz>(0.0),
        gamma: Frequency::new::<hertz>(0.1),
        drag_coefficient: 1.0,
        frontal_area: Area::new::<square_meter>(0.5),
        mass: Mass::new::<kilogram>(70.0),
        rho: MassDensity::new::<kilogram_per_cubic_meter>(1.225),
        convection: HeatTransfer::new::<watt_per_square_meter_kelvin>(10.0),
        alpha: 0.5,
        psi: 0.01,
        const_v: Velocity::new::<meter_per_second>(3.5),
        const_f: Acceleration::new::<meter_per_second_squared>(0.0),
        pacing: PacingStrategy::Constant,
    }).collect();

    let input = SimulationInput {
        config,
        weather: Weather {
            temperature: ThermodynamicTemperature::new::<degree_celsius>(20.0),
            humidity: 50.0,
            solar_radiation: HeatFluxDensity::new::<watt_per_square_meter>(700.0),
        },
        course: CourseProfile {
            distance: vec![Length::new::<meter>(0.0), Length::new::<meter>(1000.0)],
            grade: vec![0.0, 0.0],
            headwind: vec![Velocity::new::<meter_per_second>(0.0), Velocity::new::<meter_per_second>(0.0)],
        },
        runners,
    };

    let mut sim = MonteCarloSimulation::new(input).expect("init failed");
    sim.simulate().expect("simulate failed");
    assert!(result_path.exists());
}