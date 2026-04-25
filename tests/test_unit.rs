use stride_sim_rust::monte_carlo_simulation::{
    MonteCarloSimulation, SimulationConfig, SimulationInput, Weather, CourseProfile, RunnerParams, PacingStrategy
};
use stride_sim_rust::param_config::get_min_max_values;
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

// entire simulation pipeline is tested here, from input struct to output file writing
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

// specific unit tests input variables
#[test]
fn runner_params_within_valid_range() {
    let runner =  RunnerParams {
        runner_id: 1 as u32,
        f_max: Acceleration::new::<meter_per_second_squared>(3.0),
        e_init: AvailableEnergy::new::<joule_per_kilogram>(5000.0),
        tau: Time::new::<second>(5.0),
        sigma: SpecificPower::new::<watt_per_kilogram>(5.0),
        k: Frequency::new::<hertz>(0.1),
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
    };

    // check that all parameters are within expected ranges
    // f_max
    let Ok((f_max_min, f_max_max)) = get_min_max_values("physical", "acceleration") else {
        panic!("failed to get min/max values for f_max");
    };
    assert!(runner.f_max.get::<meter_per_second_squared>() >= f_max_min && runner.f_max.get::<meter_per_second_squared>() <= f_max_max);
    // e_init
    let Ok((e_init_min, e_init_max)) = get_min_max_values("physical", "energy") else {
        panic!("failed to get min/max values for e_init");
    };
    assert!(runner.e_init.get::<joule_per_kilogram>() >= e_init_min && runner.e_init.get::<joule_per_kilogram>() <= e_init_max);
    // tau
    let Ok((tau_min, tau_max)) = get_min_max_values("physical", "tau") else {
        panic!("failed to get min/max values for tau");
    };
    assert!(runner.tau.get::<second>() >= tau_min && runner.tau.get::<second>() <= tau_max);
    // sigma
    let Ok((sigma_min, sigma_max)) = get_min_max_values("physical", "sigma") else {
        panic!("failed to get min/max values for sigma");
    };
    assert!(runner.sigma.get::<watt_per_kilogram>() >= sigma_min && runner.sigma.get::<watt_per_kilogram>() <= sigma_max);
    // k
    let Ok((k_min, k_max)) = get_min_max_values("physical", "gamma") else {
        panic!("failed to get min/max values for k");
    };
    assert!(runner.k.get::<hertz>() >= k_min/2.0 && runner.k.get::<hertz>() <= k_max*2.0);
    // gamma
    let Ok((gamma_min, gamma_max)) = get_min_max_values("physical", "gamma") else {
        panic!("failed to get min/max values for gamma");
    };
    assert!(runner.gamma.get::<hertz>() >= gamma_min && runner.gamma.get::<hertz>() <= gamma_max);
    // drag_coefficient
    let Ok((drag_coefficient_min, drag_coefficient_max)) = get_min_max_values("environmental", "drag_coefficient") else {
        panic!("failed to get min/max values for drag_coefficient");
    };
    assert!(runner.drag_coefficient >= drag_coefficient_min && runner.drag_coefficient <= drag_coefficient_max);
    // frontal_area
    let Ok((frontal_area_min, frontal_area_max)) = get_min_max_values("physical", "area") else {
        panic!("failed to get min/max values for frontal_area");
    };
    assert!(runner.frontal_area.get::<square_meter>() >= frontal_area_min && runner.frontal_area.get::<square_meter>() <= frontal_area_max);
    // mass
    let Ok((mass_min, mass_max)) = get_min_max_values("physical", "mass") else {
        panic!("failed to get min/max values for mass");
    };
    assert!(runner.mass.get::<kilogram>() >= mass_min && runner.mass.get::<kilogram>() <= mass_max);
    // rho
    let Ok((rho_min, rho_max)) = get_min_max_values("environmental", "air_density") else {
        panic!("failed to get min/max values for rho");
    };
    assert!(runner.rho.get::<kilogram_per_cubic_meter>() >= rho_min && runner.rho.get::<kilogram_per_cubic_meter>() <= rho_max);
    // convection
    let Ok((convection_min, convection_max)) = get_min_max_values("environmental", "convection") else {
        panic!("failed to get min/max values for convection");
    };
    assert!(runner.convection.get::<watt_per_square_meter_kelvin>() >= convection_min && runner.convection.get::<watt_per_square_meter_kelvin>() <= convection_max);
    // alpha
    let Ok((alpha_min, alpha_max)) = get_min_max_values("environmental", "alpha") else {
        panic!("failed to get min/max values for alpha");
    };
    assert!(runner.alpha >= alpha_min && runner.alpha <= alpha_max);
    // psi
    let Ok((psi_min, psi_max)) = get_min_max_values("environmental", "psi") else {
        panic!("failed to get min/max values for psi");
    };
    assert!(runner.psi >= psi_min && runner.psi <= psi_max);
    // const_v
    let Ok((const_v_min, const_v_max)) = get_min_max_values("physical", "velocity") else {
        panic!("failed to get min/max values for const_v");
    };
    assert!(runner.const_v.get::<meter_per_second>() >= const_v_min && runner.const_v.get::<meter_per_second>() <= const_v_max);
    
}