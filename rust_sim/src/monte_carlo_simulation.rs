/*
Script performs a Monte Carlo simulation to determine all possible outcomes of running a marathon.
*/

use uom::si::f64::*;
use uom::si::frequency::hertz;
use uom::si::length::meter;
use uom::si::time::second;
use uom::si::velocity::meter_per_second;
use uom::si::acceleration::meter_per_second_squared;
use uom::si::available_energy::joule_per_kilogram;
use uom::si::specific_power::watt_per_kilogram;
use uom::si::heat_transfer::watt_per_square_meter_kelvin;
use uom::si::thermodynamic_temperature::degree_celsius;
use uom::si::heat_flux_density::watt_per_square_meter;

use crate::constants::*;

use std::fs::File;
use std::sync::Arc;
use arrow_array::{
    ArrayRef, RecordBatch, UInt32Array, Float32Array
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

const PARQUET_CHUNK_ROWS: usize = 50_000;


#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PacingStrategy {
    Constant,
    EvenEffort,
}

#[derive(Clone, Debug)]
pub struct SimulationConfig {
    pub target_dist: Length,            // target distance in meters   
    pub num_sim: usize,                 // number of monte carlo simulations to run (i.e. how many runners to simulate)
    pub dt: Time,                       // time step for the simulation (e.g. 0.1 means simulate every 0.1 seconds)
    pub max_steps: usize,               // maximum number of steps to simulate for each runner (to prevent infinite loops if they can't finish)
    pub sample_rate: Time,              // time in seconds to simulate before writing to parquet (e.g. 100 means write every 100 seconds)
    pub result_path: String,            // path to save results (e.g. parquet file)
}

#[derive(Clone, Debug)]
pub struct Weather {
    pub temperature: ThermodynamicTemperature,
    pub humidity: f64,                  // relative humidity as a percentage
    pub solar_radiation: HeatFluxDensity, // solar radiation in W/m^2
}

#[derive(Clone, Debug)]
pub struct CourseProfile {
    pub distance: Vec<Length>,          // distance points for the course profile (m)
    pub grade: Vec<f64>,                // grade at each distance point as a percentage
    pub headwind: Vec<Velocity>,        // headwind speed at each distance point (m/s, positive for headwind, negative for tailwind)
}

#[derive(Clone, Debug)]
pub struct RunnerParams {
    pub f_max: Acceleration,            // maximum thrust (m/s^2)
    pub e_init: AvailableEnergy,        // initial energy (m^2/s^2)
    pub tau: Time,                      // resistance coefficient (s)
    pub sigma: SpecificPower,           // energy supply rate (m^2/s^3)
    pub k: Frequency,                   // fatigue coefficient for energy supply drop with velocity (1/s)
    pub gamma: Frequency,               // fatigue constant (1/s)
    pub drag_coefficient: f64,          // drag coefficient (dimensionless)
    pub frontal_area: Area,             // frontal area (m^2)
    pub mass: Mass,                     // mass (kg)
    pub rho: MassDensity,               // air density at sea level (kg/m^3)
    pub convection: HeatTransfer,       // convection heat transfer coefficient (W/m^2K)
    pub alpha: f64,                     // absorption coefficient for solar radiation (dimensionless)
    pub psi: f64,                       // weighting factor for the drop in aerobic power per temperature (dimensionless)
    pub const_v: Velocity,              // constant velocity for "constant" pacing strategy (m/s)
    pub const_f: Acceleration,          // constant acceleration for "even effort" pacing strategy (m/s^2)
    pub pacing: PacingStrategy,         // pacing strategy type
}

#[derive(Debug)]
pub struct SimulationInput {
    pub config: SimulationConfig,
    pub weather: Weather,
    pub course: CourseProfile,
    pub runners: Vec<RunnerParams>, // len must equal num_sim
}

#[derive(Debug, Clone)]
struct RunnerState {
    velocity: Velocity,
    energy: AvailableEnergy,
    distance: Length,
    time_elapsed: Time,
    finished_step: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ResultBatch {
    pub runner: Vec<u32>,
    pub time_s: Vec<f32>,
    pub velocity_mps: Vec<f32>,
    pub energy_jpkg: Vec<f32>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum SimError {
    InvalidConfig(&'static str),
    LengthMismatch(&'static str),
    InvalidValue(&'static str),
    EmptyCourse(&'static str),
}

pub struct MonteCarloSimulation {
    input: SimulationInput,
}

impl RunnerState {
    fn new(runner: &RunnerParams) -> Self {

        Self {
            velocity: Velocity::new::<meter_per_second>(1e-6),
            energy: runner.e_init,
            distance: Length::new::<meter>(0.0),
            time_elapsed: Time::new::<second>(0.0),
            finished_step: None,
        }
    }
}

impl ResultBatch {
    fn new() -> Self {
        Self {
            runner: Vec::with_capacity(PARQUET_CHUNK_ROWS),
            time_s: Vec::with_capacity(PARQUET_CHUNK_ROWS),
            velocity_mps: Vec::with_capacity(PARQUET_CHUNK_ROWS),
            energy_jpkg: Vec::with_capacity(PARQUET_CHUNK_ROWS),
        }
    }

    fn take_columns(&mut self) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            std::mem::replace(&mut self.runner, Vec::with_capacity(PARQUET_CHUNK_ROWS)),
            std::mem::replace(&mut self.time_s, Vec::with_capacity(PARQUET_CHUNK_ROWS)),
            std::mem::replace(&mut self.velocity_mps, Vec::with_capacity(PARQUET_CHUNK_ROWS)),
            std::mem::replace(&mut self.energy_jpkg, Vec::with_capacity(PARQUET_CHUNK_ROWS)),
        )
    }

    // standard length method
    pub fn len(&self) -> usize {
        self.runner.len()
    }

    // always implement this if you have len()
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MonteCarloSimulation {
    pub fn new(input: SimulationInput) -> Result<Self, SimError> {
        // validate input and initialize state
        if input.config.num_sim == 0 {
            return Err(SimError::InvalidConfig("num_sim must be > 0"));
        }
        if input.config.max_steps == 0 {
            return Err(SimError::InvalidConfig("max_steps must be > 0"));
        }
        if input.config.dt <= Time::new::<second>(0.0) {
            return Err(SimError::InvalidConfig("dt must be > 0"));
        }
        if input.config.target_dist <= Length::new::<meter>(0.0) {
            return Err(SimError::InvalidConfig("target_dist must be > 0"));
        }
        if input.runners.len() != input.config.num_sim {
            return Err(SimError::LengthMismatch("runners length must match num_sim"));
        }

        let course_len = input.course.distance.len();
        if course_len == 0 {
            return Err(SimError::EmptyCourse("course distance must be non-empty"));
        }
        if input.course.grade.len() != course_len || input.course.headwind.len() != course_len {
            return Err(SimError::LengthMismatch("course vectors must have equal lengths"));
        }

        Ok(Self { input })
    }

    pub fn simulate(&mut self) -> Result<(), SimError> {
        /*
        Use to simulate the monte carlo process
         */
        let n = self.input.config.num_sim;
        let tmax = self.input.config.max_steps;

        let mut writer = Self::make_parquet_writer(&self.input.config.result_path)?;
        // let mut row_buffer: Vec<ResultRow> = Vec::with_capacity(PARQUET_CHUNK_ROWS);
        let mut row_buffer = ResultBatch::new();

        let mut count = 0;
        let sample_rate_steps = (self.input.config.sample_rate.get::<second>() / self.input.config.dt.get::<second>()).round() as usize;

        // initialize state for each runner and perform the simulation
        for runner in 0..n {
            // calculate the new input parameters for this runner based on the weather conditions and their individual psi parameters, then perform the simulation for this runner until they finish or reach max steps
            self.init_runner(runner)?;
            let runner_param = &self.input.runners[runner];

            // initialize the state for this runner
            let mut state = RunnerState::new(runner_param);

            // perform the simulation for this runner until they finish or reach max steps
            for step in 0..tmax {
                // check if runner has finished
                if state.distance >= self.input.config.target_dist {
                    state.finished_step = Some(step);
                    break;
                }

                if state.velocity <= Velocity::new::<meter_per_second>(1e-8)
                    && state.energy <= AvailableEnergy::new::<joule_per_kilogram>(1e-8)
                {
                    state.finished_step = Some(step);
                    break;
                }
                
                // determine if we should write to parquet based on sample_rate
                if count == sample_rate_steps {
                    row_buffer.runner.push(runner as u32);
                    row_buffer.time_s.push(state.time_elapsed.get::<second>() as f32);
                    row_buffer.velocity_mps.push(state.velocity.get::<meter_per_second>() as f32);
                    row_buffer.energy_jpkg.push(state.energy.get::<joule_per_kilogram>() as f32);
                    count = 0; // reset the counter
                }

                if row_buffer.len() >= PARQUET_CHUNK_ROWS {
                    Self::flush_parquet_rows(self, &mut writer, &mut row_buffer)?;
                }

                self.step_runner(runner_param, &mut state)?;
                count += 1;
            }
        }

        if !row_buffer.is_empty() {
            Self::flush_parquet_rows(self, &mut writer, &mut row_buffer)?;
        }
        Self::close_parquet_writer(writer)?;

        Ok(())
    }

    fn init_runner(
        &mut self,
        runner_idx: usize,
    ) -> Result<(), SimError> {
        // first update the sigma values for all runners based on the weather conditions and their individual psi parameters
        let wbgt_c = self.get_wbgt(&self.input.runners[runner_idx])?.get::<degree_celsius>();
        // let wbgt_c = self.get_wbgt(runner_param)?.get::<degree_celsius>();
        let ref_temp_c = reference_temp_c();
        let temp_diff = (wbgt_c - ref_temp_c).max(0.0);
        let psi = self.input.runners[runner_idx].psi;
        self.input.runners[runner_idx].sigma *= 1.0 - psi * temp_diff;
        
        
        // also update the k value which is just the 2 times of gamma
        self.input.runners[runner_idx].k = Frequency::new::<hertz>(2.0 * self.input.runners[runner_idx].gamma.get::<hertz>());
        
        // if pacing strategy is even effort, calculate the constant acceleration
        if self.input.runners[runner_idx].pacing == PacingStrategy::EvenEffort {
            let (rho, cd, area, mass, const_v, tau) = {
                let r = &self.input.runners[runner_idx];
                (r.rho, r.drag_coefficient, r.frontal_area, r.mass, r.const_v, r.tau)
            };
            
            let f_resist = 0.5 * rho * cd * area * const_v * const_v / mass;
            let const_f = f_resist + (const_v / tau);
            
            self.input.runners[runner_idx].const_f = const_f;
        }

        Ok(())
    }

    fn parquet_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("runner_id", DataType::UInt32, false),
            Field::new("time_s", DataType::Float32, false),
            Field::new("velocity_mps", DataType::Float32, false),
            Field::new("energy_jpkg", DataType::Float32, false),
        ]))
    }

    fn make_parquet_writer(path: &str) -> Result<ArrowWriter<File>, SimError> {
        let file = File::create(path)
            .map_err(|_| SimError::InvalidValue("failed to create parquet file"))?;
        let props = WriterProperties::builder()
            .set_dictionary_enabled(true)
            .build();
        ArrowWriter::try_new(file, Self::parquet_schema(), Some(props))
            .map_err(|_| SimError::InvalidValue("failed to create parquet writer"))
    }

    fn flush_parquet_rows(
        &self,
        writer: &mut ArrowWriter<File>,
        results: &mut ResultBatch,
        // rows: &mut Vec<ResultRow>,
    ) -> Result<(), SimError> {
        // let runner_array = UInt32Array::from(rows.iter().map(|r| r.runner).collect::<Vec<u32>>());
        // let time_array = Float64Array::from(rows.iter().map(|r| r.time_s).collect::<Vec<f64>>());
        // let velocity_array = Float64Array::from(rows.iter().map(|r| r.velocity_mps).collect::<Vec<f64>>());
        // let energy_array = Float64Array::from(rows.iter().map(|r| r.energy_j_per_kg).collect::<Vec<f64>>());
        let (runner_col, time_col, velocity_col, energy_col) = results.take_columns();

        let runner_array = UInt32Array::from(runner_col);
        let time_array = Float32Array::from(time_col);
        let velocity_array = Float32Array::from(velocity_col);
        let energy_array = Float32Array::from(energy_col);
        let batch = RecordBatch::try_new(
            Self::parquet_schema(),
            vec![
                Arc::new(runner_array) as ArrayRef,
                Arc::new(time_array) as ArrayRef,
                Arc::new(velocity_array) as ArrayRef,
                Arc::new(energy_array) as ArrayRef,
            ],
        ).map_err(|_| SimError::InvalidValue("failed to create record batch"))?;

        writer.write(&batch)
            .map_err(|_| SimError::InvalidValue("failed to write record batch to parquet"))?;

        // results.clear();
        Ok(())
    }

    fn close_parquet_writer(writer: ArrowWriter<File>) -> Result<(), SimError> {
        writer.close()
            .map_err(|_| SimError::InvalidValue("failed to close parquet writer"))?;
        Ok(())
    }

    fn step_runner(&self, runner_params: &RunnerParams, state: &mut RunnerState) -> Result<(), SimError> {
        /*
        Use to perform one time step update for a given runner, calculating its current location and all the physics
         */
        let dt = self.input.config.dt;

        // determine the grade and headwind at the runner's current distance
        let grade = self.get_grade(state.distance)?;
        let headwind = self.get_headwind(state.distance)?;

        // determine the desired acceleration based on the pacing strategy and current state
        let f_desired = match runner_params.pacing {
            PacingStrategy::Constant => (runner_params.const_v - state.velocity)/dt,
            PacingStrategy::EvenEffort => runner_params.const_f,
        };

        // where the math model calculates monte carlo
        self.math_model(runner_params, state, f_desired, grade, headwind)?;
        

        Ok(())
    }

    fn math_model(&self,  runner_params: &RunnerParams, state: &mut RunnerState, f_desired: Acceleration, theta: f64, headwind: Velocity) -> Result<(), SimError> {
        /*
        Model the physics of the runner's movement for one time step, updating the velocity, energy, and distance in the state.
         */
        // define local variables for readability
        let (rho, cd, area, mass, e_init, tau, sigma, k, f_max) = {
            let r = runner_params;
            (r.rho, r.drag_coefficient, r.frontal_area, r.mass, r.e_init, r.tau, r.sigma, r.k, r.f_max)
        };

        let dt = self.input.config.dt;

        let current_v = state.velocity;
        let current_e = state.energy;
        let current_t = state.time_elapsed;
        let current_d = state.distance;

        // firt calculate all the resistive forces
        let v_rel = current_v + headwind; // relative velocity for drag calculation
        let f_resistance = Acceleration::new::<meter_per_second_squared>(gravity_mps2())*theta.sin()
                        + (0.5*rho*cd*area*v_rel*v_rel)/mass;

        // calculate the actual force applied by the runner, which is limited by the maximum thrust
        let f_possible = f_desired.min(f_max);

        // check if the runner has enough energy to apply the actual force
        let f_final =  if current_e > AvailableEnergy::new::<joule_per_kilogram>(0.0) {
            f_possible
        } else {
            // if not enough energy, calculate the maximum possible force based on remaining energy and power supply rate
            (sigma - (k*current_v*current_v*current_t)/tau) / current_v
        };

        // calculate the change in velocity and energy based on the actual force applied
        let dv = f_desired - f_resistance - (1.0/tau) * current_v;

        let de = if current_e > AvailableEnergy::new::<joule_per_kilogram>(0.0) {
            sigma - (f_final + f_resistance)*current_v - (k*current_v*current_v*current_t)/tau
        } else {
            SpecificPower::new::<watt_per_kilogram>(0.0)
        };

        // update velocity and energy for the next iteration
        let v_next = (current_v + dv*dt).max(Velocity::new::<meter_per_second>(0.0)); // velocity cannot be negative
        let e_next = if current_e + de*dt < AvailableEnergy::new::<joule_per_kilogram>(0.0) {
            AvailableEnergy::new::<joule_per_kilogram>(0.0)
        } else if current_e + de*dt > e_init {
            e_init
        } 
        else {
            current_e + de*dt
        };

        state.velocity = v_next;
        state.energy = e_next;

        // update distance and time
        state.distance = current_d + v_next * dt;
        state.time_elapsed = current_t + dt;

        Ok(())
    }

    fn get_grade(&self, distance: Length) -> Result<f64, SimError> {
        /*
        Helper function to get the grade at a given distance based on the course profile.
         */
        let course = &self.input.course;
        let closest_index = course.distance
        .iter()
        .enumerate() // Gives us (index, &value)
        .min_by(|(_, a), (_, b)| {
            let diff_a = (**a - distance).abs().get::<meter>();
            let diff_b = (**b - distance).abs().get::<meter>();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
        .map(|(index, _)| index)
        .ok_or(SimError::EmptyCourse("course distance_m must be non-empty"))?;
        
        // convert grade from percent to angle (radians) for physics calculations
        let decimal_grade = course.grade[closest_index] / 100.0;
        
        Ok(decimal_grade.atan())
    }

    fn get_headwind(&self, distance: Length) -> Result<Velocity, SimError> {
        /*
        Helper function to get the headwind at a given distance based on the course profile.
         */
        let course = &self.input.course;
        let closest_index = course.distance
        .iter()
        .enumerate() // Gives us (index, &value)
        .min_by(|(_, a), (_, b)| {
            let diff_a = (**a - distance).abs().get::<meter>();
            let diff_b = (**b - distance).abs().get::<meter>();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
        .map(|(index, _)| index)
        .ok_or(SimError::EmptyCourse("course distance_m must be non-empty"))?;

        Ok(course.headwind[closest_index])
    }

    fn get_wbgt(&self, runner: &RunnerParams) -> Result<ThermodynamicTemperature, SimError> {
        /*
        Helper function to calculate the Wet Bulb Globe Temperature (WBGT) based on the weather conditions.
         */
        let weather = &self.input.weather;

        // converrt units to dimensionless for empirical formula
        let temp_c = weather.temperature.get::<degree_celsius>();
        let humidity_percent = weather.humidity; // already in percentage
        let solar_w_m2 = weather.solar_radiation.get::<watt_per_square_meter>();
        let conv_w_m2k = runner.convection.get::<watt_per_square_meter_kelvin>();
        let alpha = runner.alpha;

        let temp_w = temp_c * (0.151977*((humidity_percent + 8.313659).powf(0.5))).atan()
            + (temp_c + humidity_percent).atan()
            - (humidity_percent - 1.676331).atan()
            + 0.00391838*(humidity_percent).powf(1.5) * (0.023101 * humidity_percent).atan()
            - 4.686035;
        let temp_g = temp_c + solar_w_m2 / (conv_w_m2k * alpha);
        let wbgt = 0.7*temp_w + 0.2*temp_g + 0.1*temp_c;

        Ok(ThermodynamicTemperature::new::<degree_celsius>(wbgt))

    }

}

