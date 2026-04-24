use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use once_cell::sync::Lazy;

#[derive(Debug, Deserialize)]
struct ConstantsFile {
    physics: Physics,
}

#[derive(Debug, Deserialize)]
struct Physics {
    gravity: ConstantEntry,
    reference_temp: ConstantEntry,
}

#[derive(Debug, Deserialize)]
struct ConstantEntry {
    value: f64,
}


fn constants_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../config/constants.yml")
}


static CONSTANTS: Lazy<ConstantsFile> = Lazy::new(|| {
    let text = fs::read_to_string(constants_path())
        .expect("failed to read config/constants.yml");
    serde_yaml::from_str(&text)
        .expect("failed to parse constants.yml")
});

pub fn reference_temp_c() -> f64 {
    CONSTANTS.physics.reference_temp.value
}

pub fn gravity_mps2() -> f64 {
    CONSTANTS.physics.gravity.value
}