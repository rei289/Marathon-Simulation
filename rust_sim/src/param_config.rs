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

#[allow(dead_code)]
pub fn get_min_max_values(category: &str, name: &str) -> Result<(f64, f64), String> {
    // read the config/parameters.yml file and return the min and max values for the given parameter name
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../config/parameters.yml");
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("read {} failed: {}", path.display(), e))?;
    let params: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| format!("parse parameters.yml failed: {}", e))?;

    let obj = params.get(category)
        .and_then(|v| v.get(name))
        .ok_or_else(|| format!("missing path: {} -> {}", category, name))?;

    let min = obj.get("min")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{} -> {} -> min is missing or not numeric", category, name))?;

    let max = obj.get("max")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("{} -> {} -> max is missing or not numeric", category, name))?;

    Ok((min, max))
}