use std::path::PathBuf;

use hipfire_config::{configured_models_dir, hipfire_dir, HipfireConfig};
use hipfire_model::find_model_in;

pub fn find_model(arg: &str, config: &HipfireConfig) -> Option<PathBuf> {
    let mdir = configured_models_dir(config);
    let aliases = hipfire_dir().join("models.json");
    find_model_in(arg, &mdir, Some(&aliases))
}
