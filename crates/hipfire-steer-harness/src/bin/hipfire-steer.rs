// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone `hipfire-steer` executable — see `hipfire_steer_harness::cli`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    hipfire_steer_harness::cli::main()
}
