// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone `hipfire-coexistence` executable — see `hipfire_coexistence::cli`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    hipfire_coexistence::cli::main()
}
