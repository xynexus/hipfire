// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Fixed-geometry BF16 Qwen3 SwiGLU on AIE2P.

use crate::{DeviceBuffer, NpuKernel, XdnaError};
use hipfire_primitives::conv::{bf16_bits_to_f32, f32_to_bf16_bits};
use std::sync::OnceLock;

pub struct NpuQwen3SwiGlu {
    rows: usize,
    intermediate_size: usize,
    backend: SwiGluBackend,
}

enum SwiGluBackend {
    Npu {
        kernel: NpuKernel,
        gate: DeviceBuffer,
        up: DeviceBuffer,
        output: DeviceBuffer,
    },
    Host,
}

impl NpuQwen3SwiGlu {
    pub fn load(
        xclbin: &[u8],
        instructions: &[u8],
        rows: usize,
        intermediate_size: usize,
    ) -> Result<Self, XdnaError> {
        let elements = validate_geometry(rows, intermediate_size)?;
        let kernel = NpuKernel::load(xclbin, instructions)?;
        let bytes = elements * size_of::<u16>();
        let gate = kernel.alloc_arg(bytes)?;
        let up = kernel.alloc_arg(bytes)?;
        let output = kernel.alloc_arg(bytes)?;
        Ok(Self {
            rows,
            intermediate_size,
            backend: SwiGluBackend::Npu {
                kernel,
                gate,
                up,
                output,
            },
        })
    }

    /// Use the host for the bandwidth-bound BF16 activation while projections
    /// and attention remain resident on AIE2P. The inputs are already host
    /// vectors at this component-image boundary, so this avoids a round trip
    /// through the scalar AIE2P sigmoid without adding a device transfer.
    pub fn load_host(rows: usize, intermediate_size: usize) -> Result<Self, XdnaError> {
        validate_geometry(rows, intermediate_size)?;
        Ok(Self {
            rows,
            intermediate_size,
            backend: SwiGluBackend::Host,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn run(&mut self, gate: &[u16], up: &[u16]) -> Result<Vec<u16>, XdnaError> {
        let elements = self.rows * self.intermediate_size;
        if gate.len() != elements || up.len() != elements {
            return Err(invalid(format!(
                "Qwen3 SwiGLU inputs must each contain {elements} BF16 values"
            )));
        }
        let SwiGluBackend::Npu {
            kernel,
            gate: gate_buffer,
            up: up_buffer,
            output,
        } = &mut self.backend
        else {
            return Ok(host_swiglu(gate, up));
        };
        encode_bf16(gate_buffer.as_mut_slice(), gate);
        encode_bf16(up_buffer.as_mut_slice(), up);
        kernel.recreate_hwctx()?;
        kernel.dispatch_synced(&[&*gate_buffer, &*up_buffer, &*output], &[true, true, true])?;
        kernel.sync_output(output)?;
        Ok(decode_bf16(output.as_slice()))
    }
}

fn validate_geometry(rows: usize, intermediate_size: usize) -> Result<usize, XdnaError> {
    let elements = rows
        .checked_mul(intermediate_size)
        .ok_or_else(|| invalid("Qwen3 SwiGLU geometry overflow"))?;
    if rows == 0 || rows > 4096 || intermediate_size == 0 || !elements.is_multiple_of(128) {
        return Err(invalid(
            "Qwen3 SwiGLU needs positive geometry and 128-aligned element count",
        ));
    }
    Ok(elements)
}

fn host_swiglu(gate: &[u16], up: &[u16]) -> Vec<u16> {
    static SILU_BF16: OnceLock<Box<[u16; 1 << 16]>> = OnceLock::new();
    let silu = SILU_BF16.get_or_init(|| {
        let mut table = Box::new([0u16; 1 << 16]);
        for (bits, output) in table.iter_mut().enumerate() {
            let value = bf16_bits_to_f32(bits as u16);
            *output = f32_to_bf16_bits(value / (1.0 + (-value).exp()));
        }
        table
    });
    gate.iter()
        .zip(up)
        .map(|(&gate, &up)| {
            f32_to_bf16_bits(bf16_bits_to_f32(silu[gate as usize]) * bf16_bits_to_f32(up))
        })
        .collect()
}

fn encode_bf16(destination: &mut [u8], values: &[u16]) {
    for (bytes, value) in destination.chunks_exact_mut(2).zip(values) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}

fn decode_bf16(source: &[u8]) -> Vec<u16> {
    source
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

fn invalid(message: impl Into<String>) -> XdnaError {
    XdnaError::InvalidOpus(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_swiglu_matches_bf16_rounding_contract() {
        let gate_values = [-12.0f32, -1.0, 0.0, 1.0, 12.0];
        let up_values = [3.0f32, -2.0, 0.5, 2.0, -3.0];
        let gate = gate_values.map(f32_to_bf16_bits);
        let up = up_values.map(f32_to_bf16_bits);
        let actual = host_swiglu(&gate, &up);
        let expected = gate_values
            .into_iter()
            .zip(up_values)
            .map(|(gate, up)| {
                let gate = bf16_bits_to_f32(f32_to_bf16_bits(gate));
                let up = bf16_bits_to_f32(f32_to_bf16_bits(up));
                let silu = bf16_bits_to_f32(f32_to_bf16_bits(gate / (1.0 + (-gate).exp())));
                f32_to_bf16_bits(silu * up)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
