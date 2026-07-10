//! M-parallel sparse-three residual projection for compact mixed Opus formats.
#![cfg(target_os = "linux")]

use crate::{DeviceBuffer, NpuKernel, XdnaError};

const K: usize = 256;
const MR: usize = 4;
const NT: usize = 4;
const MN: usize = 16;
const BYTES_PER_COLUMN: usize = 6;

/// A resident AIE2P kernel computing three sparse residual products per output column.
pub struct NpuSparse3Mp {
    kernel: NpuKernel,
    cols: usize,
    mt: usize,
    nb: usize,
    a_buf: DeviceBuffer,
    w_buf: DeviceBuffer,
    c_buf: DeviceBuffer,
    w_loaded: bool,
}

impl NpuSparse3Mp {
    pub fn load_cached(dir: &str) -> Result<Self, XdnaError> {
        let xclbin = std::fs::read(format!("{dir}/final.xclbin")).map_err(XdnaError::Open)?;
        let insts = std::fs::read(format!("{dir}/insts.bin")).map_err(XdnaError::Open)?;
        let base = std::path::Path::new(dir)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let tokens: Vec<&str> = base.split('_').collect();
        let bad = || XdnaError::BadCacheName(base.to_string());
        if !tokens.contains(&"sparse3") {
            return Err(bad());
        }
        let prefixed = |prefix: &str| {
            tokens.iter().find_map(|token| {
                token
                    .strip_prefix(prefix)
                    .and_then(|rest| rest.parse().ok())
            })
        };
        let nb = prefixed("nb").ok_or_else(bad)?;
        let cols = prefixed("c").ok_or_else(bad)?;
        let dims = tokens
            .iter()
            .find(|token| token.split('x').count() == 3)
            .ok_or_else(bad)?;
        let dims: Vec<usize> = dims
            .split('x')
            .filter_map(|part| part.parse().ok())
            .collect();
        if dims.len() != 3 || dims[1] != NT || dims[2] != 16 {
            return Err(bad());
        }
        let mt = dims[0];
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let a_buf = kernel.alloc_arg(cols * mt * MR * K)?;
        let w_buf = kernel.alloc_arg(nb * NT * MN * BYTES_PER_COLUMN)?;
        let c_buf = kernel.alloc_arg(cols * nb * mt * MR * NT * MN * 4)?;
        Ok(Self {
            kernel,
            cols,
            mt,
            nb,
            a_buf,
            w_buf,
            c_buf,
            w_loaded: false,
        })
    }

    pub fn rows_per_dispatch(&self) -> usize {
        self.cols * self.mt * MR
    }

    pub fn k(&self) -> usize {
        K
    }

    pub fn n(&self) -> usize {
        self.nb * NT * MN
    }

    /// Load `[N,3,(u8 index,i8 delta)]` bytes, contiguous by output column.
    pub fn load_weights(&mut self, sparse: &[u8]) {
        assert_eq!(sparse.len(), self.n() * BYTES_PER_COLUMN);
        self.w_buf.as_mut_slice().copy_from_slice(sparse);
        self.w_loaded = true;
    }

    pub fn run(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        activations: &[i8],
        output: &mut [i32],
    ) -> Result<(), XdnaError> {
        assert!(self.w_loaded, "call load_weights before run");
        assert_eq!(k, K);
        assert_eq!(n, self.n());
        assert_eq!(activations.len(), m * K);
        assert_eq!(output.len(), m * n);
        let rows_per = self.rows_per_dispatch();
        assert_eq!(m % rows_per, 0);
        for dispatch in 0..m / rows_per {
            let row0 = dispatch * rows_per;
            let a = self.a_buf.as_mut_slice();
            for core in 0..self.cols {
                for local_row in 0..self.mt * MR {
                    let source_row = row0 + core * self.mt * MR + local_row;
                    let source = source_row * K;
                    let destination = (core * self.mt * MR + local_row) * K;
                    for inner in 0..K {
                        a[destination + inner] = activations[source + inner] as u8;
                    }
                }
            }
            self.kernel
                .dispatch(&[&self.a_buf, &self.w_buf, &self.c_buf])?;
            self.read_output(row0, n, output);
        }
        Ok(())
    }

    fn read_output(&self, row0: usize, n: usize, output: &mut [i32]) {
        let block_rows = self.mt * MR;
        let block_cols = NT * MN;
        let block_elements = block_rows * block_cols;
        let device_output: &[i32] = unsafe {
            std::slice::from_raw_parts(
                self.c_buf.as_slice().as_ptr() as *const i32,
                self.cols * self.nb * block_elements,
            )
        };
        for core in 0..self.cols {
            for slab in 0..self.nb {
                for local_row in 0..block_rows {
                    let source = (core * self.nb + slab) * block_elements + local_row * block_cols;
                    let destination =
                        (row0 + core * block_rows + local_row) * n + slab * block_cols;
                    output[destination..destination + block_cols]
                        .copy_from_slice(&device_output[source..source + block_cols]);
                }
            }
        }
    }
}
