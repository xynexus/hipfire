//! Minimal `.npy` reader/writer for the DFlash native-driver examples.
//!
//! The Python harness is the parity reference, so the native driver has to eat
//! the exact arrays it produced (quantized int8 operands, int32 GEMM results,
//! and the f32 Phase-A golden). Only the subset of the format numpy actually
//! emits here is handled: version 1.0/2.0, C-contiguous, little-endian, no
//! pickled objects.
#![allow(dead_code)]

#[derive(Debug)]
pub struct Npy {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_i8(&self) -> &[i8] {
        assert!(self.dtype.ends_with("i1"), "expected int8, got {}", self.dtype);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i8, self.elems()) }
    }

    pub fn as_i32(&self) -> &[i32] {
        assert!(self.dtype.ends_with("i4"), "expected int32, got {}", self.dtype);
        assert_eq!(self.data.as_ptr() as usize % 4, 0, "misaligned i32 payload");
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i32, self.elems()) }
    }

    /// f32 view. numpy may hand us f64 goldens, so widen/narrow as needed.
    pub fn to_f32(&self) -> Vec<f32> {
        let n = self.elems();
        if self.dtype.ends_with("f4") {
            let s = unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, n) };
            s.to_vec()
        } else if self.dtype.ends_with("f8") {
            let s = unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f64, n) };
            s.iter().map(|&v| v as f32).collect()
        } else {
            panic!("expected float array, got {}", self.dtype);
        }
    }
}

pub fn read(path: &str) -> std::io::Result<Npy> {
    let raw = std::fs::read(path)?;
    assert!(raw.len() > 10 && &raw[0..6] == b"\x93NUMPY", "{path}: not a .npy file");
    let major = raw[6];
    // v1.0 uses a 2-byte header length, v2.0+ a 4-byte one.
    let (hdr_len, hdr_start) = if major == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&raw[hdr_start..hdr_start + hdr_len])
        .expect("npy header utf8")
        .to_string();

    let dtype = quoted(extract(&header, "'descr':")).to_string();
    assert!(
        !dtype.starts_with('>'),
        "{path}: big-endian arrays unsupported ({dtype})"
    );
    let fortran = extract(&header, "'fortran_order':");
    assert!(
        fortran.starts_with("False"),
        "{path}: Fortran-order arrays unsupported"
    );

    let shape_str = extract(&header, "'shape':");
    let shape: Vec<usize> = shape_str
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("shape dim"))
        .collect();

    Ok(Npy {
        dtype,
        shape,
        data: raw[hdr_start + hdr_len..].to_vec(),
    })
}

/// Pull everything following `key` out of the npy header dict literal. The
/// caller narrows it to the actual value (the header is a Python dict literal,
/// so values end at a quote, a paren, or a comma depending on the field).
fn extract<'a>(header: &'a str, key: &str) -> &'a str {
    let at = header.find(key).unwrap_or_else(|| panic!("npy header missing {key}"));
    header[at + key.len()..].trim_start()
}

/// First single-quoted token of `s` (e.g. `'<f4', 'fortran_order': ...` -> `<f4`).
fn quoted(s: &str) -> &str {
    let rest = s.strip_prefix('\'').expect("expected quoted npy header value");
    &rest[..rest.find('\'').expect("unterminated npy header value")]
}

/// Write a 1-D or 2-D f32 array as a v1.0 `.npy` (so Python can diff results).
pub fn write_f32(path: &str, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        )
    };
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    // The header (magic + len field included) must be 64-byte aligned.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}
