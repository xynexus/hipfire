use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::store::AuthError;

pub const TOKEN_PREFIX: &str = "hfr";
const PEPPER_LEN: usize = 32;
const TOKEN_SECRET_LEN: usize = 32;

#[derive(Clone)]
pub struct Pepper([u8; PEPPER_LEN]);

impl std::fmt::Debug for Pepper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Pepper([REDACTED])")
    }
}

impl Pepper {
    pub fn read_or_create(path: &Path) -> Result<Self, AuthError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
            set_dir_private(parent)?;
        }

        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => {
                set_file_private(path)?;
                let mut bytes = [0u8; PEPPER_LEN];
                getrandom::getrandom(&mut bytes)
                    .map_err(|error| AuthError::Random(error.to_string()))?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                Ok(Self(bytes))
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                set_file_private(path)?;
                let mut file = OpenOptions::new().read(true).open(path)?;
                let mut bytes = Vec::new();
                file.read_to_end(&mut bytes)?;
                let bytes: [u8; PEPPER_LEN] = bytes.try_into().map_err(|_| {
                    AuthError::Corrupt("token pepper must contain exactly 32 bytes".into())
                })?;
                Ok(Self(bytes))
            }
            Err(error) => Err(error.into()),
        }
    }

    pub fn digest(&self, raw_token: &str) -> [u8; 32] {
        hmac_sha256::HMAC::mac(raw_token.as_bytes(), self.0)
    }

    pub fn verify(&self, raw_token: &str, expected: &[u8; 32]) -> bool {
        hmac_sha256::HMAC::verify(raw_token.as_bytes(), self.0, expected)
    }
}

pub(crate) fn mint_token(token_id: &str, pepper: &Pepper) -> Result<(String, [u8; 32]), AuthError> {
    let mut secret = [0u8; TOKEN_SECRET_LEN];
    getrandom::getrandom(&mut secret).map_err(|error| AuthError::Random(error.to_string()))?;
    let raw = format!(
        "{TOKEN_PREFIX}_{token_id}_{}",
        URL_SAFE_NO_PAD.encode(secret)
    );
    let digest = pepper.digest(&raw);
    Ok((raw, digest))
}

pub fn parse_token(raw: &str) -> Option<(&str, &str)> {
    let mut parts = raw.split('_');
    if parts.next()? != TOKEN_PREFIX {
        return None;
    }
    let token_id = parts.next()?;
    let secret = parts.next()?;
    if parts.next().is_some()
        || token_id.len() != 32
        || !token_id.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return None;
    }
    let decoded = URL_SAFE_NO_PAD.decode(secret).ok()?;
    (decoded.len() == TOKEN_SECRET_LEN).then_some((token_id, secret))
}

#[cfg(unix)]
fn set_file_private(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_file_private(_path: &Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
pub(crate) fn set_dir_private(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
}

#[cfg(not(unix))]
pub(crate) fn set_dir_private(_path: &Path) -> Result<(), std::io::Error> {
    Ok(())
}

#[cfg(unix)]
pub(crate) fn create_private_file(path: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs::OpenOptionsExt;
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(file) => {
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => set_file_private(path),
        Err(error) => Err(error),
    }
}

#[cfg(not(unix))]
pub(crate) fn create_private_file(path: &Path) -> Result<(), std::io::Error> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => {
            file.sync_all()?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}
