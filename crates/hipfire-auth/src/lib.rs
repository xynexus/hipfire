//! Durable API access primitives for hipfire.
//!
//! The crate deliberately has no dependency on the HTTP server or inference
//! runtime. Blocking redb transactions live here; callers publish immutable
//! [`CredentialSnapshot`] values to the request hot path.

mod cache;
mod crypto;
mod rate_limit;
mod store;
mod types;

pub use cache::{CredentialError, CredentialSnapshot};
pub use crypto::{parse_token, Pepper, TOKEN_PREFIX};
pub use rate_limit::*;
pub use store::{AccessStore, AuthError, StorePaths, CURRENT_SCHEMA_VERSION};
pub use types::*;
