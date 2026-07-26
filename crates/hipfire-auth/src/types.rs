use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const DEFAULT_TOKEN_TTL_SECS: u64 = 90 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Text,
    Embeddings,
    Images,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserStatus {
    Enabled,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthKind {
    ApiToken,
    AnonymousLocal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPrincipal {
    pub user_id: Option<String>,
    pub token_id: Option<String>,
    pub scopes: BTreeSet<Scope>,
    pub auth_kind: AuthKind,
}

impl RequestPrincipal {
    pub fn anonymous_local() -> Self {
        Self {
            user_id: None,
            token_id: None,
            scopes: BTreeSet::from([
                Scope::Text,
                Scope::Embeddings,
                Scope::Images,
                Scope::Training,
            ]),
            auth_kind: AuthKind::AnonymousLocal,
        }
    }

    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RatePolicyOverride {
    pub requests_per_minute: Option<u64>,
    pub request_burst: Option<u64>,
    pub text_tokens_per_minute: Option<u64>,
    pub text_token_burst: Option<u64>,
    pub max_in_flight_text: Option<u32>,
    pub max_in_flight_images: Option<u32>,
    pub megapixel_steps_per_minute: Option<u64>,
    pub megapixel_step_burst: Option<u64>,
    pub max_in_flight_training: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserRecord {
    pub id: String,
    pub name: String,
    pub status: UserStatus,
    pub rate_policy: RatePolicyOverride,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewUser {
    pub name: String,
    #[serde(default)]
    pub rate_policy: RatePolicyOverride,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRecord {
    pub id: String,
    pub user_id: String,
    pub label: String,
    pub scopes: BTreeSet<Scope>,
    pub digest: [u8; 32],
    pub rate_policy: RatePolicyOverride,
    pub created_at: u64,
    pub expires_at: u64,
    pub revoked_at: Option<u64>,
}

/// Token creation result. `secret` is serializable for the single admin API
/// response, but deliberately redacted from `Debug` output and never accepted
/// by any persistence method.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct CreatedToken {
    pub token: TokenRecord,
    pub secret: String,
}

impl fmt::Debug for CreatedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CreatedToken")
            .field("token", &self.token)
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewToken {
    pub label: String,
    pub scopes: BTreeSet<Scope>,
    #[serde(default)]
    pub rate_policy: RatePolicyOverride,
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageCounters {
    pub requests: u64,
    pub errors: u64,
    pub rate_limit_hits: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_tokens: u64,
    pub images: u64,
    pub megapixel_steps: u64,
    pub training_seconds: u64,
}

impl std::ops::AddAssign for UsageCounters {
    fn add_assign(&mut self, rhs: Self) {
        self.requests = self.requests.saturating_add(rhs.requests);
        self.errors = self.errors.saturating_add(rhs.errors);
        self.rate_limit_hits = self.rate_limit_hits.saturating_add(rhs.rate_limit_hits);
        self.input_tokens = self.input_tokens.saturating_add(rhs.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(rhs.output_tokens);
        self.cache_tokens = self.cache_tokens.saturating_add(rhs.cache_tokens);
        self.images = self.images.saturating_add(rhs.images);
        self.megapixel_steps = self.megapixel_steps.saturating_add(rhs.megapixel_steps);
        self.training_seconds = self.training_seconds.saturating_add(rhs.training_seconds);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HourlyUsageRecord {
    pub hour_start: u64,
    pub user_id: String,
    pub token_id: String,
    pub workload: String,
    pub counters: UsageCounters,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseContextRecord {
    pub user_id: String,
    pub response_id: String,
    pub parent_response_id: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub expires_at: u64,
    /// Opaque server-owned serialized context. This is conversation state, not
    /// usage telemetry, and is bounded before insertion by the caller/store.
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub sequence: u64,
    pub created_at: u64,
    pub actor: String,
    pub action: String,
    pub user_id: Option<String>,
    pub token_id: Option<String>,
    pub detail: Option<String>,
}
