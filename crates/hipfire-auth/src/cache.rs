use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::{
    parse_token, AccessStore, AuthKind, Pepper, RatePolicyOverride, RequestPrincipal, TokenRecord,
    UserStatus,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CredentialError {
    #[error("invalid API credential")]
    Invalid,
    #[error("API credential expired")]
    Expired,
    #[error("API credential revoked")]
    Revoked,
    #[error("API user disabled")]
    UserDisabled,
}

#[derive(Debug, Clone)]
struct CachedCredential {
    token: TokenRecord,
    user_status: UserStatus,
    user_rate_policy: RatePolicyOverride,
}

/// Immutable credential data suitable for lock-free sharing through an `Arc`.
/// Mutations rebuild and atomically replace the whole snapshot at the server
/// boundary; verification itself performs no storage I/O.
#[derive(Debug, Clone)]
pub struct CredentialSnapshot {
    pepper: Pepper,
    by_token_id: Arc<HashMap<String, CachedCredential>>,
}

impl CredentialSnapshot {
    pub fn load(store: &AccessStore) -> Result<Self, crate::store::AuthError> {
        let users = store
            .list_users()?
            .into_iter()
            .map(|user| (user.id, (user.status, user.rate_policy)))
            .collect::<HashMap<_, _>>();
        let mut by_token_id = HashMap::new();
        for token in store.list_tokens()? {
            let Some((user_status, user_rate_policy)) = users.get(&token.user_id).cloned() else {
                return Err(crate::store::AuthError::Corrupt(format!(
                    "token {} references missing user",
                    token.id
                )));
            };
            by_token_id.insert(
                token.id.clone(),
                CachedCredential {
                    token,
                    user_status,
                    user_rate_policy,
                },
            );
        }
        Ok(Self {
            pepper: store.pepper().clone(),
            by_token_id: Arc::new(by_token_id),
        })
    }

    pub fn verify(&self, raw_token: &str, now: u64) -> Result<RequestPrincipal, CredentialError> {
        let (token_id, _) = parse_token(raw_token).ok_or(CredentialError::Invalid)?;
        let credential = self
            .by_token_id
            .get(token_id)
            .ok_or(CredentialError::Invalid)?;
        if !self.pepper.verify(raw_token, &credential.token.digest) {
            return Err(CredentialError::Invalid);
        }
        if credential.user_status == UserStatus::Disabled {
            return Err(CredentialError::UserDisabled);
        }
        if credential.token.revoked_at.is_some() {
            return Err(CredentialError::Revoked);
        }
        if credential.token.expires_at <= now {
            return Err(CredentialError::Expired);
        }
        Ok(RequestPrincipal {
            user_id: Some(credential.token.user_id.clone()),
            token_id: Some(credential.token.id.clone()),
            scopes: credential.token.scopes.clone(),
            auth_kind: AuthKind::ApiToken,
        })
    }

    pub fn len(&self) -> usize {
        self.by_token_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_token_id.is_empty()
    }

    pub fn rate_policies(
        &self,
        principal: &RequestPrincipal,
    ) -> Option<(RatePolicyOverride, RatePolicyOverride)> {
        let token_id = principal.token_id.as_deref()?;
        let credential = self.by_token_id.get(token_id)?;
        Some((
            credential.user_rate_policy.clone(),
            credential.token.rate_policy.clone(),
        ))
    }
}
