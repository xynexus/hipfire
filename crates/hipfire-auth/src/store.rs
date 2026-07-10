use std::path::{Path, PathBuf};
use std::sync::Arc;

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;

use crate::crypto::{create_private_file, mint_token, set_dir_private, Pepper};
use crate::{
    AuditEvent, CreatedToken, HourlyUsageRecord, NewToken, NewUser, ResponseContextRecord,
    TokenRecord, UserRecord, UserStatus, DEFAULT_TOKEN_TTL_SECS,
};

pub const CURRENT_SCHEMA_VERSION: u64 = 1;
pub const MAX_RESPONSE_CONTEXT_BYTES: usize = 2 * 1024 * 1024;

const META: TableDefinition<&str, u64> = TableDefinition::new("meta_v1");
const USERS: TableDefinition<&str, &[u8]> = TableDefinition::new("users_v1");
const USER_NAMES: TableDefinition<&str, &[u8]> = TableDefinition::new("user_names_v1");
const TOKENS: TableDefinition<&str, &[u8]> = TableDefinition::new("tokens_v1");
const USER_TOKENS: TableDefinition<&str, &[u8]> = TableDefinition::new("user_tokens_v1");
const TOKEN_EXPIRY: TableDefinition<&str, &[u8]> = TableDefinition::new("token_expiry_v1");
const USAGE_HOURLY: TableDefinition<&str, &[u8]> = TableDefinition::new("usage_hourly_v1");
const RESPONSES: TableDefinition<&str, &[u8]> = TableDefinition::new("responses_v1");
const RESPONSE_EXPIRY: TableDefinition<&str, &[u8]> = TableDefinition::new("response_expiry_v1");
const AUDIT: TableDefinition<u64, &[u8]> = TableDefinition::new("audit_v1");

const SCHEMA_VERSION_KEY: &str = "schema_version";
const NEXT_AUDIT_SEQUENCE_KEY: &str = "next_audit_sequence";

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("access database error: {0}")]
    Database(String),
    #[error("access database is corrupt: {0}")]
    Corrupt(String),
    #[error("access schema {found} is newer than supported schema {supported}")]
    UnsupportedSchema { found: u64, supported: u64 },
    #[error("user name already exists")]
    DuplicateUserName,
    #[error("user not found")]
    UserNotFound,
    #[error("token not found")]
    TokenNotFound,
    #[error("invalid access record: {0}")]
    Invalid(String),
    #[error("secure random generation failed: {0}")]
    Random(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct StorePaths {
    pub database: PathBuf,
    pub pepper: PathBuf,
}

impl StorePaths {
    pub fn in_hipfire_dir(directory: impl AsRef<Path>) -> Self {
        let directory = directory.as_ref();
        Self {
            database: directory.join("access.redb"),
            pepper: directory.join("access.pepper"),
        }
    }
}

#[derive(Clone)]
pub struct AccessStore {
    database: Arc<Database>,
    pepper: Pepper,
    paths: StorePaths,
}

impl std::fmt::Debug for AccessStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessStore")
            .field("database", &self.paths.database)
            .field("pepper", &"[REDACTED]")
            .finish()
    }
}

impl AccessStore {
    pub fn open(paths: StorePaths) -> Result<Self, AuthError> {
        if let Some(parent) = paths.database.parent() {
            std::fs::create_dir_all(parent)?;
            set_dir_private(parent)?;
        }
        create_private_file(&paths.database)?;
        let pepper = Pepper::read_or_create(&paths.pepper)?;
        let database = Database::create(&paths.database).map_err(db)?;
        let store = Self {
            database: Arc::new(database),
            pepper,
            paths,
        };
        store.migrate()?;
        store.validate_indexes()?;
        Ok(store)
    }

    pub fn open_in(directory: impl AsRef<Path>) -> Result<Self, AuthError> {
        Self::open(StorePaths::in_hipfire_dir(directory))
    }

    pub fn paths(&self) -> &StorePaths {
        &self.paths
    }

    pub fn pepper(&self) -> &Pepper {
        &self.pepper
    }

    pub fn schema_version(&self) -> Result<u64, AuthError> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(META).map_err(db)?;
        Ok(table
            .get(SCHEMA_VERSION_KEY)
            .map_err(db)?
            .map(|value| value.value())
            .unwrap_or(0))
    }

    fn migrate(&self) -> Result<(), AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let found = {
            let table = write.open_table(META).map_err(db)?;
            let found = table
                .get(SCHEMA_VERSION_KEY)
                .map_err(db)?
                .map(|value| value.value())
                .unwrap_or(0);
            found
        };
        if found > CURRENT_SCHEMA_VERSION {
            return Err(AuthError::UnsupportedSchema {
                found,
                supported: CURRENT_SCHEMA_VERSION,
            });
        }
        let mut version = found;
        if version == 0 {
            // Opening a table is the redb schema creation operation. Keep every
            // table versioned so later migrations can coexist during rewrites.
            drop(write.open_table(USERS).map_err(db)?);
            drop(write.open_table(USER_NAMES).map_err(db)?);
            drop(write.open_table(TOKENS).map_err(db)?);
            drop(write.open_table(USER_TOKENS).map_err(db)?);
            drop(write.open_table(TOKEN_EXPIRY).map_err(db)?);
            drop(write.open_table(USAGE_HOURLY).map_err(db)?);
            drop(write.open_table(RESPONSES).map_err(db)?);
            drop(write.open_table(RESPONSE_EXPIRY).map_err(db)?);
            drop(write.open_table(AUDIT).map_err(db)?);
            let mut meta = write.open_table(META).map_err(db)?;
            meta.insert(NEXT_AUDIT_SEQUENCE_KEY, 1).map_err(db)?;
            meta.insert(SCHEMA_VERSION_KEY, 1).map_err(db)?;
            version = 1;
        }
        debug_assert_eq!(version, CURRENT_SCHEMA_VERSION);
        write.commit().map_err(db)
    }

    pub fn create_user(&self, new: NewUser, now: u64) -> Result<UserRecord, AuthError> {
        let name = new.name.trim();
        if name.is_empty() || name.len() > 128 {
            return Err(AuthError::Invalid(
                "user name must contain 1 to 128 characters".into(),
            ));
        }
        let normalized = normalize_name(name);
        let write = self.database.begin_write().map_err(db)?;
        {
            let names = write.open_table(USER_NAMES).map_err(db)?;
            if names.get(normalized.as_str()).map_err(db)?.is_some() {
                return Err(AuthError::DuplicateUserName);
            }
        }
        let user = UserRecord {
            id: uuid::Uuid::new_v4().simple().to_string(),
            name: name.to_string(),
            status: UserStatus::Enabled,
            rate_policy: new.rate_policy,
            created_at: now,
            updated_at: now,
        };
        let encoded = encode(&user)?;
        {
            write
                .open_table(USERS)
                .map_err(db)?
                .insert(user.id.as_str(), encoded.as_slice())
                .map_err(db)?;
            write
                .open_table(USER_NAMES)
                .map_err(db)?
                .insert(normalized.as_str(), user.id.as_bytes())
                .map_err(db)?;
        }
        append_audit_in_tx(
            &write,
            now,
            "admin",
            "user.created",
            Some(&user.id),
            None,
            Some(&user.name),
        )?;
        write.commit().map_err(db)?;
        Ok(user)
    }

    pub fn get_user(&self, user_id: &str) -> Result<Option<UserRecord>, AuthError> {
        self.get_json(USERS, user_id)
    }

    pub fn list_users(&self) -> Result<Vec<UserRecord>, AuthError> {
        self.list_json(USERS)
    }

    pub fn set_user_status(
        &self,
        user_id: &str,
        status: UserStatus,
        now: u64,
    ) -> Result<UserRecord, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let mut user: UserRecord =
            get_json_in_tx(&write, USERS, user_id)?.ok_or(AuthError::UserNotFound)?;
        user.status = status;
        user.updated_at = now;
        let encoded = encode(&user)?;
        write
            .open_table(USERS)
            .map_err(db)?
            .insert(user_id, encoded.as_slice())
            .map_err(db)?;
        append_audit_in_tx(
            &write,
            now,
            "admin",
            match status {
                UserStatus::Enabled => "user.enabled",
                UserStatus::Disabled => "user.disabled",
            },
            Some(user_id),
            None,
            None,
        )?;
        write.commit().map_err(db)?;
        Ok(user)
    }

    pub fn set_user_rate_policy(
        &self,
        user_id: &str,
        policy: crate::RatePolicyOverride,
        now: u64,
    ) -> Result<UserRecord, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let mut user: UserRecord =
            get_json_in_tx(&write, USERS, user_id)?.ok_or(AuthError::UserNotFound)?;
        user.rate_policy = policy;
        user.updated_at = now;
        let encoded = encode(&user)?;
        write
            .open_table(USERS)
            .map_err(db)?
            .insert(user_id, encoded.as_slice())
            .map_err(db)?;
        append_audit_in_tx(
            &write,
            now,
            "admin",
            "user.rate_policy.updated",
            Some(user_id),
            None,
            None,
        )?;
        write.commit().map_err(db)?;
        Ok(user)
    }

    pub fn issue_token(
        &self,
        user_id: &str,
        new: NewToken,
        now: u64,
    ) -> Result<CreatedToken, AuthError> {
        if new.label.trim().is_empty() || new.label.len() > 128 {
            return Err(AuthError::Invalid(
                "token label must contain 1 to 128 characters".into(),
            ));
        }
        if new.scopes.is_empty() {
            return Err(AuthError::Invalid(
                "token must contain at least one scope".into(),
            ));
        }
        let user = self.get_user(user_id)?.ok_or(AuthError::UserNotFound)?;
        if user.status == UserStatus::Disabled {
            return Err(AuthError::Invalid(
                "cannot issue a token for a disabled user".into(),
            ));
        }
        let expires_at = new
            .expires_at
            .unwrap_or_else(|| now.saturating_add(DEFAULT_TOKEN_TTL_SECS));
        if expires_at <= now {
            return Err(AuthError::Invalid(
                "token expiry must be in the future".into(),
            ));
        }
        let token_id = uuid::Uuid::new_v4().simple().to_string();
        let (secret, digest) = mint_token(&token_id, &self.pepper)?;
        let token = TokenRecord {
            id: token_id,
            user_id: user_id.to_string(),
            label: new.label.trim().to_string(),
            scopes: new.scopes,
            digest,
            rate_policy: new.rate_policy,
            created_at: now,
            expires_at,
            revoked_at: None,
        };
        let encoded = encode(&token)?;
        let write = self.database.begin_write().map_err(db)?;
        write
            .open_table(TOKENS)
            .map_err(db)?
            .insert(token.id.as_str(), encoded.as_slice())
            .map_err(db)?;
        let user_token_key = user_token_key(user_id, &token.id);
        write
            .open_table(USER_TOKENS)
            .map_err(db)?
            .insert(user_token_key.as_str(), token.id.as_bytes())
            .map_err(db)?;
        let expiry_key = expiry_key(expires_at, &token.id);
        write
            .open_table(TOKEN_EXPIRY)
            .map_err(db)?
            .insert(expiry_key.as_str(), token.id.as_bytes())
            .map_err(db)?;
        append_audit_in_tx(
            &write,
            now,
            "admin",
            "token.created",
            Some(user_id),
            Some(&token.id),
            Some(&token.label),
        )?;
        write.commit().map_err(db)?;
        Ok(CreatedToken { token, secret })
    }

    pub fn get_token(&self, token_id: &str) -> Result<Option<TokenRecord>, AuthError> {
        self.get_json(TOKENS, token_id)
    }

    pub fn list_tokens(&self) -> Result<Vec<TokenRecord>, AuthError> {
        self.list_json(TOKENS)
    }

    pub fn list_user_tokens(&self, user_id: &str) -> Result<Vec<TokenRecord>, AuthError> {
        let prefix = format!("{user_id}\0");
        let token_ids = {
            let read = self.database.begin_read().map_err(db)?;
            let index = read.open_table(USER_TOKENS).map_err(db)?;
            let mut token_ids = Vec::new();
            for entry in index.iter().map_err(db)? {
                let (key, value) = entry.map_err(db)?;
                if key.value().starts_with(&prefix) {
                    token_ids.push(
                        std::str::from_utf8(value.value())
                            .map_err(|_| {
                                AuthError::Corrupt("invalid user-token index value".into())
                            })?
                            .to_string(),
                    );
                }
            }
            token_ids
        };
        token_ids
            .into_iter()
            .map(|token_id| {
                self.get_token(&token_id)?.ok_or_else(|| {
                    AuthError::Corrupt(format!(
                        "user-token index references missing token {token_id}"
                    ))
                })
            })
            .collect()
    }

    pub fn revoke_token(&self, token_id: &str, now: u64) -> Result<TokenRecord, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let mut token: TokenRecord =
            get_json_in_tx(&write, TOKENS, token_id)?.ok_or(AuthError::TokenNotFound)?;
        if token.revoked_at.is_none() {
            token.revoked_at = Some(now);
            let encoded = encode(&token)?;
            write
                .open_table(TOKENS)
                .map_err(db)?
                .insert(token_id, encoded.as_slice())
                .map_err(db)?;
            append_audit_in_tx(
                &write,
                now,
                "admin",
                "token.revoked",
                Some(&token.user_id),
                Some(token_id),
                None,
            )?;
        }
        write.commit().map_err(db)?;
        Ok(token)
    }

    pub fn add_usage(&self, record: &HourlyUsageRecord) -> Result<(), AuthError> {
        if record.hour_start % 3600 != 0 {
            return Err(AuthError::Invalid(
                "usage hour_start must be aligned to one hour".into(),
            ));
        }
        let key = usage_key(record);
        let write = self.database.begin_write().map_err(db)?;
        let merged = match get_json_in_tx::<HourlyUsageRecord>(&write, USAGE_HOURLY, &key)? {
            Some(mut current) => {
                current.counters += record.counters;
                current
            }
            None => record.clone(),
        };
        let encoded = encode(&merged)?;
        write
            .open_table(USAGE_HOURLY)
            .map_err(db)?
            .insert(key.as_str(), encoded.as_slice())
            .map_err(db)?;
        write.commit().map_err(db)
    }

    pub fn list_usage(&self) -> Result<Vec<HourlyUsageRecord>, AuthError> {
        self.list_json(USAGE_HOURLY)
    }

    pub fn prune_usage_before(&self, hour_start: u64) -> Result<u64, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let keys = {
            let table = write.open_table(USAGE_HOURLY).map_err(db)?;
            let mut keys = Vec::new();
            for entry in table.iter().map_err(db)? {
                let (key, value) = entry.map_err(db)?;
                let record: HourlyUsageRecord = decode(value.value())?;
                if record.hour_start < hour_start {
                    keys.push(key.value().to_string());
                }
            }
            keys
        };
        let mut table = write.open_table(USAGE_HOURLY).map_err(db)?;
        for key in &keys {
            table.remove(key.as_str()).map_err(db)?;
        }
        drop(table);
        write.commit().map_err(db)?;
        Ok(keys.len() as u64)
    }

    pub fn put_response(&self, record: &ResponseContextRecord) -> Result<(), AuthError> {
        self.put_response_bounded(record, usize::MAX)
    }

    pub fn put_response_bounded(
        &self,
        record: &ResponseContextRecord,
        max_per_user: usize,
    ) -> Result<(), AuthError> {
        if record.payload.len() > MAX_RESPONSE_CONTEXT_BYTES {
            return Err(AuthError::Invalid(format!(
                "response context exceeds {MAX_RESPONSE_CONTEXT_BYTES} bytes"
            )));
        }
        if max_per_user == 0 {
            return Err(AuthError::Invalid(
                "response context capacity must be non-zero".into(),
            ));
        }
        let key = response_key(&record.user_id, &record.response_id);
        let expiry = response_expiry_key(record.expires_at, &record.user_id, &record.response_id);
        let encoded = encode(record)?;
        let write = self.database.begin_write().map_err(db)?;
        let evicted = {
            let table = write.open_table(RESPONSES).map_err(db)?;
            let mut owned = Vec::new();
            for entry in table.iter().map_err(db)? {
                let (stored_key, value) = entry.map_err(db)?;
                let stored: ResponseContextRecord = decode(value.value())?;
                if stored.user_id == record.user_id && stored_key.value() != key {
                    owned.push((stored.created_at, stored_key.value().to_string()));
                }
            }
            owned.sort_by(|left, right| left.cmp(right));
            let evict = owned.len().saturating_add(1).saturating_sub(max_per_user);
            owned
                .into_iter()
                .take(evict)
                .map(|(_, key)| key)
                .collect::<Vec<_>>()
        };
        {
            let mut table = write.open_table(RESPONSES).map_err(db)?;
            for key in evicted {
                table.remove(key.as_str()).map_err(db)?;
            }
        }
        write
            .open_table(RESPONSES)
            .map_err(db)?
            .insert(key.as_str(), encoded.as_slice())
            .map_err(db)?;
        write
            .open_table(RESPONSE_EXPIRY)
            .map_err(db)?
            .insert(expiry.as_str(), key.as_bytes())
            .map_err(db)?;
        write.commit().map_err(db)
    }

    pub fn get_response(
        &self,
        user_id: &str,
        response_id: &str,
    ) -> Result<Option<ResponseContextRecord>, AuthError> {
        self.get_json(RESPONSES, &response_key(user_id, response_id))
    }

    pub fn list_user_responses(
        &self,
        user_id: &str,
    ) -> Result<Vec<ResponseContextRecord>, AuthError> {
        Ok(self
            .list_json::<ResponseContextRecord>(RESPONSES)?
            .into_iter()
            .filter(|record| record.user_id == user_id)
            .collect())
    }

    pub fn prune_responses_expired(&self, now: u64) -> Result<u64, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let expired = {
            let table = write.open_table(RESPONSES).map_err(db)?;
            let mut expired = Vec::new();
            for entry in table.iter().map_err(db)? {
                let (key, value) = entry.map_err(db)?;
                let record: ResponseContextRecord = decode(value.value())?;
                if record.expires_at <= now {
                    expired.push(key.value().to_string());
                }
            }
            expired
        };
        let mut table = write.open_table(RESPONSES).map_err(db)?;
        for key in &expired {
            table.remove(key.as_str()).map_err(db)?;
        }
        drop(table);
        write.commit().map_err(db)?;
        Ok(expired.len() as u64)
    }

    pub fn append_audit(&self, event: AuditEvent) -> Result<AuditEvent, AuthError> {
        let write = self.database.begin_write().map_err(db)?;
        let event = append_audit_in_tx(
            &write,
            event.created_at,
            &event.actor,
            &event.action,
            event.user_id.as_deref(),
            event.token_id.as_deref(),
            event.detail.as_deref(),
        )?;
        write.commit().map_err(db)?;
        Ok(event)
    }

    pub fn list_audit(&self) -> Result<Vec<AuditEvent>, AuthError> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(AUDIT).map_err(db)?;
        let mut records = Vec::new();
        for entry in table.iter().map_err(db)? {
            let (_, value) = entry.map_err(db)?;
            records.push(decode(value.value())?);
        }
        Ok(records)
    }

    fn get_json<T: DeserializeOwned>(
        &self,
        definition: TableDefinition<&str, &[u8]>,
        key: &str,
    ) -> Result<Option<T>, AuthError> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(definition).map_err(db)?;
        let value = table.get(key).map_err(db)?;
        value.map(|value| decode(value.value())).transpose()
    }

    fn list_json<T: DeserializeOwned>(
        &self,
        definition: TableDefinition<&str, &[u8]>,
    ) -> Result<Vec<T>, AuthError> {
        let read = self.database.begin_read().map_err(db)?;
        let table = read.open_table(definition).map_err(db)?;
        let mut records = Vec::new();
        for entry in table.iter().map_err(db)? {
            let (_, value) = entry.map_err(db)?;
            records.push(decode(value.value())?);
        }
        Ok(records)
    }

    fn validate_indexes(&self) -> Result<(), AuthError> {
        let users = self.list_users()?;
        for user in &users {
            let read = self.database.begin_read().map_err(db)?;
            let names = read.open_table(USER_NAMES).map_err(db)?;
            let normalized = normalize_name(&user.name);
            let indexed = names
                .get(normalized.as_str())
                .map_err(db)?
                .ok_or_else(|| AuthError::Corrupt(format!("user {} has no name index", user.id)))?;
            if indexed.value() != user.id.as_bytes() {
                return Err(AuthError::Corrupt(format!(
                    "user {} name index points elsewhere",
                    user.id
                )));
            }
        }
        let user_ids = users
            .into_iter()
            .map(|user| user.id)
            .collect::<std::collections::HashSet<_>>();
        for token in self.list_tokens()? {
            if !user_ids.contains(&token.user_id) {
                return Err(AuthError::Corrupt(format!(
                    "token {} references missing user {}",
                    token.id, token.user_id
                )));
            }
        }
        Ok(())
    }
}

fn get_json_in_tx<T: DeserializeOwned>(
    transaction: &redb::WriteTransaction,
    definition: TableDefinition<&str, &[u8]>,
    key: &str,
) -> Result<Option<T>, AuthError> {
    let table = transaction.open_table(definition).map_err(db)?;
    let value = table
        .get(key)
        .map_err(db)?
        .map(|value| decode(value.value()))
        .transpose()?;
    Ok(value)
}

fn append_audit_in_tx(
    transaction: &redb::WriteTransaction,
    created_at: u64,
    actor: &str,
    action: &str,
    user_id: Option<&str>,
    token_id: Option<&str>,
    detail: Option<&str>,
) -> Result<AuditEvent, AuthError> {
    let sequence = {
        let mut meta = transaction.open_table(META).map_err(db)?;
        let sequence = meta
            .get(NEXT_AUDIT_SEQUENCE_KEY)
            .map_err(db)?
            .map(|value| value.value())
            .unwrap_or(1);
        meta.insert(NEXT_AUDIT_SEQUENCE_KEY, sequence.saturating_add(1))
            .map_err(db)?;
        sequence
    };
    let event = AuditEvent {
        sequence,
        created_at,
        actor: actor.to_string(),
        action: action.to_string(),
        user_id: user_id.map(ToOwned::to_owned),
        token_id: token_id.map(ToOwned::to_owned),
        detail: detail.map(ToOwned::to_owned),
    };
    let encoded = encode(&event)?;
    transaction
        .open_table(AUDIT)
        .map_err(db)?
        .insert(sequence, encoded.as_slice())
        .map_err(db)?;
    Ok(event)
}

fn normalize_name(name: &str) -> String {
    name.trim().to_lowercase()
}

fn user_token_key(user_id: &str, token_id: &str) -> String {
    format!("{user_id}\0{token_id}")
}

fn expiry_key(expires_at: u64, id: &str) -> String {
    format!("{expires_at:020}\0{id}")
}

fn usage_key(record: &HourlyUsageRecord) -> String {
    format!(
        "{:020}\0{}\0{}\0{}",
        record.hour_start, record.user_id, record.token_id, record.workload
    )
}

fn response_key(user_id: &str, response_id: &str) -> String {
    format!("{user_id}\0{response_id}")
}

fn response_expiry_key(expires_at: u64, user_id: &str, response_id: &str) -> String {
    format!("{expires_at:020}\0{user_id}\0{response_id}")
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, AuthError> {
    Ok(serde_json::to_vec(value)?)
}

fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, AuthError> {
    serde_json::from_slice(bytes)
        .map_err(|error| AuthError::Corrupt(format!("invalid persisted JSON: {error}")))
}

fn db(error: impl std::fmt::Display) -> AuthError {
    AuthError::Database(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;

    use super::*;
    use crate::{CredentialError, CredentialSnapshot, NewToken, RatePolicyOverride, Scope};

    fn open() -> (tempfile::TempDir, AccessStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = AccessStore::open_in(dir.path()).unwrap();
        (dir, store)
    }

    fn user(store: &AccessStore, name: &str) -> UserRecord {
        store
            .create_user(
                NewUser {
                    name: name.into(),
                    rate_policy: RatePolicyOverride::default(),
                },
                100,
            )
            .unwrap()
    }

    fn token(store: &AccessStore, user_id: &str) -> CreatedToken {
        store
            .issue_token(
                user_id,
                NewToken {
                    label: "automation".into(),
                    scopes: BTreeSet::from([Scope::Text, Scope::Embeddings]),
                    rate_policy: RatePolicyOverride::default(),
                    expires_at: Some(1_000),
                },
                100,
            )
            .unwrap()
    }

    #[test]
    fn migrates_empty_database_to_current_schema() {
        let (_dir, store) = open();
        assert_eq!(store.schema_version().unwrap(), CURRENT_SCHEMA_VERSION);
        assert_eq!(store.list_users().unwrap(), Vec::new());
    }

    #[test]
    fn creates_private_database_and_pepper_files() {
        let (_dir, store) = open();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&store.paths().database)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(&store.paths().pepper)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn enforces_normalized_unique_user_names() {
        let (_dir, store) = open();
        user(&store, "Research Team");
        let error = store
            .create_user(
                NewUser {
                    name: " research team ".into(),
                    rate_policy: RatePolicyOverride::default(),
                },
                101,
            )
            .unwrap_err();
        assert!(matches!(error, AuthError::DuplicateUserName));
    }

    #[test]
    fn token_secret_is_one_time_and_never_persisted() {
        let (dir, store) = open();
        let user = user(&store, "ops");
        let created = token(&store, &user.id);
        assert!(created.secret.starts_with("hfr_"));
        assert!(!format!("{created:?}").contains(&created.secret));
        drop(store);
        let database = fs::read(dir.path().join("access.redb")).unwrap();
        let pepper = fs::read(dir.path().join("access.pepper")).unwrap();
        assert!(!database
            .windows(created.secret.len())
            .any(|window| window == created.secret.as_bytes()));
        assert!(!pepper
            .windows(created.secret.len())
            .any(|window| window == created.secret.as_bytes()));
    }

    #[test]
    fn snapshot_verifies_scope_expiry_revocation_and_disable() {
        let (_dir, store) = open();
        let user = user(&store, "ops");
        let created = token(&store, &user.id);
        let snapshot = CredentialSnapshot::load(&store).unwrap();
        let principal = snapshot.verify(&created.secret, 999).unwrap();
        assert!(principal.has_scope(Scope::Text));
        assert!(!principal.has_scope(Scope::Images));
        assert_eq!(
            snapshot.verify(&created.secret, 1_000),
            Err(CredentialError::Expired)
        );

        store.revoke_token(&created.token.id, 500).unwrap();
        let snapshot = CredentialSnapshot::load(&store).unwrap();
        assert_eq!(
            snapshot.verify(&created.secret, 600),
            Err(CredentialError::Revoked)
        );

        let second = token(&store, &user.id);
        store
            .set_user_status(&user.id, UserStatus::Disabled, 700)
            .unwrap();
        let snapshot = CredentialSnapshot::load(&store).unwrap();
        assert_eq!(
            snapshot.verify(&second.secret, 800),
            Err(CredentialError::UserDisabled)
        );
    }

    #[test]
    fn revocation_is_idempotent_and_audited_once() {
        let (_dir, store) = open();
        let user = user(&store, "ops");
        let created = token(&store, &user.id);
        store.revoke_token(&created.token.id, 200).unwrap();
        store.revoke_token(&created.token.id, 300).unwrap();
        let events = store.list_audit().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.action == "token.revoked")
                .count(),
            1
        );
    }

    #[test]
    fn usage_rollups_merge_and_prune_without_content_fields() {
        let (_dir, store) = open();
        let mut record = HourlyUsageRecord {
            hour_start: 3_600,
            user_id: "u".into(),
            token_id: "t".into(),
            workload: "text".into(),
            counters: crate::UsageCounters {
                requests: 1,
                input_tokens: 10,
                ..Default::default()
            },
        };
        store.add_usage(&record).unwrap();
        record.counters.requests = 2;
        record.counters.input_tokens = 5;
        store.add_usage(&record).unwrap();
        store.add_usage(&record).unwrap();
        let rows = store.list_usage().unwrap();
        assert_eq!(rows[0].counters.requests, 5);
        assert_eq!(rows[0].counters.input_tokens, 20);
        assert_eq!(store.prune_usage_before(7_200).unwrap(), 1);
    }

    #[test]
    fn response_lookup_is_scoped_by_user() {
        let (_dir, store) = open();
        store
            .put_response(&ResponseContextRecord {
                user_id: "alice".into(),
                response_id: "resp_1".into(),
                parent_response_id: None,
                created_at: 1,
                updated_at: 1,
                expires_at: 2,
                payload: b"context".to_vec(),
            })
            .unwrap();
        assert!(store.get_response("alice", "resp_1").unwrap().is_some());
        assert!(store.get_response("bob", "resp_1").unwrap().is_none());
    }

    #[test]
    fn responses_are_bounded_per_user_and_expiry_is_pruned() {
        let (_dir, store) = open();
        for index in 0..3 {
            store
                .put_response_bounded(
                    &ResponseContextRecord {
                        user_id: "alice".into(),
                        response_id: format!("resp_{index}"),
                        parent_response_id: None,
                        created_at: index,
                        updated_at: index,
                        expires_at: if index == 2 { 10 } else { 100 },
                        payload: vec![index as u8],
                    },
                    2,
                )
                .unwrap();
        }
        let rows = store.list_user_responses("alice").unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.response_id != "resp_0"));
        assert_eq!(store.prune_responses_expired(10).unwrap(), 1);
        assert!(store.get_response("alice", "resp_2").unwrap().is_none());
    }

    #[test]
    fn corrupt_pepper_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("access.pepper"), b"short").unwrap();
        let error = AccessStore::open_in(dir.path()).unwrap_err();
        assert!(matches!(error, AuthError::Corrupt(_)));
    }

    #[test]
    fn newer_schema_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let paths = StorePaths::in_hipfire_dir(dir.path());
        let store = AccessStore::open(paths.clone()).unwrap();
        let write = store.database.begin_write().unwrap();
        write
            .open_table(META)
            .unwrap()
            .insert(SCHEMA_VERSION_KEY, CURRENT_SCHEMA_VERSION + 1)
            .unwrap();
        write.commit().unwrap();
        drop(store);
        let error = AccessStore::open(paths).unwrap_err();
        assert!(matches!(error, AuthError::UnsupportedSchema { .. }));
    }
}
