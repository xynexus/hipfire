use std::collections::BTreeSet;

use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use hipfire_admin_types::{
    AccessAuditEvent, AccessRateLimitRow, AccessRatePolicy, AccessScope, AccessToken,
    AccessUsageCounters, AccessUsageResponse, AccessUsageRow, AccessUser, AccessUserStatus,
    CreateAccessTokenRequest, CreateAccessUserRequest, CreatedAccessToken, CursorPage,
    EffectiveAccessRatePolicy, PatchAccessUserRequest,
};
use hipfire_auth::{
    AuthKind, NewToken, NewUser, RatePolicy, RatePolicyOverride, RequestPrincipal, Scope,
    UserStatus, WorkloadClass,
};
use serde::Deserialize;
use serde_json::json;

use crate::SharedState;

#[derive(Debug, Default, Deserialize)]
pub struct PageQuery {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
pub struct UserListQuery {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub search: Option<String>,
    pub status: Option<AccessUserStatus>,
}

#[derive(Debug, Default, Deserialize)]
pub struct UsageQuery {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub from: Option<u64>,
    pub to: Option<u64>,
    pub user_id: Option<String>,
    pub token_id: Option<String>,
    pub workload: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct RateLimitQuery {
    pub cursor: Option<String>,
    pub limit: Option<usize>,
    pub user_id: Option<String>,
    pub token_id: Option<String>,
}

pub async fn list_users(
    State(state): State<SharedState>,
    Query(query): Query<UserListQuery>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    match tokio::task::spawn_blocking(move || {
        let tokens = store.list_tokens().map_err(|error| error.to_string())?;
        let mut users = store
            .list_users()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|user| {
                let count = tokens
                    .iter()
                    .filter(|token| token.user_id == user.id)
                    .count();
                user_to_api(user, count)
            })
            .collect::<Vec<_>>();
        if let Some(search) = query.search.as_deref() {
            let search = search.trim().to_lowercase();
            users.retain(|user| user.name.to_lowercase().contains(&search));
        }
        if let Some(status) = query.status {
            users.retain(|user| user.status == status);
        }
        users.sort_by(|left, right| left.id.cmp(&right.id));
        Ok::<_, String>(paginate(
            users,
            query.cursor.as_deref(),
            query.limit,
            |user| user.id.clone(),
        ))
    })
    .await
    {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(error)) => internal(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn create_user(
    State(state): State<SharedState>,
    Json(request): Json<CreateAccessUserRequest>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    let result = tokio::task::spawn_blocking(move || {
        store.create_user(
            NewUser {
                name: request.name,
                rate_policy: policy_from_api(request.rate_policy),
            },
            now_secs(),
        )
    })
    .await;
    match result {
        Ok(Ok(user)) => (StatusCode::CREATED, Json(user_to_api(user, 0))).into_response(),
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn get_user(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    match tokio::task::spawn_blocking(move || {
        let user = store.get_user(&id)?;
        let count = store.list_user_tokens(&id)?.len();
        Ok::<_, hipfire_auth::AuthError>(user.map(|user| user_to_api(user, count)))
    })
    .await
    {
        Ok(Ok(Some(user))) => Json(user).into_response(),
        Ok(Ok(None)) => not_found("user not found"),
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn patch_user(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<PatchAccessUserRequest>,
) -> Response {
    if request.status.is_none() && request.rate_policy.is_none() {
        return bad_request("status or rate_policy is required");
    }
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    let user_id = id.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut user = store
            .get_user(&user_id)?
            .ok_or(hipfire_auth::AuthError::UserNotFound)?;
        let now = now_secs();
        if let Some(status) = request.status {
            user = store.set_user_status(&user_id, status_from_api(status), now)?;
        }
        if let Some(policy) = request.rate_policy {
            user = store.set_user_rate_policy(&user_id, policy_from_api(policy), now)?;
        }
        let count = store.list_user_tokens(&user_id)?.len();
        Ok::<_, hipfire_auth::AuthError>((user, count))
    })
    .await;
    match result {
        Ok(Ok((user, count))) => {
            if user.status == UserStatus::Disabled {
                let cancelled = state.prefill_scheduler.lock().await.cancel_by_user(&id);
                if !cancelled.is_empty() {
                    state.prefill_notify.notify_waiters();
                }
            }
            if let Err(error) = state.access.refresh_credentials() {
                return internal(error);
            }
            Json(user_to_api(user, count)).into_response()
        }
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn list_user_tokens(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Query(query): Query<PageQuery>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    match tokio::task::spawn_blocking(move || {
        if store.get_user(&id)?.is_none() {
            return Ok(None);
        }
        let mut tokens = store
            .list_user_tokens(&id)?
            .into_iter()
            .map(token_to_api)
            .collect::<Vec<_>>();
        tokens.sort_by(|left, right| left.id.cmp(&right.id));
        Ok::<_, hipfire_auth::AuthError>(Some(paginate(
            tokens,
            query.cursor.as_deref(),
            query.limit,
            |token| token.id.clone(),
        )))
    })
    .await
    {
        Ok(Ok(Some(page))) => Json(page).into_response(),
        Ok(Ok(None)) => not_found("user not found"),
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn create_token(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(request): Json<CreateAccessTokenRequest>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    let result = tokio::task::spawn_blocking(move || {
        store.issue_token(
            &id,
            NewToken {
                label: request.label,
                scopes: request.scopes.into_iter().map(scope_from_api).collect(),
                rate_policy: policy_from_api(request.rate_policy),
                expires_at: request.expires_at,
            },
            now_secs(),
        )
    })
    .await;
    match result {
        Ok(Ok(created)) => {
            if let Err(error) = state.access.refresh_credentials() {
                return internal(error);
            }
            let body = CreatedAccessToken {
                token: token_to_api(created.token),
                secret: created.secret,
            };
            (
                StatusCode::CREATED,
                [(header::CACHE_CONTROL, "no-store")],
                Json(body),
            )
                .into_response()
        }
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn revoke_token(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    let token_id = id.clone();
    let result =
        tokio::task::spawn_blocking(move || store.revoke_token(&token_id, now_secs())).await;
    match result {
        Ok(Ok(token)) => {
            let cancelled = state.prefill_scheduler.lock().await.cancel_by_token(&id);
            if !cancelled.is_empty() {
                state.prefill_notify.notify_waiters();
            }
            if let Err(error) = state.access.refresh_credentials() {
                return internal(error);
            }
            Json(json!({"revoked": true, "token": token_to_api(token)})).into_response()
        }
        Ok(Err(hipfire_auth::AuthError::TokenNotFound)) => {
            Json(json!({"revoked": false})).into_response()
        }
        Ok(Err(error)) => auth_error(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn get_usage(
    State(state): State<SharedState>,
    Query(query): Query<UsageQuery>,
) -> Response {
    if let Some(writer) = &state.usage_writer {
        if let Err(error) = writer.flush(now_secs()) {
            return internal(error);
        }
    }
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    match tokio::task::spawn_blocking(move || {
        let mut rows = store.list_usage().map_err(|error| error.to_string())?;
        rows.retain(|row| query.from.is_none_or(|from| row.hour_start >= from));
        rows.retain(|row| query.to.is_none_or(|to| row.hour_start < to));
        rows.retain(|row| query.user_id.as_ref().is_none_or(|id| &row.user_id == id));
        rows.retain(|row| query.token_id.as_ref().is_none_or(|id| &row.token_id == id));
        rows.retain(|row| {
            query
                .workload
                .as_ref()
                .is_none_or(|kind| &row.workload == kind)
        });
        rows.sort_by(|left, right| usage_key(left).cmp(&usage_key(right)));
        let api_rows = rows.into_iter().map(usage_to_api).collect::<Vec<_>>();
        let totals = api_rows
            .iter()
            .fold(AccessUsageCounters::default(), |mut sum, row| {
                add_counters(&mut sum, &row.counters);
                sum
            });
        let page = paginate(
            api_rows,
            query.cursor.as_deref(),
            query.limit,
            usage_api_key,
        );
        Ok::<_, String>(AccessUsageResponse { rows: page, totals })
    })
    .await
    {
        Ok(Ok(response)) => Json(response).into_response(),
        Ok(Err(error)) => internal(error),
        Err(error) => internal(error.to_string()),
    }
}

pub async fn get_rate_limits(
    State(state): State<SharedState>,
    Query(query): Query<RateLimitQuery>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    let (users, tokens) = match tokio::task::spawn_blocking(move || {
        Ok::<_, hipfire_auth::AuthError>((store.list_users()?, store.list_tokens()?))
    })
    .await
    {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => return auth_error(error),
        Err(error) => return internal(error.to_string()),
    };
    let mut rows = Vec::new();
    for user in users {
        if query.user_id.as_ref().is_some_and(|id| id != &user.id) {
            continue;
        }
        let owned = tokens
            .iter()
            .filter(|token| token.user_id == user.id)
            .collect::<Vec<_>>();
        if owned.is_empty() && query.token_id.is_none() {
            rows.push(rate_limit_row(&state, &user, None));
        }
        for token in owned {
            if query.token_id.as_ref().is_some_and(|id| id != &token.id) {
                continue;
            }
            rows.push(rate_limit_row(&state, &user, Some(token)));
        }
    }
    rows.sort_by(|left, right| rate_limit_key(left).cmp(&rate_limit_key(right)));
    Json(paginate(
        rows,
        query.cursor.as_deref(),
        query.limit,
        rate_limit_key,
    ))
    .into_response()
}

pub async fn get_audit(
    State(state): State<SharedState>,
    Query(query): Query<PageQuery>,
) -> Response {
    let store = match state.access.store() {
        Ok(store) => store,
        Err(error) => return internal(error),
    };
    match tokio::task::spawn_blocking(move || {
        let mut events = store
            .list_audit()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|event| AccessAuditEvent {
                sequence: event.sequence,
                created_at: event.created_at,
                actor: event.actor,
                action: event.action,
                user_id: event.user_id,
                token_id: event.token_id,
                detail: event.detail,
            })
            .collect::<Vec<_>>();
        events.sort_by_key(|event| event.sequence);
        Ok::<_, String>(paginate(
            events,
            query.cursor.as_deref(),
            query.limit,
            |event| format!("{:020}", event.sequence),
        ))
    })
    .await
    {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(error)) => internal(error),
        Err(error) => internal(error.to_string()),
    }
}

fn rate_limit_row(
    state: &SharedState,
    user: &hipfire_auth::UserRecord,
    token: Option<&hipfire_auth::TokenRecord>,
) -> AccessRateLimitRow {
    let principal = RequestPrincipal {
        user_id: Some(user.id.clone()),
        token_id: token.map(|token| token.id.clone()),
        scopes: BTreeSet::new(),
        auth_kind: AuthKind::ApiToken,
    };
    let user_policy = RatePolicy::default().with_override(&user.rate_policy);
    let effective = token
        .map(|token| user_policy.stricter_token_policy(&token.rate_policy))
        .unwrap_or(user_policy);
    let token_policy = token
        .map(|token| token.rate_policy.clone())
        .unwrap_or_default();
    let status = state.rate_limiter.status_at(
        now_secs() as f64,
        &principal,
        &user.rate_policy,
        &token_policy,
    );
    AccessRateLimitRow {
        user_id: user.id.clone(),
        token_id: token.map(|token| token.id.clone()),
        effective_policy: effective_policy_to_api(effective),
        request_remaining: status.request_remaining,
        text_token_remaining: status.text_token_remaining,
        active_text: state
            .rate_limiter
            .in_flight(&principal, WorkloadClass::Text),
        active_images: state
            .rate_limiter
            .in_flight(&principal, WorkloadClass::Image),
        active_training: state
            .rate_limiter
            .in_flight(&principal, WorkloadClass::Training),
    }
}

fn paginate<T>(
    items: Vec<T>,
    cursor: Option<&str>,
    limit: Option<usize>,
    key: impl Fn(&T) -> String,
) -> CursorPage<T> {
    let limit = limit.unwrap_or(50).clamp(1, 100);
    let mut filtered = items
        .into_iter()
        .filter(|item| cursor.is_none_or(|cursor| key(item).as_str() > cursor))
        .collect::<Vec<_>>();
    let has_more = filtered.len() > limit;
    filtered.truncate(limit);
    let next_cursor = has_more.then(|| key(filtered.last().unwrap()));
    CursorPage {
        items: filtered,
        next_cursor,
    }
}

fn user_to_api(user: hipfire_auth::UserRecord, token_count: usize) -> AccessUser {
    AccessUser {
        id: user.id,
        name: user.name,
        status: status_to_api(user.status),
        rate_policy: policy_to_api(user.rate_policy),
        token_count,
        created_at: user.created_at,
        updated_at: user.updated_at,
    }
}

fn token_to_api(token: hipfire_auth::TokenRecord) -> AccessToken {
    AccessToken {
        id: token.id,
        user_id: token.user_id,
        label: token.label,
        scopes: token.scopes.into_iter().map(scope_to_api).collect(),
        rate_policy: policy_to_api(token.rate_policy),
        created_at: token.created_at,
        expires_at: token.expires_at,
        revoked_at: token.revoked_at,
    }
}

fn usage_to_api(row: hipfire_auth::HourlyUsageRecord) -> AccessUsageRow {
    AccessUsageRow {
        hour_start: row.hour_start,
        user_id: row.user_id,
        token_id: row.token_id,
        workload: row.workload,
        counters: AccessUsageCounters {
            requests: row.counters.requests,
            errors: row.counters.errors,
            rate_limit_hits: row.counters.rate_limit_hits,
            input_tokens: row.counters.input_tokens,
            output_tokens: row.counters.output_tokens,
            cache_tokens: row.counters.cache_tokens,
            images: row.counters.images,
            megapixel_steps: row.counters.megapixel_steps,
            training_seconds: row.counters.training_seconds,
        },
    }
}

fn add_counters(sum: &mut AccessUsageCounters, row: &AccessUsageCounters) {
    sum.requests = sum.requests.saturating_add(row.requests);
    sum.errors = sum.errors.saturating_add(row.errors);
    sum.rate_limit_hits = sum.rate_limit_hits.saturating_add(row.rate_limit_hits);
    sum.input_tokens = sum.input_tokens.saturating_add(row.input_tokens);
    sum.output_tokens = sum.output_tokens.saturating_add(row.output_tokens);
    sum.cache_tokens = sum.cache_tokens.saturating_add(row.cache_tokens);
    sum.images = sum.images.saturating_add(row.images);
    sum.megapixel_steps = sum.megapixel_steps.saturating_add(row.megapixel_steps);
    sum.training_seconds = sum.training_seconds.saturating_add(row.training_seconds);
}

fn usage_key(row: &hipfire_auth::HourlyUsageRecord) -> String {
    format!(
        "{:020}:{}:{}:{}",
        row.hour_start, row.user_id, row.token_id, row.workload
    )
}

fn usage_api_key(row: &AccessUsageRow) -> String {
    format!(
        "{:020}:{}:{}:{}",
        row.hour_start, row.user_id, row.token_id, row.workload
    )
}

fn rate_limit_key(row: &AccessRateLimitRow) -> String {
    format!(
        "{}:{}",
        row.user_id,
        row.token_id.as_deref().unwrap_or("aggregate")
    )
}

fn policy_from_api(policy: AccessRatePolicy) -> RatePolicyOverride {
    RatePolicyOverride {
        requests_per_minute: policy.requests_per_minute,
        request_burst: policy.request_burst,
        text_tokens_per_minute: policy.text_tokens_per_minute,
        text_token_burst: policy.text_token_burst,
        max_in_flight_text: policy.max_in_flight_text,
        max_in_flight_images: policy.max_in_flight_images,
        megapixel_steps_per_minute: policy.megapixel_steps_per_minute,
        megapixel_step_burst: policy.megapixel_step_burst,
        max_in_flight_training: policy.max_in_flight_training,
    }
}

fn policy_to_api(policy: RatePolicyOverride) -> AccessRatePolicy {
    AccessRatePolicy {
        requests_per_minute: policy.requests_per_minute,
        request_burst: policy.request_burst,
        text_tokens_per_minute: policy.text_tokens_per_minute,
        text_token_burst: policy.text_token_burst,
        max_in_flight_text: policy.max_in_flight_text,
        max_in_flight_images: policy.max_in_flight_images,
        megapixel_steps_per_minute: policy.megapixel_steps_per_minute,
        megapixel_step_burst: policy.megapixel_step_burst,
        max_in_flight_training: policy.max_in_flight_training,
    }
}

fn effective_policy_to_api(policy: RatePolicy) -> EffectiveAccessRatePolicy {
    EffectiveAccessRatePolicy {
        requests_per_minute: policy.requests_per_minute,
        request_burst: policy.request_burst,
        text_tokens_per_minute: policy.text_tokens_per_minute,
        text_token_burst: policy.text_token_burst,
        max_in_flight_text: policy.max_in_flight_text,
        max_in_flight_images: policy.max_in_flight_images,
        megapixel_steps_per_minute: policy.megapixel_steps_per_minute,
        megapixel_step_burst: policy.megapixel_step_burst,
        max_in_flight_training: policy.max_in_flight_training,
    }
}

fn scope_from_api(scope: AccessScope) -> Scope {
    match scope {
        AccessScope::Text => Scope::Text,
        AccessScope::Embeddings => Scope::Embeddings,
        AccessScope::Images => Scope::Images,
        AccessScope::Training => Scope::Training,
    }
}

fn scope_to_api(scope: Scope) -> AccessScope {
    match scope {
        Scope::Text => AccessScope::Text,
        Scope::Embeddings => AccessScope::Embeddings,
        Scope::Images => AccessScope::Images,
        Scope::Training => AccessScope::Training,
    }
}

fn status_from_api(status: AccessUserStatus) -> UserStatus {
    match status {
        AccessUserStatus::Enabled => UserStatus::Enabled,
        AccessUserStatus::Disabled => UserStatus::Disabled,
    }
}

fn status_to_api(status: UserStatus) -> AccessUserStatus {
    match status {
        UserStatus::Enabled => AccessUserStatus::Enabled,
        UserStatus::Disabled => AccessUserStatus::Disabled,
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn auth_error(error: hipfire_auth::AuthError) -> Response {
    match error {
        hipfire_auth::AuthError::UserNotFound | hipfire_auth::AuthError::TokenNotFound => {
            not_found("access record not found")
        }
        hipfire_auth::AuthError::DuplicateUserName | hipfire_auth::AuthError::Invalid(_) => {
            bad_request(error.to_string())
        }
        other => internal(other.to_string()),
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, "invalid_request_error")
}

fn not_found(message: impl Into<String>) -> Response {
    error_response(StatusCode::NOT_FOUND, message, "not_found")
}

fn internal(message: impl Into<String>) -> Response {
    error_response(StatusCode::INTERNAL_SERVER_ERROR, message, "server_error")
}

fn error_response(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.into(), "type": kind}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use hipfire_config::{HipfireConfig, LoadedConfig};
    use tower::ServiceExt;

    fn state(directory: &std::path::Path) -> SharedState {
        crate::AppState::new_loaded_with_directories(
            LoadedConfig::from_config(HipfireConfig::default()),
            directory.join("training"),
            directory.join("access"),
        )
    }

    async fn json_body(response: Response) -> serde_json::Value {
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn admin_crud_discloses_secret_once_and_audits_mutations() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        let created = create_user(
            State(state.clone()),
            Json(CreateAccessUserRequest {
                name: "research".into(),
                rate_policy: AccessRatePolicy::default(),
            }),
        )
        .await;
        assert_eq!(created.status(), StatusCode::CREATED);
        let created = json_body(created).await;
        let user_id = created["id"].as_str().unwrap().to_string();

        let token = create_token(
            State(state.clone()),
            Path(user_id.clone()),
            Json(CreateAccessTokenRequest {
                label: "ci".into(),
                scopes: vec![AccessScope::Text],
                rate_policy: AccessRatePolicy::default(),
                expires_at: None,
            }),
        )
        .await;
        assert_eq!(token.status(), StatusCode::CREATED);
        assert_eq!(token.headers()[header::CACHE_CONTROL], "no-store");
        let token = json_body(token).await;
        let token_id = token["token"]["id"].as_str().unwrap().to_string();
        let secret = token["secret"].as_str().unwrap();
        assert!(secret.starts_with("hfr_"));
        assert!(token["token"].get("digest").is_none());

        let listed = list_user_tokens(
            State(state.clone()),
            Path(user_id.clone()),
            Query(PageQuery::default()),
        )
        .await;
        let listed = json_body(listed).await;
        assert_eq!(listed["items"].as_array().unwrap().len(), 1);
        assert!(listed.to_string().find(secret).is_none());

        let first = revoke_token(State(state.clone()), Path(token_id.clone())).await;
        assert_eq!(json_body(first).await["revoked"], true);
        let second = revoke_token(State(state.clone()), Path(token_id)).await;
        assert_eq!(json_body(second).await["revoked"], true);

        let disabled = patch_user(
            State(state.clone()),
            Path(user_id),
            Json(PatchAccessUserRequest {
                status: Some(AccessUserStatus::Disabled),
                rate_policy: None,
            }),
        )
        .await;
        assert_eq!(json_body(disabled).await["status"], "disabled");

        let audit = get_audit(State(state), Query(PageQuery::default())).await;
        let audit = json_body(audit).await;
        let actions = audit["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|event| event["action"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(actions.contains(&"user.created"));
        assert!(actions.contains(&"token.created"));
        assert_eq!(
            actions
                .iter()
                .filter(|action| **action == "token.revoked")
                .count(),
            1
        );
        assert!(actions.contains(&"user.disabled"));
    }

    #[tokio::test]
    async fn admin_lists_are_cursor_paginated_and_usage_filtered() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        for name in ["alpha", "beta"] {
            let response = create_user(
                State(state.clone()),
                Json(CreateAccessUserRequest {
                    name: name.into(),
                    rate_policy: AccessRatePolicy::default(),
                }),
            )
            .await;
            assert_eq!(response.status(), StatusCode::CREATED);
        }
        let first = list_users(
            State(state.clone()),
            Query(UserListQuery {
                limit: Some(1),
                ..Default::default()
            }),
        )
        .await;
        let first = json_body(first).await;
        let cursor = first["next_cursor"].as_str().unwrap().to_string();
        let second = list_users(
            State(state.clone()),
            Query(UserListQuery {
                cursor: Some(cursor),
                limit: Some(1),
                ..Default::default()
            }),
        )
        .await;
        let second = json_body(second).await;
        assert_eq!(second["items"].as_array().unwrap().len(), 1);

        let hour = now_secs() / 3600 * 3600;
        state
            .access
            .store()
            .unwrap()
            .add_usage(&hipfire_auth::HourlyUsageRecord {
                hour_start: hour,
                user_id: "u".into(),
                token_id: "t".into(),
                workload: "text".into(),
                counters: hipfire_auth::UsageCounters {
                    requests: 2,
                    input_tokens: 9,
                    ..Default::default()
                },
            })
            .unwrap();
        let usage = get_usage(
            State(state),
            Query(UsageQuery {
                from: Some(hour),
                to: Some(hour + 3_600),
                workload: Some("text".into()),
                ..Default::default()
            }),
        )
        .await;
        let usage = json_body(usage).await;
        assert_eq!(usage["totals"]["requests"], 2);
        assert_eq!(usage["totals"]["input_tokens"], 9);
    }

    #[tokio::test]
    async fn access_routes_are_admin_gated_and_mutations_are_same_origin() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(directory.path());
        let secret = state.admin_secret.clone();
        let app = crate::build_router(state, &[]);

        let missing = app
            .clone()
            .oneshot(
                Request::get("/admin/access/users")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

        let authorized = app
            .clone()
            .oneshot(
                Request::get("/admin/access/users")
                    .header(header::AUTHORIZATION, format!("Bearer {secret}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);

        let cross_site = app
            .oneshot(
                Request::post("/admin/access/users")
                    .header(header::AUTHORIZATION, format!("Bearer {secret}"))
                    .header(header::HOST, "localhost:11435")
                    .header(header::ORIGIN, "https://evil.example")
                    .header("sec-fetch-site", "cross-site")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"name":"blocked"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cross_site.status(), StatusCode::FORBIDDEN);
    }
}
