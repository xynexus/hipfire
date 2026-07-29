//! API-token middleware for `/v1/*` and `/sdapi/*`.
//!
//! This boundary is intentionally separate from `auth::admin_gate`: API
//! credentials never authorize administrator routes, and admin cookies/local
//! bearer secrets never become inference principals.

use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{header, HeaderName, HeaderValue, Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use hipfire_auth::{
    CredentialError, RateLimitError, RateLimitStatus, RateReservation, RequestPrincipal,
    ReservationCost, Scope, WorkloadClass,
};
use hipfire_config::{ApiAuthMode, HipfireConfig};
use serde_json::json;

use crate::accounting::{record_rate_limit_hit, RequestAccounting};
use crate::SharedState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiAuthPolicy {
    Off,
    Optional,
    Required,
}

pub fn effective_api_auth_policy(config: &HipfireConfig) -> ApiAuthPolicy {
    match config.api_auth_mode {
        ApiAuthMode::Auto if is_loopback_host(&config.host) => ApiAuthPolicy::Optional,
        ApiAuthMode::Auto => ApiAuthPolicy::Required,
        ApiAuthMode::Off => ApiAuthPolicy::Off,
        ApiAuthMode::Optional => ApiAuthPolicy::Optional,
        ApiAuthMode::Required => ApiAuthPolicy::Required,
    }
}

pub fn validate_api_auth_config(config: &HipfireConfig) -> Result<ApiAuthPolicy, String> {
    let policy = effective_api_auth_policy(config);
    if !is_loopback_host(&config.host)
        && policy != ApiAuthPolicy::Required
        && !config.unsafe_allow_unauthenticated_remote
    {
        return Err(format!(
            "refusing unauthenticated non-loopback bind on {}: set api_auth_mode=required/auto or explicitly set unsafe_allow_unauthenticated_remote=true",
            config.host
        ));
    }
    Ok(policy)
}

pub async fn api_gate(
    State(state): State<SharedState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !is_api_path(request.uri().path()) {
        return next.run(request).await;
    }

    let policy = {
        let config = state.config.lock().await;
        effective_api_auth_policy(&config)
    };
    if policy == ApiAuthPolicy::Off {
        let principal = RequestPrincipal::anonymous_local();
        if let Some(response) = enforce_scope(&principal, request.uri().path()) {
            return response;
        }
        return admit_request(state, request, next, principal).await;
    }

    let presented = match request.headers().get(header::AUTHORIZATION) {
        None if policy == ApiAuthPolicy::Optional => {
            let principal = RequestPrincipal::anonymous_local();
            if let Some(response) = enforce_scope(&principal, request.uri().path()) {
                return response;
            }
            return admit_request(state, request, next, principal).await;
        }
        None => return unauthorized("API credential required"),
        Some(value) => match value
            .to_str()
            .ok()
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            Some(value) => value,
            None => return unauthorized("invalid API credential"),
        },
    };

    let snapshot = match state.access.credentials() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::error!(error = %error, "API credential cache unavailable");
            return service_unavailable();
        }
    };
    match snapshot.verify(presented, now_secs()) {
        Ok(principal) => {
            if let Some(response) = enforce_scope(&principal, request.uri().path()) {
                return response;
            }
            admit_request(state, request, next, principal).await
        }
        Err(error) => unauthorized(match error {
            CredentialError::Invalid => "invalid API credential",
            CredentialError::Expired => "API credential expired",
            CredentialError::Revoked => "API credential revoked",
            CredentialError::UserDisabled => "API user disabled",
        }),
    }
}

async fn admit_request(
    state: SharedState,
    request: Request<Body>,
    next: Next,
    principal: RequestPrincipal,
) -> Response {
    let (mut request, cost, estimated_images) = match estimate_request(request).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let (user_policy, token_policy) = if principal.token_id.is_some() {
        match state
            .access
            .credentials()
            .ok()
            .and_then(|snapshot| snapshot.rate_policies(&principal))
        {
            Some(policies) => policies,
            None => return service_unavailable(),
        }
    } else {
        Default::default()
    };
    let reservation = match state.rate_limiter.reserve_at(
        now_secs_f64(),
        &principal,
        &user_policy,
        &token_policy,
        cost,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            record_rate_limit_hit(state.usage_writer.as_ref(), &principal, cost.workload);
            return rate_limited(error);
        }
    };
    let status = reservation.status();
    let accounting = RequestAccounting::new(
        principal.clone(),
        cost,
        reservation.reporter(),
        state.usage_writer.clone(),
        estimated_images,
    );
    request.extensions_mut().insert(principal);
    request.extensions_mut().insert(accounting.clone());
    let mut response = next.run(request).await;
    add_rate_headers(response.headers_mut(), status);
    if response.status().is_client_error() || response.status().is_server_error() {
        accounting.fail();
        reservation.cancel();
        return response;
    }

    let (parts, body) = response.into_parts();
    let stream = async_stream::stream! {
        let reservation = reservation;
        let mut data = body.into_data_stream();
        while let Some(item) = data.next().await {
            match item {
                Ok(bytes) => yield Ok::<_, axum::Error>(bytes),
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }
        accounting.complete();
        reservation.complete();
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

async fn estimate_request(
    request: Request<Body>,
) -> Result<(Request<Body>, ReservationCost, u64), Response> {
    let path = request.uri().path().to_string();
    let method = request.method().clone();
    let workload = workload_class(&method, &path);
    let cost = ReservationCost::request(workload);
    let estimated_images = 0;
    if method == Method::GET || method == Method::DELETE {
        return Ok((request, cost, estimated_images));
    }
    let is_json = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.starts_with("application/json"))
        .unwrap_or(false);
    if !is_json {
        return Ok((request, cost, estimated_images));
    }
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, 16 * 1024 * 1024).await.map_err(|_| {
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({"error": {
                "message": "request body exceeds the admission limit",
                "type": "invalid_request_error",
                "code": "request_too_large"
            }})),
        )
            .into_response()
    })?;
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).ok();
    let (cost, estimated_images) = estimate_json_cost(workload, bytes.len(), value.as_ref());
    Ok((
        Request::from_parts(parts, Body::from(bytes)),
        cost,
        estimated_images,
    ))
}

fn estimate_json_cost(
    workload: WorkloadClass,
    body_bytes: usize,
    value: Option<&serde_json::Value>,
) -> (ReservationCost, u64) {
    let mut cost = ReservationCost::request(workload);
    let mut estimated_images = 0;
    match workload {
        WorkloadClass::Text => {
            let input_estimate = (body_bytes as f64 / 4.0).ceil();
            let output_estimate = value
                .and_then(|value| {
                    value
                        .get("max_output_tokens")
                        .or_else(|| value.get("max_tokens"))
                        .and_then(|value| value.as_u64())
                })
                .unwrap_or(512) as f64;
            cost.text_tokens = input_estimate + output_estimate;
        }
        WorkloadClass::Image => {
            let width = json_u64(value, "width", 512) as f64;
            let height = json_u64(value, "height", 512) as f64;
            let steps = json_u64(value, "steps", 20) as f64;
            let images =
                json_u64(value, "batch_size", 1).saturating_mul(json_u64(value, "n_iter", 1));
            estimated_images = images;
            let images = images as f64;
            cost.megapixel_steps = (width * height / 1_000_000.0) * steps * images;
            if value
                .and_then(|value| value.get("enable_hr"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
            {
                let scale = value
                    .and_then(|value| value.get("hr_scale"))
                    .and_then(|value| value.as_f64())
                    .unwrap_or(2.0);
                let second_steps = json_u64(value, "hr_second_pass_steps", steps as u64) as f64;
                cost.megapixel_steps +=
                    (width * scale * height * scale / 1_000_000.0) * second_steps * images;
            }
        }
        WorkloadClass::Other | WorkloadClass::Training => {}
    }
    (cost, estimated_images)
}

/// Reserve and account for an API item executed outside the Axum middleware
/// path (currently file batches). Each item remains owned and metered exactly
/// like a direct request instead of inheriting only the outer control call.
pub(crate) fn reserve_internal_json(
    state: &SharedState,
    principal: &RequestPrincipal,
    path: &str,
    body: &serde_json::Value,
) -> Result<(RateReservation, RequestAccounting), RateLimitError> {
    let bytes = serde_json::to_vec(body)
        .map(|bytes| bytes.len())
        .unwrap_or(0);
    let workload = workload_class(&Method::POST, path);
    let (cost, estimated_images) = estimate_json_cost(workload, bytes, Some(body));
    let (user_policy, token_policy) = if principal.token_id.is_some() {
        state
            .access
            .credentials()
            .ok()
            .and_then(|snapshot| snapshot.rate_policies(principal))
            .unwrap_or_default()
    } else {
        Default::default()
    };
    let reservation = state.rate_limiter.reserve_at(
        now_secs_f64(),
        principal,
        &user_policy,
        &token_policy,
        cost,
    )?;
    let accounting = RequestAccounting::new(
        principal.clone(),
        cost,
        reservation.reporter(),
        state.usage_writer.clone(),
        estimated_images,
    );
    Ok((reservation, accounting))
}

fn json_u64(value: Option<&serde_json::Value>, key: &str, default: u64) -> u64 {
    value
        .and_then(|value| value.get(key))
        .and_then(|value| value.as_u64())
        .unwrap_or(default)
}

fn workload_class(method: &Method, path: &str) -> WorkloadClass {
    if method != Method::POST {
        return WorkloadClass::Other;
    }
    if path.starts_with("/sdapi/v1/create/") || path.starts_with("/sdapi/v1/train/") {
        return WorkloadClass::Training;
    }
    if matches!(
        path,
        "/sdapi/v1/txt2img"
            | "/sdapi/v1/img2img"
            | "/sdapi/v1/extra-single-image"
            | "/sdapi/v1/extra-batch-images"
            | "/sdapi/v1/interrogate"
    ) {
        return WorkloadClass::Image;
    }
    if matches!(
        path,
        "/v1/chat/completions" | "/v1/responses" | "/v1/embeddings" | "/v1/rerank"
    ) {
        return WorkloadClass::Text;
    }
    WorkloadClass::Other
}

fn is_api_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/") || path == "/sdapi" || path.starts_with("/sdapi/")
}

fn required_scope(path: &str) -> Option<Scope> {
    match path {
        "/v1/models" => None,
        "/v1/embeddings" | "/v1/rerank" => Some(Scope::Embeddings),
        path if path.starts_with("/sdapi/v1/create/") || path.starts_with("/sdapi/v1/train/") => {
            Some(Scope::Training)
        }
        path if path.starts_with("/sdapi/") => Some(Scope::Images),
        path if path.starts_with("/v1/") => Some(Scope::Text),
        _ => None,
    }
}

fn enforce_scope(principal: &RequestPrincipal, path: &str) -> Option<Response> {
    let scope = required_scope(path)?;
    (!principal.has_scope(scope)).then(|| forbidden(scope))
}

pub(crate) fn principal_has_scope_for_path(principal: &RequestPrincipal, path: &str) -> bool {
    required_scope(path).is_none_or(|scope| principal.has_scope(scope))
}

pub(crate) fn is_loopback_host(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host.ends_with(".localhost")
        || host
            .parse::<IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false)
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(json!({
            "error": {
                "message": message,
                "type": "authentication_error",
                "code": "invalid_api_key"
            }
        })),
    )
        .into_response()
}

fn service_unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": "API credential service unavailable",
                "type": "server_error",
                "code": "auth_unavailable"
            }
        })),
    )
        .into_response()
}

fn forbidden(scope: Scope) -> Response {
    let scope = serde_json::to_value(scope)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "required".to_string());
    (
        StatusCode::FORBIDDEN,
        Json(json!({
            "error": {
                "message": format!("API credential is missing the {scope} scope"),
                "type": "permission_error",
                "code": "missing_scope"
            }
        })),
    )
        .into_response()
}

fn rate_limited(error: RateLimitError) -> Response {
    let mut response = (
        StatusCode::TOO_MANY_REQUESTS,
        Json(json!({
            "error": {
                "message": format!("rate limit exceeded for {}", error.resource),
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        })),
    )
        .into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_str(&error.retry_after_secs.to_string()).unwrap(),
    );
    let suffix = match error.resource {
        "text_tokens" => "tokens",
        "megapixel_steps" => "megapixel-steps",
        other => other,
    };
    let limit_name =
        HeaderName::from_bytes(format!("x-ratelimit-limit-{suffix}").as_bytes()).unwrap();
    let remaining_name =
        HeaderName::from_bytes(format!("x-ratelimit-remaining-{suffix}").as_bytes()).unwrap();
    response
        .headers_mut()
        .insert(limit_name, decimal_header(error.limit));
    response
        .headers_mut()
        .insert(remaining_name, decimal_header(error.remaining));
    response
}

fn add_rate_headers(headers: &mut axum::http::HeaderMap, status: RateLimitStatus) {
    headers.insert(
        "x-ratelimit-limit-requests",
        decimal_header(status.request_limit),
    );
    headers.insert(
        "x-ratelimit-remaining-requests",
        decimal_header(status.request_remaining),
    );
    headers.insert(
        "x-ratelimit-limit-tokens",
        decimal_header(status.text_token_limit),
    );
    headers.insert(
        "x-ratelimit-remaining-tokens",
        decimal_header(status.text_token_remaining),
    );
}

fn decimal_header(value: f64) -> HeaderValue {
    let rendered = if value.fract().abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    };
    HeaderValue::from_str(&rendered).unwrap()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use axum::{body::Body, http::Request};
    use hipfire_auth::{NewToken, NewUser, RatePolicyOverride};
    use hipfire_config::LoadedConfig;
    use tower::ServiceExt;

    async fn accounted_stream(
        axum::extract::Extension(accounting): axum::extract::Extension<RequestAccounting>,
    ) -> Response {
        accounting.report_text(7, 3, 2);
        let chunks = futures::stream::iter([Ok::<_, std::io::Error>(
            axum::body::Bytes::from_static(b"done"),
        )]);
        Body::from_stream(chunks).into_response()
    }

    fn streaming_test_router(state: SharedState) -> axum::Router {
        axum::Router::new()
            .route("/v1/test", axum::routing::post(accounted_stream))
            .route_layer(axum::middleware::from_fn_with_state(
                state.clone(),
                api_gate,
            ))
            .with_state(state)
    }

    fn state(config: HipfireConfig, directory: &std::path::Path) -> SharedState {
        crate::AppState::new_loaded_with_directories(
            LoadedConfig::from_config(config),
            directory.join("training"),
            directory.join("access"),
        )
    }

    fn bearer_for(state: &SharedState, scopes: BTreeSet<Scope>) -> String {
        bearer_for_with_policy(state, scopes, RatePolicyOverride::default())
    }

    fn bearer_for_with_policy(
        state: &SharedState,
        scopes: BTreeSet<Scope>,
        rate_policy: RatePolicyOverride,
    ) -> String {
        let store = state.access.store().unwrap();
        let user = store
            .create_user(
                NewUser {
                    name: "api-user".into(),
                    rate_policy,
                },
                1,
            )
            .unwrap();
        let created = store
            .issue_token(
                &user.id,
                NewToken {
                    label: "test".into(),
                    scopes,
                    rate_policy: RatePolicyOverride::default(),
                    expires_at: Some(u64::MAX),
                },
                1,
            )
            .unwrap();
        state.access.refresh_credentials().unwrap();
        created.secret
    }

    #[test]
    fn auto_is_optional_only_for_loopback() {
        for host in ["127.0.0.1", "::1", "[::1]", "localhost", "api.localhost"] {
            let config = HipfireConfig {
                host: host.into(),
                ..Default::default()
            };
            assert_eq!(effective_api_auth_policy(&config), ApiAuthPolicy::Optional);
        }
        for host in ["0.0.0.0", "::", "192.168.1.5", "api.example.test"] {
            let config = HipfireConfig {
                host: host.into(),
                ..Default::default()
            };
            assert_eq!(effective_api_auth_policy(&config), ApiAuthPolicy::Required);
        }
    }

    #[test]
    fn unsafe_remote_optional_requires_explicit_override() {
        let mut config = HipfireConfig {
            host: "0.0.0.0".into(),
            api_auth_mode: ApiAuthMode::Optional,
            ..Default::default()
        };
        assert!(validate_api_auth_config(&config).is_err());
        config.unsafe_allow_unauthenticated_remote = true;
        assert_eq!(
            validate_api_auth_config(&config),
            Ok(ApiAuthPolicy::Optional)
        );
    }

    #[tokio::test]
    async fn remote_auto_rejects_missing_credentials_before_handler() {
        let directory = tempfile::tempdir().unwrap();
        let config = HipfireConfig {
            host: "0.0.0.0".into(),
            ..Default::default()
        };
        let app = crate::build_router(state(config, directory.path()), &[]);
        let response = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
    }

    #[tokio::test]
    async fn loopback_auto_preserves_anonymous_compatibility() {
        let directory = tempfile::tempdir().unwrap();
        let app = crate::build_router(state(HipfireConfig::default(), directory.path()), &[]);
        let response = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn malformed_optional_credential_is_not_treated_as_anonymous() {
        let directory = tempfile::tempdir().unwrap();
        let app = crate::build_router(state(HipfireConfig::default(), directory.path()), &[]);
        let response = app
            .oneshot(
                Request::get("/v1/models")
                    .header(header::AUTHORIZATION, "Bearer malformed")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn any_valid_token_lists_models_but_route_scopes_are_enforced() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(
            HipfireConfig {
                host: "0.0.0.0".into(),
                ..Default::default()
            },
            directory.path(),
        );
        let bearer = bearer_for(&state, BTreeSet::from([Scope::Images]));
        let app = crate::build_router(state, &[]);
        let models = app
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(models.status(), StatusCode::OK);

        let chat = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(chat.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn embeddings_images_and_training_routes_require_distinct_scopes() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(HipfireConfig::default(), directory.path());
        let bearer = bearer_for(&state, BTreeSet::from([Scope::Text]));
        let app = crate::build_router(state, &[]);
        for (method, path, body) in [
            (Method::POST, "/v1/embeddings", "{}"),
            (Method::GET, "/sdapi/v1/options", ""),
            (Method::POST, "/sdapi/v1/train/embedding", "{}"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{path}");
        }
    }

    #[test]
    fn text_image_and_training_costs_are_workload_aware() {
        let text = json!({"max_tokens": 64});
        let (text_cost, images) = estimate_json_cost(WorkloadClass::Text, 40, Some(&text));
        assert_eq!(text_cost.text_tokens, 74.0);
        assert_eq!(images, 0);

        let image = json!({
            "width": 1024,
            "height": 512,
            "steps": 10,
            "batch_size": 2,
            "n_iter": 1
        });
        let (image_cost, images) = estimate_json_cost(WorkloadClass::Image, 0, Some(&image));
        assert!((image_cost.megapixel_steps - 10.48576).abs() < 1e-6);
        assert_eq!(images, 2);

        assert_eq!(
            workload_class(&Method::POST, "/sdapi/v1/train/embedding"),
            WorkloadClass::Training
        );
        assert_eq!(
            workload_class(&Method::POST, "/v1/embeddings"),
            WorkloadClass::Text
        );
    }

    #[tokio::test]
    async fn api_token_never_authorizes_admin_routes() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(HipfireConfig::default(), directory.path());
        let bearer = bearer_for(&state, BTreeSet::from([Scope::Text]));
        let app = crate::build_router(state, &[]);
        let response = app
            .oneshot(
                Request::get("/admin/stats")
                    .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn completed_request_exhausts_bucket_with_openai_429_headers() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(
            HipfireConfig {
                host: "0.0.0.0".into(),
                ..Default::default()
            },
            directory.path(),
        );
        let bearer = bearer_for_with_policy(
            &state,
            BTreeSet::from([Scope::Images]),
            RatePolicyOverride {
                requests_per_minute: Some(0),
                request_burst: Some(1),
                ..Default::default()
            },
        );
        let store = state.access.store().unwrap();
        let usage_writer = state.usage_writer.clone().unwrap();
        let app = crate::build_router(state, &[]);
        let first = app
            .clone()
            .oneshot(
                Request::get("/v1/models")
                    .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(first.headers()["x-ratelimit-limit-requests"], "1");
        axum::body::to_bytes(first.into_body(), usize::MAX)
            .await
            .unwrap();

        let second = app
            .oneshot(
                Request::get("/v1/models")
                    .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(second.headers()[header::RETRY_AFTER], "60");
        assert_eq!(second.headers()["x-ratelimit-limit-requests"], "1");
        let body = axum::body::to_bytes(second.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "rate_limit_error");
        usage_writer.flush(now_secs()).unwrap();
        let usage = store.list_usage().unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].counters.requests, 2);
        assert_eq!(usage[0].counters.errors, 1);
        assert_eq!(usage[0].counters.rate_limit_hits, 1);
    }

    #[test]
    fn batch_items_are_checked_against_their_own_route_scope() {
        let principal = RequestPrincipal {
            user_id: Some("user-a".into()),
            token_id: Some("token-a".into()),
            scopes: BTreeSet::from([Scope::Images]),
            auth_kind: hipfire_auth::AuthKind::ApiToken,
        };
        assert!(!principal_has_scope_for_path(
            &principal,
            "/v1/chat/completions"
        ));
        assert!(principal_has_scope_for_path(
            &principal,
            "/sdapi/v1/txt2img"
        ));
    }

    #[test]
    fn internal_batch_items_reserve_and_settle_independently() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(HipfireConfig::default(), directory.path());
        let bearer = bearer_for_with_policy(
            &state,
            BTreeSet::from([Scope::Text]),
            RatePolicyOverride {
                requests_per_minute: Some(0),
                request_burst: Some(1),
                ..Default::default()
            },
        );
        let principal = state
            .access
            .credentials()
            .unwrap()
            .verify(&bearer, 2)
            .unwrap();
        let body = json!({"model": "test", "messages": [], "max_tokens": 4});
        let (reservation, accounting) =
            reserve_internal_json(&state, &principal, "/v1/chat/completions", &body).unwrap();
        accounting.report_text(2, 1, 0);
        accounting.complete();
        reservation.complete();

        let second = reserve_internal_json(&state, &principal, "/v1/chat/completions", &body);
        assert_eq!(second.unwrap_err().resource, "requests");

        state
            .usage_writer
            .as_ref()
            .unwrap()
            .flush(now_secs())
            .unwrap();
        let usage = state.access.store().unwrap().list_usage().unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].counters.requests, 1);
        assert_eq!(usage[0].counters.input_tokens, 2);
        assert_eq!(usage[0].counters.output_tokens, 1);
    }

    #[tokio::test]
    async fn response_stream_eof_settles_usage_and_disconnect_refunds() {
        let directory = tempfile::tempdir().unwrap();
        let state = state(HipfireConfig::default(), directory.path());
        let bearer = bearer_for_with_policy(
            &state,
            BTreeSet::from([Scope::Text]),
            RatePolicyOverride {
                requests_per_minute: Some(0),
                request_burst: Some(1),
                ..Default::default()
            },
        );
        let app = streaming_test_router(state.clone());
        let request = || {
            Request::post("/v1/test")
                .header(header::AUTHORIZATION, format!("Bearer {bearer}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap()
        };

        let disconnected = app.clone().oneshot(request()).await.unwrap();
        assert_eq!(disconnected.status(), StatusCode::OK);
        drop(disconnected);

        let completed = app.oneshot(request()).await.unwrap();
        assert_eq!(completed.status(), StatusCode::OK);
        assert_eq!(
            axum::body::to_bytes(completed.into_body(), usize::MAX)
                .await
                .unwrap(),
            "done"
        );

        state
            .usage_writer
            .as_ref()
            .unwrap()
            .flush(now_secs())
            .unwrap();
        let usage = state.access.store().unwrap().list_usage().unwrap();
        assert_eq!(usage.len(), 1);
        assert_eq!(usage[0].counters.requests, 2);
        assert_eq!(usage[0].counters.errors, 1);
        assert_eq!(usage[0].counters.input_tokens, 14);
        assert_eq!(usage[0].counters.output_tokens, 6);
        assert_eq!(usage[0].counters.cache_tokens, 4);
    }
}
