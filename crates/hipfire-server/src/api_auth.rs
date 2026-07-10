//! API-token middleware for `/v1/*` and `/sdapi/*`.
//!
//! This boundary is intentionally separate from `auth::admin_gate`: API
//! credentials never authorize administrator routes, and admin cookies/local
//! bearer secrets never become inference principals.

use std::net::IpAddr;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    body::Body,
    extract::State,
    http::{header, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use hipfire_auth::{CredentialError, RequestPrincipal, Scope};
use hipfire_config::{ApiAuthMode, HipfireConfig};
use serde_json::json;

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
    mut request: Request<Body>,
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
        request.extensions_mut().insert(principal);
        return next.run(request).await;
    }

    let presented = match request.headers().get(header::AUTHORIZATION) {
        None if policy == ApiAuthPolicy::Optional => {
            let principal = RequestPrincipal::anonymous_local();
            if let Some(response) = enforce_scope(&principal, request.uri().path()) {
                return response;
            }
            request.extensions_mut().insert(principal);
            return next.run(request).await;
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
            request.extensions_mut().insert(principal);
            next.run(request).await
        }
        Err(error) => unauthorized(match error {
            CredentialError::Invalid => "invalid API credential",
            CredentialError::Expired => "API credential expired",
            CredentialError::Revoked => "API credential revoked",
            CredentialError::UserDisabled => "API user disabled",
        }),
    }
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

fn is_loopback_host(host: &str) -> bool {
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

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use axum::{body::Body, http::Request};
    use hipfire_auth::{NewToken, NewUser, RatePolicyOverride};
    use hipfire_config::LoadedConfig;
    use tower::ServiceExt;

    fn state(config: HipfireConfig, directory: &std::path::Path) -> SharedState {
        crate::AppState::new_loaded_with_directories(
            LoadedConfig::from_config(config),
            directory.join("training"),
            directory.join("access"),
        )
    }

    fn bearer_for(state: &SharedState, scopes: BTreeSet<Scope>) -> String {
        let store = state.access.store().unwrap();
        let user = store
            .create_user(
                NewUser {
                    name: "api-user".into(),
                    rate_policy: RatePolicyOverride::default(),
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
}
