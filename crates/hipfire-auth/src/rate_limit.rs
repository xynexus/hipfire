use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{RatePolicyOverride, RequestPrincipal};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatePolicy {
    pub requests_per_minute: f64,
    pub request_burst: f64,
    pub text_tokens_per_minute: f64,
    pub text_token_burst: f64,
    pub max_in_flight_text: u32,
    pub max_in_flight_images: u32,
    pub megapixel_steps_per_minute: f64,
    pub megapixel_step_burst: f64,
    pub max_in_flight_training: u32,
}

impl Default for RatePolicy {
    /// Base policy for a server reachable from the network. Every effective
    /// policy starts here and is narrowed by user/token overrides.
    fn default() -> Self {
        Self {
            requests_per_minute: 60.0,
            request_burst: 15.0,
            text_tokens_per_minute: 120_000.0,
            text_token_burst: 30_000.0,
            max_in_flight_text: 4,
            max_in_flight_images: 1,
            megapixel_steps_per_minute: 80.0,
            megapixel_step_burst: 40.0,
            max_in_flight_training: 1,
        }
    }
}

impl RatePolicy {
    /// Base policy for a server bound to a loopback address, where the only
    /// possible clients are processes on this machine and throttling them
    /// mostly gets in the way. Selected by the caller from the BIND address
    /// (see `hipfire-server`'s `is_loopback_host`), never from the request
    /// principal: `AuthKind::AnonymousLocal` means "no credential presented",
    /// which a REMOTE client also gets under `api_auth_mode = off/optional`
    /// with `unsafe_allow_unauthenticated_remote`. Keying this off the
    /// principal would hand these limits to anonymous internet traffic.
    ///
    /// Concurrency 0 means unlimited (the `limit > 0` guard in `reserve_at`).
    /// Buckets have no such escape, so they get an unreachable rate rather
    /// than 0 — and a finite one, because `f64::INFINITY` would surface as a
    /// non-finite `limit` in the rate-limit response headers.
    ///
    /// Narrow any of it with `local_rate_policy` in the daemon config.
    pub fn loopback_default() -> Self {
        Self {
            requests_per_minute: 1e9,
            request_burst: 1e9,
            text_tokens_per_minute: 1e9,
            text_token_burst: 1e9,
            max_in_flight_text: 0,
            max_in_flight_images: 1,
            megapixel_steps_per_minute: 80.0,
            megapixel_step_burst: 40.0,
            max_in_flight_training: 1,
        }
    }

    pub fn with_override(self, value: &RatePolicyOverride) -> Self {
        Self {
            requests_per_minute: value
                .requests_per_minute
                .map_or(self.requests_per_minute, |v| v as f64),
            request_burst: value.request_burst.map_or(self.request_burst, |v| v as f64),
            text_tokens_per_minute: value
                .text_tokens_per_minute
                .map_or(self.text_tokens_per_minute, |v| v as f64),
            text_token_burst: value
                .text_token_burst
                .map_or(self.text_token_burst, |v| v as f64),
            max_in_flight_text: value.max_in_flight_text.unwrap_or(self.max_in_flight_text),
            max_in_flight_images: value
                .max_in_flight_images
                .unwrap_or(self.max_in_flight_images),
            megapixel_steps_per_minute: value
                .megapixel_steps_per_minute
                .map_or(self.megapixel_steps_per_minute, |v| v as f64),
            megapixel_step_burst: value
                .megapixel_step_burst
                .map_or(self.megapixel_step_burst, |v| v as f64),
            max_in_flight_training: value
                .max_in_flight_training
                .unwrap_or(self.max_in_flight_training),
        }
    }

    /// Token overrides can only narrow their owning user's effective policy.
    pub fn stricter_token_policy(self, value: &RatePolicyOverride) -> Self {
        let requested = self.with_override(value);
        Self {
            requests_per_minute: self.requests_per_minute.min(requested.requests_per_minute),
            request_burst: self.request_burst.min(requested.request_burst),
            text_tokens_per_minute: self
                .text_tokens_per_minute
                .min(requested.text_tokens_per_minute),
            text_token_burst: self.text_token_burst.min(requested.text_token_burst),
            max_in_flight_text: self.max_in_flight_text.min(requested.max_in_flight_text),
            max_in_flight_images: self
                .max_in_flight_images
                .min(requested.max_in_flight_images),
            megapixel_steps_per_minute: self
                .megapixel_steps_per_minute
                .min(requested.megapixel_steps_per_minute),
            megapixel_step_burst: self
                .megapixel_step_burst
                .min(requested.megapixel_step_burst),
            max_in_flight_training: self
                .max_in_flight_training
                .min(requested.max_in_flight_training),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    Other,
    Text,
    Image,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReservationCost {
    pub requests: f64,
    pub text_tokens: f64,
    pub megapixel_steps: f64,
    pub workload: WorkloadClass,
}

impl ReservationCost {
    pub fn request(workload: WorkloadClass) -> Self {
        Self {
            requests: 1.0,
            text_tokens: 0.0,
            megapixel_steps: 0.0,
            workload,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RateLimitError {
    pub resource: &'static str,
    pub retry_after_secs: u64,
    pub limit: f64,
    pub remaining: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimitStatus {
    pub request_limit: f64,
    pub request_remaining: f64,
    pub text_token_limit: f64,
    pub text_token_remaining: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OwnerKey(String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Resource {
    Requests,
    TextTokens,
    MegapixelSteps,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BucketKey {
    owner: OwnerKey,
    resource: Resource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConcurrencyKey {
    owner: OwnerKey,
    workload: WorkloadClass,
}

#[derive(Debug, Clone, Copy)]
struct BucketState {
    tokens: f64,
    updated_at: f64,
}

#[derive(Debug, Default)]
struct LimiterState {
    buckets: HashMap<BucketKey, BucketState>,
    in_flight: HashMap<ConcurrencyKey, u32>,
}

#[derive(Debug, Default)]
struct LimiterInner {
    state: Mutex<LimiterState>,
}

#[derive(Debug, Clone)]
pub struct RateLimiter {
    inner: Arc<LimiterInner>,
    /// Policy every effective policy is derived from, before user/token
    /// overrides. Chosen once at construction — loopback binds pass
    /// `RatePolicy::loopback_default()` narrowed by config.
    base: RatePolicy,
}

impl Default for RateLimiter {
    /// Network-safe base. Callers on a loopback bind want
    /// [`RateLimiter::with_base`] instead.
    fn default() -> Self {
        Self::with_base(RatePolicy::default())
    }
}

impl RateLimiter {
    pub fn with_base(base: RatePolicy) -> Self {
        Self {
            inner: Arc::new(LimiterInner::default()),
            base,
        }
    }

    /// The un-overridden base policy, for callers that report effective
    /// limits (admin console) and must not re-derive it from `default()`.
    pub fn base(&self) -> RatePolicy {
        self.base
    }

    pub fn reserve_at(
        &self,
        now_secs: f64,
        principal: &RequestPrincipal,
        user_override: &RatePolicyOverride,
        token_override: &RatePolicyOverride,
        cost: ReservationCost,
    ) -> Result<RateReservation, RateLimitError> {
        let user_policy = self.base.with_override(user_override);
        let token_policy = user_policy.stricter_token_policy(token_override);
        let owners = owner_policies(principal, user_policy, token_policy);
        let mut state = self.inner.state.lock().unwrap();
        let mut checks = Vec::new();

        for (owner, policy) in &owners {
            check_bucket(
                &state,
                now_secs,
                owner,
                Resource::Requests,
                cost.requests,
                policy.requests_per_minute,
                policy.request_burst,
                &mut checks,
            )?;
            if cost.text_tokens > 0.0 {
                check_bucket(
                    &state,
                    now_secs,
                    owner,
                    Resource::TextTokens,
                    cost.text_tokens,
                    policy.text_tokens_per_minute,
                    policy.text_token_burst,
                    &mut checks,
                )?;
            }
            if cost.megapixel_steps > 0.0 {
                check_bucket(
                    &state,
                    now_secs,
                    owner,
                    Resource::MegapixelSteps,
                    cost.megapixel_steps,
                    policy.megapixel_steps_per_minute,
                    policy.megapixel_step_burst,
                    &mut checks,
                )?;
            }
            let limit = concurrency_limit(*policy, cost.workload);
            let key = ConcurrencyKey {
                owner: owner.clone(),
                workload: cost.workload,
            };
            if limit > 0 && state.in_flight.get(&key).copied().unwrap_or(0) >= limit {
                return Err(RateLimitError {
                    resource: concurrency_resource(cost.workload),
                    retry_after_secs: 1,
                    limit: limit as f64,
                    remaining: 0.0,
                });
            }
        }

        let mut bucket_keys = Vec::new();
        for check in checks {
            state.buckets.insert(
                check.key.clone(),
                BucketState {
                    tokens: check.available - check.cost,
                    updated_at: now_secs,
                },
            );
            bucket_keys.push((check.key, check.cost, check.capacity));
        }
        let mut concurrency_keys = Vec::new();
        for (owner, policy) in &owners {
            if concurrency_limit(*policy, cost.workload) > 0 {
                let key = ConcurrencyKey {
                    owner: owner.clone(),
                    workload: cost.workload,
                };
                *state.in_flight.entry(key.clone()).or_default() += 1;
                concurrency_keys.push(key);
            }
        }
        let status = status_for(&state, &owners[0], now_secs);
        drop(state);
        Ok(RateReservation {
            inner: self.inner.clone(),
            bucket_keys,
            concurrency_keys,
            estimated: cost,
            reporter: RateLimitReporter::default(),
            status,
            finished: false,
        })
    }

    pub fn in_flight(&self, principal: &RequestPrincipal, workload: WorkloadClass) -> u32 {
        let key = ConcurrencyKey {
            owner: user_owner(principal),
            workload,
        };
        self.inner
            .state
            .lock()
            .unwrap()
            .in_flight
            .get(&key)
            .copied()
            .unwrap_or(0)
    }

    pub fn status_at(
        &self,
        now_secs: f64,
        principal: &RequestPrincipal,
        user_override: &RatePolicyOverride,
        token_override: &RatePolicyOverride,
    ) -> RateLimitStatus {
        let user_policy = self.base.with_override(user_override);
        let token_policy = user_policy.stricter_token_policy(token_override);
        let owners = owner_policies(principal, user_policy, token_policy);
        let state = self.inner.state.lock().unwrap();
        let user = status_for(&state, &owners[0], now_secs);
        let Some(token) = owners.get(1) else {
            return user;
        };
        let token = status_for(&state, token, now_secs);
        RateLimitStatus {
            request_limit: token.request_limit,
            request_remaining: user.request_remaining.min(token.request_remaining),
            text_token_limit: token.text_token_limit,
            text_token_remaining: user.text_token_remaining.min(token.text_token_remaining),
        }
    }
}

#[derive(Debug)]
pub struct RateReservation {
    inner: Arc<LimiterInner>,
    bucket_keys: Vec<(BucketKey, f64, f64)>,
    concurrency_keys: Vec<ConcurrencyKey>,
    estimated: ReservationCost,
    reporter: RateLimitReporter,
    status: RateLimitStatus,
    finished: bool,
}

impl RateReservation {
    pub fn status(&self) -> RateLimitStatus {
        self.status
    }
    pub fn reporter(&self) -> RateLimitReporter {
        self.reporter.clone()
    }
    pub fn report_actual(&mut self, actual: ReservationCost) {
        self.reporter.report(actual);
    }
    pub fn complete(mut self) {
        self.finish(false);
    }
    pub fn cancel(mut self) {
        self.finish(true);
    }

    fn finish(&mut self, cancelled: bool) {
        if self.finished {
            return;
        }
        let mut state = self.inner.state.lock().unwrap();
        for key in &self.concurrency_keys {
            if let Some(current) = state.in_flight.get_mut(key) {
                *current = current.saturating_sub(1);
                if *current == 0 {
                    state.in_flight.remove(key);
                }
            }
        }
        for (key, estimated, capacity) in &self.bucket_keys {
            let actual = if cancelled {
                0.0
            } else {
                actual_cost_for(
                    self.reporter.actual().unwrap_or(self.estimated),
                    key.resource,
                )
            };
            if let Some(bucket) = state.buckets.get_mut(key) {
                bucket.tokens = (bucket.tokens + (*estimated - actual)).min(*capacity);
            }
        }
        self.finished = true;
    }
}

#[derive(Debug, Clone, Default)]
pub struct RateLimitReporter(Arc<Mutex<Option<ReservationCost>>>);

impl RateLimitReporter {
    pub fn report(&self, actual: ReservationCost) {
        *self.0.lock().unwrap() = Some(actual);
    }

    pub fn actual(&self) -> Option<ReservationCost> {
        *self.0.lock().unwrap()
    }
}

impl Drop for RateReservation {
    fn drop(&mut self) {
        // Pending means the handler errored, was cancelled, or its response
        // stream disconnected before EOF. Refund fully and always release
        // concurrency; normal EOF calls `complete` first.
        self.finish(true);
    }
}

struct BucketCheck {
    key: BucketKey,
    cost: f64,
    capacity: f64,
    available: f64,
}

#[allow(clippy::too_many_arguments)]
fn check_bucket(
    state: &LimiterState,
    now: f64,
    owner: &OwnerKey,
    resource: Resource,
    cost: f64,
    per_minute: f64,
    capacity: f64,
    checks: &mut Vec<BucketCheck>,
) -> Result<(), RateLimitError> {
    let key = BucketKey {
        owner: owner.clone(),
        resource,
    };
    let old = state.buckets.get(&key).copied().unwrap_or(BucketState {
        tokens: capacity,
        updated_at: now,
    });
    let refill = per_minute / 60.0;
    let available = (old.tokens + (now - old.updated_at).max(0.0) * refill).min(capacity);
    if available + f64::EPSILON < cost {
        let retry = if refill > 0.0 {
            ((cost - available) / refill).ceil().max(1.0) as u64
        } else {
            60
        };
        return Err(RateLimitError {
            resource: resource_name(resource),
            retry_after_secs: retry,
            limit: capacity,
            remaining: available.max(0.0),
        });
    }
    checks.push(BucketCheck {
        key,
        cost,
        capacity,
        available,
    });
    Ok(())
}

fn owner_policies(
    principal: &RequestPrincipal,
    user: RatePolicy,
    token: RatePolicy,
) -> Vec<(OwnerKey, RatePolicy)> {
    let mut owners = vec![(user_owner(principal), user)];
    if let Some(id) = &principal.token_id {
        owners.push((OwnerKey(format!("token:{id}")), token));
    }
    owners
}

fn user_owner(principal: &RequestPrincipal) -> OwnerKey {
    OwnerKey(format!(
        "user:{}",
        principal.user_id.as_deref().unwrap_or("anonymous-local")
    ))
}

fn concurrency_limit(policy: RatePolicy, workload: WorkloadClass) -> u32 {
    match workload {
        WorkloadClass::Other => 0,
        WorkloadClass::Text => policy.max_in_flight_text,
        WorkloadClass::Image => policy.max_in_flight_images,
        WorkloadClass::Training => policy.max_in_flight_training,
    }
}

fn concurrency_resource(workload: WorkloadClass) -> &'static str {
    match workload {
        WorkloadClass::Text => "text_concurrency",
        WorkloadClass::Image => "image_concurrency",
        WorkloadClass::Training => "training_concurrency",
        WorkloadClass::Other => "concurrency",
    }
}

fn resource_name(resource: Resource) -> &'static str {
    match resource {
        Resource::Requests => "requests",
        Resource::TextTokens => "text_tokens",
        Resource::MegapixelSteps => "megapixel_steps",
    }
}

fn actual_cost_for(cost: ReservationCost, resource: Resource) -> f64 {
    match resource {
        Resource::Requests => cost.requests,
        Resource::TextTokens => cost.text_tokens,
        Resource::MegapixelSteps => cost.megapixel_steps,
    }
}

fn status_for(state: &LimiterState, owner: &(OwnerKey, RatePolicy), now: f64) -> RateLimitStatus {
    let (owner, policy) = owner;
    RateLimitStatus {
        request_limit: policy.request_burst,
        request_remaining: current_tokens(
            state,
            owner,
            Resource::Requests,
            now,
            policy.requests_per_minute,
            policy.request_burst,
        ),
        text_token_limit: policy.text_token_burst,
        text_token_remaining: current_tokens(
            state,
            owner,
            Resource::TextTokens,
            now,
            policy.text_tokens_per_minute,
            policy.text_token_burst,
        ),
    }
}

fn current_tokens(
    state: &LimiterState,
    owner: &OwnerKey,
    resource: Resource,
    now: f64,
    per_minute: f64,
    capacity: f64,
) -> f64 {
    state
        .buckets
        .get(&BucketKey {
            owner: owner.clone(),
            resource,
        })
        .map(|bucket| {
            (bucket.tokens + (now - bucket.updated_at).max(0.0) * per_minute / 60.0).min(capacity)
        })
        .unwrap_or(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AuthKind, Scope};
    use std::collections::BTreeSet;

    fn principal(user: &str, token: &str) -> RequestPrincipal {
        RequestPrincipal {
            user_id: Some(user.into()),
            token_id: Some(token.into()),
            scopes: BTreeSet::from([Scope::Text]),
            auth_kind: AuthKind::ApiToken,
        }
    }

    fn reserve(
        limiter: &RateLimiter,
        now: f64,
        principal: &RequestPrincipal,
        policy: &RatePolicyOverride,
        cost: ReservationCost,
    ) -> Result<RateReservation, RateLimitError> {
        limiter.reserve_at(now, principal, policy, &RatePolicyOverride::default(), cost)
    }

    /// `AuthKind::AnonymousLocal` means "no credential presented", NOT "client
    /// is on this machine" — a REMOTE client gets it too under
    /// `api_auth_mode = off/optional` with `unsafe_allow_unauthenticated_remote`.
    /// So a limiter built with the network base must still throttle it. If
    /// anyone ever re-keys the loopback policy off the principal instead of the
    /// bind address, this fails.
    #[test]
    fn network_base_still_throttles_anonymous_local_principals() {
        let limiter = RateLimiter::default();
        let p = RequestPrincipal::anonymous_local();
        let none = RatePolicyOverride::default();
        let burst = RatePolicy::default().request_burst as usize;
        for _ in 0..burst {
            reserve(
                &limiter,
                0.0,
                &p,
                &none,
                ReservationCost::request(WorkloadClass::Other),
            )
            .expect("requests within the burst are admitted")
            .complete();
        }
        assert!(
            reserve(
                &limiter,
                0.0,
                &p,
                &none,
                ReservationCost::request(WorkloadClass::Other)
            )
            .is_err(),
            "anonymous-local principal must not bypass the network base policy"
        );
    }

    /// Regression guard: the permissive policy lives in `loopback_default()`,
    /// never in `default()` — `default()` is what a public bind uses.
    #[test]
    fn network_default_stays_bounded() {
        let policy = RatePolicy::default();
        assert!(policy.requests_per_minute <= 1_000.0);
        assert!(policy.text_tokens_per_minute <= 1_000_000.0);
        assert!(
            policy.max_in_flight_text > 0,
            "0 means unlimited concurrency"
        );
    }

    /// `local_rate_policy` narrows the loopback base field-wise; anything unset
    /// keeps the loopback value rather than falling back to the network one.
    #[test]
    fn loopback_base_is_narrowed_field_wise_by_config() {
        let base = RatePolicy::loopback_default();
        assert_eq!(base.max_in_flight_text, 0, "unlimited unless configured");
        let narrowed = base.with_override(&RatePolicyOverride {
            max_in_flight_text: Some(2),
            ..Default::default()
        });
        assert_eq!(narrowed.max_in_flight_text, 2);
        assert_eq!(
            narrowed.requests_per_minute, base.requests_per_minute,
            "unset fields keep the loopback base, not the network default"
        );
    }

    #[test]
    fn deterministic_request_bucket_refills() {
        let limiter = RateLimiter::default();
        let p = principal("u", "t");
        let policy = RatePolicyOverride {
            requests_per_minute: Some(60),
            request_burst: Some(2),
            ..Default::default()
        };
        reserve(
            &limiter,
            0.0,
            &p,
            &policy,
            ReservationCost::request(WorkloadClass::Other),
        )
        .unwrap()
        .complete();
        reserve(
            &limiter,
            0.0,
            &p,
            &policy,
            ReservationCost::request(WorkloadClass::Other),
        )
        .unwrap()
        .complete();
        assert_eq!(
            reserve(
                &limiter,
                0.0,
                &p,
                &policy,
                ReservationCost::request(WorkloadClass::Other)
            )
            .unwrap_err()
            .retry_after_secs,
            1
        );
        assert!(reserve(
            &limiter,
            1.0,
            &p,
            &policy,
            ReservationCost::request(WorkloadClass::Other)
        )
        .is_ok());
    }

    #[test]
    fn aggregate_user_limit_applies_across_tokens() {
        let limiter = RateLimiter::default();
        let policy = RatePolicyOverride {
            request_burst: Some(1),
            ..Default::default()
        };
        reserve(
            &limiter,
            0.0,
            &principal("u", "a"),
            &policy,
            ReservationCost::request(WorkloadClass::Other),
        )
        .unwrap()
        .complete();
        assert!(reserve(
            &limiter,
            0.0,
            &principal("u", "b"),
            &policy,
            ReservationCost::request(WorkloadClass::Other)
        )
        .is_err());
    }

    #[test]
    fn token_override_can_only_be_stricter() {
        let limiter = RateLimiter::default();
        let p = principal("u", "a");
        let user = RatePolicyOverride {
            request_burst: Some(2),
            ..Default::default()
        };
        let token = RatePolicyOverride {
            request_burst: Some(1),
            ..Default::default()
        };
        limiter
            .reserve_at(
                0.0,
                &p,
                &user,
                &token,
                ReservationCost::request(WorkloadClass::Other),
            )
            .unwrap()
            .complete();
        assert!(limiter
            .reserve_at(
                0.0,
                &p,
                &user,
                &token,
                ReservationCost::request(WorkloadClass::Other)
            )
            .is_err());
    }

    #[test]
    fn cancellation_refunds_and_releases_concurrency() {
        let limiter = RateLimiter::default();
        let p = principal("u", "t");
        let policy = RatePolicyOverride {
            request_burst: Some(1),
            max_in_flight_text: Some(1),
            ..Default::default()
        };
        let cost = ReservationCost::request(WorkloadClass::Text);
        let reservation = reserve(&limiter, 0.0, &p, &policy, cost).unwrap();
        assert_eq!(limiter.in_flight(&p, WorkloadClass::Text), 1);
        drop(reservation);
        assert_eq!(limiter.in_flight(&p, WorkloadClass::Text), 0);
        assert!(reserve(&limiter, 0.0, &p, &policy, cost).is_ok());
    }

    #[test]
    fn completion_releases_concurrency() {
        let limiter = RateLimiter::default();
        let p = principal("u", "t");
        let policy = RatePolicyOverride {
            max_in_flight_text: Some(1),
            ..Default::default()
        };
        let cost = ReservationCost {
            requests: 0.0,
            text_tokens: 0.0,
            megapixel_steps: 0.0,
            workload: WorkloadClass::Text,
        };
        let reservation = reserve(&limiter, 0.0, &p, &policy, cost).unwrap();
        assert!(reserve(&limiter, 0.0, &p, &policy, cost).is_err());
        reservation.complete();
        assert!(reserve(&limiter, 0.0, &p, &policy, cost).is_ok());
    }

    #[test]
    fn image_cost_bucket_and_training_exclusivity_are_independent() {
        let limiter = RateLimiter::default();
        let p = principal("u", "t");
        let policy = RatePolicyOverride {
            megapixel_step_burst: Some(1),
            megapixel_steps_per_minute: Some(0),
            ..Default::default()
        };
        reserve(
            &limiter,
            0.0,
            &p,
            &policy,
            ReservationCost {
                requests: 0.0,
                text_tokens: 0.0,
                megapixel_steps: 0.75,
                workload: WorkloadClass::Image,
            },
        )
        .unwrap()
        .complete();
        assert!(reserve(
            &limiter,
            0.0,
            &p,
            &policy,
            ReservationCost {
                requests: 0.0,
                text_tokens: 0.0,
                megapixel_steps: 0.3,
                workload: WorkloadClass::Image,
            },
        )
        .is_err());

        let training = ReservationCost {
            requests: 0.0,
            text_tokens: 0.0,
            megapixel_steps: 0.0,
            workload: WorkloadClass::Training,
        };
        let held = reserve(&limiter, 0.0, &p, &policy, training).unwrap();
        assert!(reserve(&limiter, 0.0, &p, &policy, training).is_err());
        held.cancel();
        assert!(reserve(&limiter, 0.0, &p, &policy, training).is_ok());
    }

    #[test]
    fn completion_refunds_overestimate() {
        let limiter = RateLimiter::default();
        let p = principal("u", "t");
        let policy = RatePolicyOverride {
            text_token_burst: Some(100),
            text_tokens_per_minute: Some(0),
            ..Default::default()
        };
        let estimated = ReservationCost {
            requests: 0.0,
            text_tokens: 100.0,
            megapixel_steps: 0.0,
            workload: WorkloadClass::Text,
        };
        let mut reservation = reserve(&limiter, 0.0, &p, &policy, estimated).unwrap();
        reservation.report_actual(ReservationCost {
            text_tokens: 25.0,
            ..estimated
        });
        reservation.complete();
        assert!(reserve(
            &limiter,
            0.0,
            &p,
            &policy,
            ReservationCost {
                text_tokens: 75.0,
                ..estimated
            }
        )
        .is_ok());
    }
}
