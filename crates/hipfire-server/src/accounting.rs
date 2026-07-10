//! Per-request accounting shared by middleware and workload handlers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use hipfire_auth::{
    HourlyUsageRecord, RateLimitReporter, RequestPrincipal, ReservationCost, UsageCounters,
    UsageWriter, WorkloadClass,
};

#[derive(Debug, Clone)]
pub struct RequestAccounting {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    principal: RequestPrincipal,
    workload: WorkloadClass,
    estimated: ReservationCost,
    estimated_images: u64,
    rate: RateLimitReporter,
    usage: Mutex<UsageCounters>,
    writer: Option<UsageWriter>,
    finalized: AtomicBool,
}

impl RequestAccounting {
    pub fn new(
        principal: RequestPrincipal,
        estimated: ReservationCost,
        rate: RateLimitReporter,
        writer: Option<UsageWriter>,
        estimated_images: u64,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                principal,
                workload: estimated.workload,
                estimated,
                estimated_images,
                rate,
                usage: Mutex::new(UsageCounters {
                    requests: 1,
                    ..Default::default()
                }),
                writer,
                finalized: AtomicBool::new(false),
            }),
        }
    }

    pub fn report_text(&self, input_tokens: u64, output_tokens: u64, cache_tokens: u64) {
        let mut usage = self.inner.usage.lock().unwrap();
        usage.input_tokens = input_tokens;
        usage.output_tokens = output_tokens;
        usage.cache_tokens = cache_tokens;
        drop(usage);
        self.inner.rate.report(ReservationCost {
            text_tokens: input_tokens.saturating_add(output_tokens) as f64,
            ..self.inner.estimated
        });
    }

    pub fn report_images(&self, images: u64, megapixel_steps: f64) {
        let mut usage = self.inner.usage.lock().unwrap();
        usage.images = images;
        usage.megapixel_steps = megapixel_steps.ceil().max(0.0) as u64;
        drop(usage);
        self.inner.rate.report(ReservationCost {
            megapixel_steps,
            ..self.inner.estimated
        });
    }

    pub fn report_training_seconds(&self, seconds: u64) {
        self.inner.usage.lock().unwrap().training_seconds = seconds;
    }

    pub fn mark_error(&self) {
        self.inner.usage.lock().unwrap().errors = 1;
    }

    pub fn complete(self) {
        self.finalize(false);
    }

    pub fn fail(self) {
        self.finalize(true);
    }

    fn finalize(&self, failed: bool) {
        if self.inner.finalized.swap(true, Ordering::AcqRel) {
            return;
        }
        if failed {
            self.inner.usage.lock().unwrap().errors = 1;
        } else if self.inner.workload == WorkloadClass::Image {
            let mut usage = self.inner.usage.lock().unwrap();
            if usage.images == 0 {
                usage.images = self.inner.estimated_images;
                usage.megapixel_steps = self.inner.estimated.megapixel_steps.ceil() as u64;
            }
        }
        let Some(writer) = &self.inner.writer else {
            return;
        };
        let now = now_secs();
        let record = HourlyUsageRecord {
            hour_start: now / 3600 * 3600,
            user_id: self
                .inner
                .principal
                .user_id
                .clone()
                .unwrap_or_else(|| "anonymous-local".to_string()),
            token_id: self
                .inner
                .principal
                .token_id
                .clone()
                .unwrap_or_else(|| "anonymous-local".to_string()),
            workload: workload_name(self.inner.workload).to_string(),
            counters: *self.inner.usage.lock().unwrap(),
        };
        if let Err(error) = writer.record(record) {
            tracing::error!(error = %error, "failed to queue API usage rollup");
        }
    }
}

impl Drop for RequestAccounting {
    fn drop(&mut self) {
        // Only the middleware-owned final clone observes a disconnect. Handler
        // extractor clones disappear earlier while the stream remains alive.
        if Arc::strong_count(&self.inner) == 1 {
            self.finalize(true);
        }
    }
}

pub fn record_rate_limit_hit(
    writer: Option<&UsageWriter>,
    principal: &RequestPrincipal,
    workload: WorkloadClass,
) {
    let Some(writer) = writer else { return };
    let now = now_secs();
    let record = HourlyUsageRecord {
        hour_start: now / 3600 * 3600,
        user_id: principal
            .user_id
            .clone()
            .unwrap_or_else(|| "anonymous-local".to_string()),
        token_id: principal
            .token_id
            .clone()
            .unwrap_or_else(|| "anonymous-local".to_string()),
        workload: workload_name(workload).to_string(),
        counters: UsageCounters {
            requests: 1,
            errors: 1,
            rate_limit_hits: 1,
            ..Default::default()
        },
    };
    if let Err(error) = writer.record(record) {
        tracing::error!(error = %error, "failed to queue rate-limit usage rollup");
    }
}

fn workload_name(workload: WorkloadClass) -> &'static str {
    match workload {
        WorkloadClass::Other => "other",
        WorkloadClass::Text => "text",
        WorkloadClass::Image => "image",
        WorkloadClass::Training => "training",
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
