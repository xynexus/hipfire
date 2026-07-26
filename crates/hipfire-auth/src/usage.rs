use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{AccessStore, AuthError, HourlyUsageRecord};

pub const DEFAULT_USAGE_RETENTION_SECS: u64 = 90 * 24 * 60 * 60;

enum Command {
    Record(HourlyUsageRecord),
    Flush {
        now: u64,
        reply: mpsc::SyncSender<Result<(), String>>,
    },
    Shutdown,
}

/// Cloneable, non-blocking handle to the dedicated redb usage writer.
#[derive(Clone)]
pub struct UsageWriter {
    inner: Arc<UsageWriterInner>,
}

struct UsageWriterInner {
    sender: mpsc::Sender<Command>,
    last_error: Arc<Mutex<Option<String>>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for UsageWriterInner {
    fn drop(&mut self) {
        let _ = self.sender.send(Command::Shutdown);
        if let Some(thread) = self.thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

impl std::fmt::Debug for UsageWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsageWriter")
            .field("last_error", &self.last_error())
            .finish_non_exhaustive()
    }
}

impl UsageWriter {
    pub fn spawn(store: AccessStore) -> Self {
        Self::spawn_with_options(
            store,
            Duration::from_secs(1),
            DEFAULT_USAGE_RETENTION_SECS,
            256,
        )
    }

    pub fn spawn_with_options(
        store: AccessStore,
        flush_interval: Duration,
        retention_secs: u64,
        max_batch: usize,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        let last_error = Arc::new(Mutex::new(None));
        let actor_error = last_error.clone();
        let thread = std::thread::Builder::new()
            .name("hipfire-usage-store".to_string())
            .spawn(move || {
                run_actor(
                    store,
                    receiver,
                    actor_error,
                    flush_interval,
                    retention_secs,
                    max_batch.max(1),
                )
            })
            .expect("failed to spawn hipfire usage storage actor");
        Self {
            inner: Arc::new(UsageWriterInner {
                sender,
                last_error,
                thread: Mutex::new(Some(thread)),
            }),
        }
    }

    pub fn record(&self, record: HourlyUsageRecord) -> Result<(), String> {
        self.inner
            .sender
            .send(Command::Record(record))
            .map_err(|_| "usage storage actor stopped".to_string())
    }

    pub fn flush(&self, now: u64) -> Result<(), String> {
        let (reply, response) = mpsc::sync_channel(1);
        self.inner
            .sender
            .send(Command::Flush { now, reply })
            .map_err(|_| "usage storage actor stopped".to_string())?;
        response
            .recv()
            .map_err(|_| "usage storage actor stopped".to_string())?
    }

    pub fn last_error(&self) -> Option<String> {
        self.inner.last_error.lock().unwrap().clone()
    }
}

fn run_actor(
    store: AccessStore,
    receiver: mpsc::Receiver<Command>,
    last_error: Arc<Mutex<Option<String>>>,
    flush_interval: Duration,
    retention_secs: u64,
    max_batch: usize,
) {
    let mut pending = HashMap::<String, HourlyUsageRecord>::new();
    loop {
        match receiver.recv_timeout(flush_interval) {
            Ok(Command::Record(record)) => {
                let key = record_key(&record);
                pending
                    .entry(key)
                    .and_modify(|current| current.counters += record.counters)
                    .or_insert(record);
                if pending.len() >= max_batch {
                    remember_result(
                        &last_error,
                        flush_pending(&store, &mut pending, now_secs(), retention_secs),
                    );
                }
            }
            Ok(Command::Flush { now, reply }) => {
                let result = flush_pending(&store, &mut pending, now, retention_secs)
                    .map_err(|error| error.to_string());
                remember_string_result(&last_error, &result);
                let _ = reply.send(result);
            }
            Ok(Command::Shutdown) => {
                remember_result(
                    &last_error,
                    flush_pending(&store, &mut pending, now_secs(), retention_secs),
                );
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => remember_result(
                &last_error,
                flush_pending(&store, &mut pending, now_secs(), retention_secs),
            ),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                remember_result(
                    &last_error,
                    flush_pending(&store, &mut pending, now_secs(), retention_secs),
                );
                break;
            }
        }
    }
}

fn flush_pending(
    store: &AccessStore,
    pending: &mut HashMap<String, HourlyUsageRecord>,
    now: u64,
    retention_secs: u64,
) -> Result<(), AuthError> {
    let drained = pending.drain().map(|(_, value)| value).collect::<Vec<_>>();
    let mut records = drained.into_iter();
    while let Some(record) = records.next() {
        if let Err(error) = store.add_usage(&record) {
            for record in std::iter::once(record).chain(records) {
                let key = record_key(&record);
                pending
                    .entry(key)
                    .and_modify(|current| current.counters += record.counters)
                    .or_insert(record);
            }
            return Err(error);
        }
    }
    let cutoff = now
        .saturating_sub(retention_secs)
        .saturating_div(3600)
        .saturating_mul(3600);
    store.prune_usage_before(cutoff)?;
    Ok(())
}

fn record_key(record: &HourlyUsageRecord) -> String {
    format!(
        "{}\0{}\0{}\0{}",
        record.hour_start, record.user_id, record.token_id, record.workload
    )
}

fn remember_result(last_error: &Mutex<Option<String>>, result: Result<(), AuthError>) {
    let result = result.map_err(|error| error.to_string());
    remember_string_result(last_error, &result);
}

fn remember_string_result(last_error: &Mutex<Option<String>>, result: &Result<(), String>) {
    let mut slot = last_error.lock().unwrap();
    match result {
        Ok(()) => *slot = None,
        Err(error) => *slot = Some(error.clone()),
    }
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
    use crate::UsageCounters;

    #[test]
    fn actor_batches_rollups_and_prunes_retention() {
        let dir = tempfile::tempdir().unwrap();
        let store = AccessStore::open_in(dir.path()).unwrap();
        let writer =
            UsageWriter::spawn_with_options(store.clone(), Duration::from_secs(60), 3600, 256);
        let record = HourlyUsageRecord {
            hour_start: 3_600,
            user_id: "u".into(),
            token_id: "t".into(),
            workload: "text".into(),
            counters: UsageCounters {
                requests: 1,
                input_tokens: 10,
                ..Default::default()
            },
        };
        writer.record(record.clone()).unwrap();
        writer.record(record).unwrap();
        writer.flush(7_200).unwrap();
        let rows = store.list_usage().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].counters.requests, 2);
        assert_eq!(rows[0].counters.input_tokens, 20);

        writer.flush(10_801).unwrap();
        assert!(store.list_usage().unwrap().is_empty());
    }
}
