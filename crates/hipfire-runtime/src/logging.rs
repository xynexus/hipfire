// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use std::io::Write;
use std::sync::OnceLock;

static LOGGING_INIT: OnceLock<()> = OnceLock::new();

/// True when stderr is the journald stream systemd gave this service.
/// `$JOURNAL_STREAM` holds "dev:inode" of that stream, but it is inherited by
/// descendants (e.g. shells under a terminal emulator's user service), so it
/// only counts if it names our actual stderr.
pub fn stderr_is_journal() -> bool {
    let Some((dev, ino)) = std::env::var("JOURNAL_STREAM").ok().and_then(|v| {
        let (d, i) = v.split_once(':')?;
        Some((d.parse::<u64>().ok()?, i.parse::<u64>().ok()?))
    }) else {
        return false;
    };
    let mut st = unsafe { std::mem::zeroed::<libc::stat>() };
    let ok = unsafe { libc::fstat(libc::STDERR_FILENO, &mut st) } == 0;
    ok && st.st_dev == dev && st.st_ino == ino
}

/// Stderr writer that prefixes each event with an sd-daemon `<N>` syslog
/// priority so journald records real severities (`journalctl -p err` works).
/// systemd parses these by default (`SyslogLevelPrefix=true`).
struct JournalStderr;

struct PriorityLine {
    pri: u8,
    first: bool,
}

impl Write for PriorityLine {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut err = std::io::stderr().lock();
        if self.first {
            // ponytail: fmt::Layer emits one write per event, so prefixing the
            // first write covers the whole line; a multi-line message logs its
            // continuation lines at default priority.
            err.write_all(&[b'<', self.pri, b'>'])?;
            self.first = false;
        }
        err.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for JournalStderr {
    type Writer = PriorityLine;

    fn make_writer(&'a self) -> Self::Writer {
        PriorityLine {
            pri: b'6',
            first: true,
        }
    }

    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        let pri = match *meta.level() {
            tracing::Level::ERROR => b'3',
            tracing::Level::WARN => b'4',
            tracing::Level::INFO => b'6',
            tracing::Level::DEBUG | tracing::Level::TRACE => b'7',
        };
        PriorityLine { pri, first: true }
    }
}

/// Initialize structured logging for runtime binaries.
///
/// The daemon reserves stdout for JSONL IPC, so this writes to stderr for now.
/// File sinks should be added here later so daemon and server business logic
/// does not learn about production log destinations.
///
/// Filter precedence: `HIPFIRE_LOG`, then `RUST_LOG`, then `default_filter`.
/// Under a systemd journal stream, events carry `<N>` severity prefixes and
/// drop the timestamp (journald stamps entries itself).
pub fn init_stderr_logging(component: &'static str, default_filter: &str) {
    LOGGING_INIT.get_or_init(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_env("HIPFIRE_LOG")
            .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter));

        let builder = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            // Plain text when stderr is a pipe, file, or the journal.
            .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()));

        // The daemon worker's stderr is a pipe relayed line-verbatim to the
        // front-end's stderr (see hipfire-daemon-adapter), so the child can't
        // detect the journal itself; the journal-connected parent exports
        // HIPFIRE_LOG_JOURNAL so the whole process tree formats for journald.
        // ponytail: the daemon.log tee then also carries <N> prefixes — strip
        // them in the adapter if that ever bothers anyone.
        let journal = stderr_is_journal()
            || std::env::var_os("HIPFIRE_LOG_JOURNAL").is_some_and(|v| v == "1");
        if journal {
            std::env::set_var("HIPFIRE_LOG_JOURNAL", "1");
            builder
                .with_writer(JournalStderr)
                .without_time()
                .with_level(false)
                .init();
        } else {
            builder.with_writer(std::io::stderr).init();
        }

        tracing::debug!(component, sink = "stderr", "logging initialized");
    });
}
