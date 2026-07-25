//! Request handlers, one module per family.
//!
//! Each handler is the body of what used to be an inline arm of the ~4.7k-line
//! `match request` in `main()`. They take `&mut DaemonState` (see `state.rs`)
//! plus whatever the arm read off the raw message.
//!
//! Two conventions come from the shape of the loop these were lifted out of:
//!
//! - Handlers return `()`. In the old loop an arm's `continue` meant "this
//!   request is finished, read the next line", and nothing followed the match
//!   inside the loop body — so `continue` became `return` with no change in
//!   behaviour. Only one `break` ever targeted the read loop and it lives in the
//!   loop preamble, not an arm, so no handler needs to signal loop exit.
//! - Handlers write their own response frames to `daemon_state.out.sink` rather
//!   than returning a value to be serialised. That is how progress frames
//!   interleave safely today: each write is a whole locked line on a single
//!   thread. It is also the thing a multi-client transport has to change.

pub mod batch;
pub mod calibrate;
pub mod diag;
pub mod generate;
pub mod hneurons;
pub mod lifecycle;
pub mod lora;
pub mod sessions;
pub mod status;
pub mod steer;
pub mod train;
