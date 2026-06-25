@./AGENTS.md
# HipFire: RDNA GPU Unlock & Rust-Native Inference Engine

## Orchestration Model

You (Claude Code Opus) are the orchestrator. You make all architectural decisions.
You dispatch Sonnet subagents via the Task tool for parallel work.
You synthesize their findings and decide what to test and in what order.

**Reasoning budget:** You are running at max reasoning effort. Think hard at every
phase transition. The subagents are cheaper — dispatch them liberally for scoped tasks.

**Experiment tracking:** Git-commit every meaningful state change. Every approach tested
gets a commit with structured results. Failed approaches are just as valuable as
successful ones — document WHY they failed so the search space narrows.
