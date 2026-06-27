.PHONY: build install

build:
	cargo build --release

install:
	./install.sh

# Developer install: symlink in-tree release binaries into ~/.local/bin and
# ~/.hipfire/bin so `cargo build --release` updates installed commands in place
# (no stale copies). Run `make build` first, or `scripts/dev-link.sh --build`.
.PHONY: link
link:
	./scripts/dev-link.sh
