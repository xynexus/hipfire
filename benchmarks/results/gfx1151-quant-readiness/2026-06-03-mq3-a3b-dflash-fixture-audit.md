# MQ3 A3B DFlash Fixture Audit

- date: 2026-06-03
- arch: gfx1151
- commit: fab9d2bc88d2de7b5febfa9ec8afed80b6700557
- branch: qwen35-native-mtp
- scope: local A3B DFlash/spec fixture availability for MQ3 readiness

This audit records why A3B DFlash/spec rows are not currently runnable for the
MQ3 MoE promotion lane. The blocker is fixture availability, not a measured
runtime failure.

## Search

Searched local hipfire model storage and the source model inventory for A3B
DFlash or draft sidecars:

```bash
rg --files /home/sadara/.hipfire /home/sadara/Models \
  -g '*a3b*dflash*' \
  -g '*dflash*a3b*' \
  -g '*a3b*draft*' \
  -g '*draft*a3b*' \
  -g '*qwen36-35b-a3b-dflash*' \
  -g '*qwen35-35b-a3b-dflash*'
```

Result: no matches.

The dense 27B DFlash draft available locally is:

```text
/home/sadara/.hipfire/models/qwen35-27b-dflash-mq4.hfq
```

That draft is only valid for the existing dense 27B target-side MQ3/MQ4
DFlash rows. It is not a paired A3B draft sidecar.

## Available A3B targets

Local A3B target artifacts are present:

```text
/home/sadara/.hipfire/models/qwen3.5-35b-a3b.mq3
/home/sadara/.hipfire/models/qwen3.5-35b-a3b.mq4
/home/sadara/.hipfire/models/qwen3.5-35b-a3b.mq6
/home/sadara/.hipfire/models/qwen3.6-35b-a3b.mq3
/home/sadara/.hipfire/models/qwen3.6-35b-a3b.mq4
/home/sadara/.hipfire/models/qwen3.6-35b-a3b.mq6
/home/sadara/.hipfire/models/qwen3.6-35b-a3b-fp16.hfq
```

The readiness candidate directory also exposes canonical target symlinks for
Qwen3.5/Qwen3.6 A3B MQ3, MQ4, and MQ6. It does not contain A3B DFlash sidecar
symlinks.

## Decision

Do not record A3B DFlash/spec as a failed MQ3 runtime result. It is blocked on
a paired, manifest-pinned A3B draft sidecar for the target fixture.

Before using A3B DFlash/spec as promotion evidence:

- generate or locate a paired Qwen3.5 and/or Qwen3.6 35B-A3B DFlash draft;
- expose it with the artifact naming convention, for example
  `qwen3.5-35b-a3b-mq4.dflash.hfq` or a documented legacy equivalent;
- record size, checksum, and source provenance;
- then run fresh-process A3B DFlash/spec rows against the MQ4 control and MQ3
  candidate with byte-identical committed prompts.
