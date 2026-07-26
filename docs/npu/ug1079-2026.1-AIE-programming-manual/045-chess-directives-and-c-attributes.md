---
title: "Chess Directives and C++ Attributes"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Chess-Directives-and-C-Attributes"
toc_id: MgzpJRKYwlc9a0VnZ_TQ8w
content_id: XbWvLO8MLR631APx5JMWOw
---

## Chess Directives and C++ Attributes

AI Engine

| Chess Directive | C++ Attribute | Note |
| --- | --- | --- |
| chess_prepare_for_pipelining | [[chess::prepare_for_pipelining]] |  |
| chess_loop_range(<minimum>, <maximum>) | [[chess::min_loop_count(<minimum>)]] [[chess::max_loop_count(<maximum>)]] | In the Chess directive, the minimum and maximum loop ranges are specified in the same directive. In C++, two attributes are needed, one for minimum loop range and one for maximum loop range. |
| chess_unroll_loop(N) | [[chess::unroll_loop]] | Partial loop unrolling is not supported by the C++ attribute, hence N cannot be specified. |
| chess_storage(<reg>) | [[chess::storage(<reg>)]] |  |
