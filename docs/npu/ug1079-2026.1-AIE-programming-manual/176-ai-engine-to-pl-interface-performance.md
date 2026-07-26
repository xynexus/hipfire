---
title: "AI Engine to PL Interface Performance"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-to-PL-Interface-Performance"
toc_id: tn_KI5woRx9kWc9Lgd8XOg
content_id: U1_xKjEnXZUNdIjZ4jRTSw
---

## AI Engine to PL Interface Performance

Versal AI Core Series devices include an AI Engine array with the following column categories.

- **PL column:** Provides PL stream access. Each column supports:Eight 64-bit slave channels for streaming data into the AI Engine Six 64-bit master channels for streaming data to the PL
- **NoC column:** Provides connectivity between the AI Engine array and the NoC. These interfaces can also connect to the PL.

To instruct the AI Engine compiler to select higher frequency interfaces, use the `--pl-freq=<number>` to specify the clock frequency (in MHz) for the PL kernels. The default value is one quarter of the AI Engine frequency and the maximum supported value is a half of the AI Engine frequency. The values depend on the speed grade. Following are examples:

- Option to enable an AI Engine to PL frequency of 300 MHz for all AI Engine to PL interfaces: `--pl-freq=300`

  ```
  --pl-freq=300
  ```
- To set a different frequency for a specific PLIO interface use the following code to set it in the ADF graph.`adf::PLIO *<input>= new adf::PLIO(<logical_name>, <plio_width>, <file>, <FreqMHz>);`

  ```
  adf::PLIO *<input>= new adf::PLIO(<logical_name>, <plio_width>, <file>, <FreqMHz>);
  ```

**Note:** AI Engine

Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine))

The AI Engine to PL AXI4-Stream channels use boundary logic interface (BLI) connections. These connections include optional BLI registers with the exception of the slave channels 3 and 7. Slave channels 3 and 7 have slower interfaces. Data transfer performance between the AI Engine and PL depends on whether optional BLI registers are enabled.

You can use all eight channels without BLI registers for less timing-critical designs. PL timing can still be met in this case.

For higher frequency designs, only the six fast channels (0,1,2,4,5,6) can be used. In these cases, you must register the timing paths from the PL using the BLI registers.

To control the use of BLI registers across the AI Engine to PL channels, use the `--pl-register-threshold=<number>` compiler option, specified in MHz. The default value is 1/8 of the AI Engine frequency based on speed grade. Following is an example:

- `–pl-register-threshold=125` If the AI Engine to PL frequency exceeds the set threshold (125 MHz in this case), the compiler maps the PLIO interface to high-speed channels with BLI registers enabled. If the PLIO interface frequency does not exceed the `pl-register-threshold` value, then any of the AI Engine to PL channels are used.

  ```
  –pl-register-threshold=125
  ```

  If the AI Engine to PL frequency exceeds the set threshold (125 MHz in this case), the compiler maps the PLIO interface to high-speed channels with BLI registers enabled. If the PLIO interface frequency does not exceed the `pl-register-threshold` value, then any of the AI Engine to PL channels are used.

In summary:

- If `pl-freq` < `pl-register-threshold` all eight channels can be used unregistered.
- If `pl-freq` > `pl-register-threshold` only the six fast channels can be used, with registering.

`pl-register-threshold` is a way to control the threshold frequency beyond which only fast channels can be used (with registering).
