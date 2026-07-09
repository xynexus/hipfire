---
title: "Throughput Measurements using Event Trace"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Throughput-Measurements-using-Event-Trace"
toc_id: nhx4dOraGB0kbLQRBec8xQ
content_id: _kR5Ly0iIgz1QZ_8vteNPQ
---

### Throughput Measurements using Event Trace

Using event trace for throughput measurements requires that you compile the AI Engine application using AI Engine compiler option `--event-trace=runtime`. Then run the complete system in hardware using the `XRT` or `XSDB` flow to extract event trace.

You can visualize the trace data after extraction within `vitis -a`. An example of trace data is as follows:

You can position markers where you want to compute the data throughput. Use icons ![vqd1681381270754.png](../assets/196-01-vqd1681381270754-png-dbb66d7ab06c.png) to precisely position on signal edges. Use icon ![nyb1681381207315.png](../assets/196-02-nyb1681381207315-png-b5d587840ec5.png) to add a new marker at the exact position of the mobile marker.

*Figure 1. First Frame and Last Frame Marker Position*

![ctc1681381720990.png](../assets/196-03-ctc1681381720990-png-90985d9ffeaf.png) ![snv1681381750782.png](../assets/196-04-snv1681381750782-png-7086fa39671d.png)

Click the flag of the fixed marker which is on the first frame edge to define it as the origin time. Relative time is shown on the time scale and at the bottom of all other markers:

![xcp1681381991956.png](../assets/196-05-xcp1681381991956-png-b14d8381555b.png)

*Figure 2. Relative Time Displayed on Time Scale and at Bottom of all Other Markers*

Using the number of samples per kernel run and the number of frames processed, you can calculate average throughput over the first-to-last frame duration.
