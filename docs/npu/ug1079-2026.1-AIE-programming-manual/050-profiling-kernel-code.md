---
title: "Profiling Kernel Code"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Profiling-Kernel-Code"
toc_id: DHxUbaJrvCKEdh3dBDE0Jg
content_id: Naup_r~gmpdqjH0yezlTuA
---

## Profiling Kernel Code

AI Engine

`aie::tile`

`cycles()`

```
aie::tile tile=aie::tile::current(); //get the tile of the kernel
unsigned long long time=tile.cycles();//cycle counter of the tile counter
```

```
aie::tile tile=aie::tile::current();
unsigned long long time1=tile.cycles(); //first time

for(...){...}

unsigned long long time2=tile.cycles(); //second time
long long time=time2-time1;
writeincr(out,time);
```

The latency of the loop in the kernel can then be examined in the host application by the second time minus the first time.

Compare the data read back between different kernel executions or loop iterations to calculate latency. For example, the following code tries to get the latency of certain operations on an asynchronous buffer:

```
aie::tile tile=aie::tile::current();
for(...){//outer loop
  unsigned long long time=tile.cycles(); //read counter value
  writeincr(out,time);
  win_in.acquire();
  for(...){...} //inner loop
  win_in.release();
}
```

The latency of asynchronous buffer acquiring and release, plus the inner loop execution time can then be calculated by the second time minus the first time.

`printf`

`volatile`

```
static unsigned long long cycle_num[2];
aie::tile tile=aie::tile::current();
volatile unsigned long long *p_cycle=cycle_num;
*p_cycle=tile.cycles();//cycle_num[0]

for(...){...}

*(p_cycle+1)=tile.cycles();//cycle_num[1]
printf("cycles=%lld\n",cycle_num[1]-cycle_num[0]);
```
