---
title: "Code Example for Sharing RTPs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Code-Example-for-Sharing-RTPs"
toc_id: Y5GtJtYHPj6bA037aV3NFg
content_id: KNkrfAZLEA2nrBp3BkDtMg
---

### Code Example for Sharing RTPs

```
// in kernels.cpp
// --------------

void func1(const int32 (&rtp_value)[8], ...) {
  // use rtp_value array
}

void func2(const int32 (&rtp_value)[8], ...) {
  // use rtp_value array
}

void func3(const int32 (&rtp_value)[8], ...) {
  // use rtp_value array
}


// In graph definition graph.h
// ---------------------------

kernel1 = adf::kernel::create(func1);
kernel2 = adf::kernel::create(func2);
kernel3 = adf::kernel::create(func3);

adf::connect<adf::parameter>(value1, adf::async(kernel1.in[0]));
adf::connect<adf::parameter>(value2, adf::async(kernel2.in[0]));
adf::connect<adf::parameter>(value3, adf::async(kernel3.in[0]));

adf::initial_value(kernel1.in[0]) = {10,20,30,40,50,60,70,80};

// Constraints to place slave kernels in the neighboring tiles of master kernel1
adf::location<kernel>(kernel1) = tile(25,0);
adf::location<kernel>(kernel2) = tile(25,1);
adf::location<kernel>(kernel3) = tile(26,0);

// Constraint to place the master parameter in the same tile as the kernel
adf::location<adf::buffer>(kernel1.in[0]) = adf::location<kernel>(kernel1);

// use new graph construct for shared RTP parameters
adf::share_parameter(kernel1.in[0], {kernel2.in[0], kernel3.in[0]});

// in graph.cpp, will show how to update the master RTP...
  gr.init();
  gr.run(1); //uses initial_value from graph.h
  gr.wait();
  int vec[8] = {101,202,303,404,505,606,707,808};
  for(int i=0;i<3;i++){
    for(int i=0;i<8;i++) vec[i]*=10;
    gr.update(gr.value1,vec,8); // update only master RTP
    gr.run(1);
    gr.wait();
  }
  gr.end();
```
