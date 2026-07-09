---
title: "Load and Store Using Pointers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-Using-Pointers"
toc_id: nh1bM1mHyW9ZuwD3bjAwRw
content_id: JYqKOzAfBp3kU4nYpC8N4g
---

#### Load and Store Using Pointers

It is mandatory to use the window API in the kernel function prototype as inputs and outputs. However, in the kernel code, it is possible to use a direct pointer reference to read/write data.

```
void func(input_window_int16 *w_input,
			output_window_cint16 *w_output){
	.....
	v16int16 *ptr_in  = (v16int16 *)w_input->ptr;
	v8cint16 *ptr_out = (v8cint16 *)w_output->ptr;
	......
}
```

The window structure is responsible for managing buffer locks tracking buffer type (ping/pong) and this can add to the cycle count. This is especially true when load/store are out-of-order (scatter-gather). Using pointers can help reduce the cycle count required for load and store.

**Note:** If using pointers to load and store data, it is the designer’s responsibility to avoid out-of-bound memory access.
