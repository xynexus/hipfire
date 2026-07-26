---
title: "Multiple Lanes Multiplication - sliding_mul"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Multiple-Lanes-Multiplication-sliding_mul"
toc_id: hHzZk7rk5cbT~3QFzuFeMA
content_id: DPzqCKzFZ_Ms_iiq5oT9GA
---

## Multiple Lanes Multiplication - sliding_mul

AI Engine provides hardware support to accelerate a type of multiple lanes multiplication, called sliding multiplication. Sliding multiplication allows multiple lanes to do MAC operations simultaneously, and adds the results to an accumulator. Sliding multiplication works especially well with (but not limited to) finite impulse response (FIR) filter implementations.

These special multiplication structures or APIs are named `aie::sliding_mul*`. They accept coefficient and data inputs. Some variants of `aie::sliding_mul_sym*` allow pre-adding of the data input symmetrically before multiplication. These classes include the following:

- `aie::sliding_mul_ops`
- `aie::sliding_mul_x_ops`
- `aie::sliding_mul_y_ops`
- `aie::sliding_mul_xy_ops`
- `aie::sliding_mul_sym_ops`
- `aie::sliding_mul_sym_x_ops`
- `aie::sliding_mul_sym_y_ops`
- `aie::sliding_mul_sym_xy_ops`
- `aie::sliding_mul_sym_uct_ops`

For more information about these APIs and supported parameters, see the AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

For example, the `aie::sliding_mul_ops` class provides a parametrized multiplication that implements the following compute pattern.

```
DSX = DataStepX
DSY = DataStepY
CS = CoeffStep
P = Points
L = Lanes
c_s = coeff_start
d_s = data_start
out[0] = coeff[c_s] * data[d_s + 0] + coeff[c_s + CS] * data[d_s + DSX] + ... + coeff[c_s + (P-1) * CS] * data[d_s + (P-1) * DSX]
out[1] = coeff[c_s] * data[d_s + DSY] + coeff[c_s + CS] * data[d_s + DSY + DSX] + ... + coeff[c_s + (P-1) * CS] * data[d_s + DSY + (P-1) * DSX]
...
out[L-1] = coeff[c_s] * data[d_s + (L-1) * DSY] + coeff[c_s + CS] * data[d_s + (L-1) * DSY + DSX] + ... + coeff[c_s + (P-1) * CS] * data[d_s + (L-1) * DSY + (P-1) * DSX]
```

| Parameter | Description |
| --- | --- |
| Lanes | Number of output elements. |
| Points | Number of data elements used to compute each lane. |
| CoeffStep | Step used to select elements from the coeff register. This step is applied to element selection within a lane. |
| DataStepX | Step used to select elements from the data register. This step is applied to element selection within a lane. |
| DataStepY | Step used to select elements from the data register. This step is applied to element selection across lanes. |
| CoeffType | Coefficient element type. |
| DataType | Data element type. |
| AccumTag | Accumulator tag that specifies the required accumulation bits. The class must be compatible with the result of the multiplication of the coefficient and data types (real/complex). |

The following figure shows how to use the `aie::sliding_mul_ops` class and its member function, `mul`, to perform the sliding multiplication. The figure also shows how each parameter corresponds to the multiplication.

![ylc1634593327535.png](../assets/037-01-ylc1634593327535-png-c7b1c0ed6c28.png)

*Figure 1. sliding_mul_ops Usage Example*

Besides the `aie::sliding_mul*` classes, AI Engine API provides `aie::sliding_mul*` functions to do sliding multiplication and `aie::sliding_mac*` functions to do sliding multiplication and accumulation. These functions are simply helpers, that use the `aie::sliding_mul*_ops` classes internally and are provided for convenience. These include:

- `aie::sliding_mul`
- `aie::sliding_mac`
- `aie::sliding_mul_sym`
- `aie::sliding_mac_sym`
- `aie::sliding_mul_antisym`
- `aie::sliding_mac_antisym`
- `aie::sliding_mul_sym_uct`
- `aie::sliding_mac_sym_uct`
- `aie::sliding_mul_antisym_uct`
- `aie::sliding_mac_antisym_uct`

The following examples perform asymmetric sliding multiplications (template prototypes are in comments for quick reference).

```
constexpr unsigned Lanes = 8, Points = 8;
constexpr unsigned CoeffStep = 1;
constexpr unsigned DataStepX = 1, DataStepY = 1;
using CoeffType = int16;
using DataType = int16;
using AccumTag = acc48;

aie::vector<int16,16> va;
aie::vector<int16,64> vb0,vb1;
aie::accum<acc48,8> acc = aie::sliding_mul_ops<Lanes, Points, CoeffStep, DataStepX, DataStepY, CoeffType, DataType, AccumTag>::mul(va, 0, vb0, 0);
acc = aie::sliding_mul_ops<Lanes, Points, CoeffStep, DataStepX, DataStepY, CoeffType, DataType, AccumTag>::mac(acc, va, 8, vb1, 0);
auto vout=acc.to_vector<int32>(15);

constexpr unsigned coeff_start = 0;
constexpr unsigned data_start = 0;
aie::vector<int32,32> data_buff;
aie::vector<int32,8> coeff_buff;
aie::accum<acc80,8> acc_buff = aie::sliding_mul<Lanes, Points>(coeff_buff, coeff_start, data_buff, data_start);
```

Following are symmetric sliding multiplication examples.

```
constexpr unsigned Lanes = 4, Points = 16;
constexpr unsigned CoeffStep = 1;
constexpr unsigned DataStepX = 1, DataStepY = 1;
using CoeffType = int16;
using DataType = cint16;
using AccumTag = cacc48;
constexpr unsigned coeff_start=0;
constexpr unsigned data_start=0;

aie::vector<cint16,16> data_buff;
aie::vector<int16,8> coeff_buff;
auto acc_buff = aie::sliding_mul_sym_ops<Lanes, Points, CoeffStep, DataStepX, DataStepY, CoeffType, DataType, AccumTag>::mul_sym(coeff_buff, coeff_start, data_buff, data_start);

auto acc_buff2 = aie::sliding_mul_sym<Lanes, Points, CoeffStep, DataStepX, DataStepY>(coeff_buff, coeff_start, data_buff, data_start);

constexpr unsigned ldata_start=0;
constexpr unsigned rdata_start=8;
aie::vector<cint16,16> ldata,rdata;
aie::vector<int16,8> coeff;
// symmetric sliding_mul using two data registers
auto acc = aie::sliding_mul_sym<Lanes, Points/2, CoeffStep, DataStepX, DataStepY>(coeff, coeff_start, ldata, ldata_start, rdata, rdata_start);
```

**Note:** Treat all registers in sliding multiplication as circular. After reaching the end, they wrap back to the start.

The following figures illustrate how the previous symmetric examples are computed.

![umg1634594722110.png](../assets/037-02-umg1634594722110-png-494daa97d2f1.png)

*Figure 2. sliding_mul_sym_ops Usage Example*

![swh1634593688143.png](../assets/037-03-swh1634593688143-png-7a5c4f758768.png)

*Figure 3. sliding_mul_sym Function with Two Data Registers*

#### Considerations When Using sliding_mul

Following are some restrictions to consider when using `sliding_mul`:

- Data width <=1024 bits, and Coefficient width <=256 bits
- Lanes * Points >= MACs per cycle for that type
- int8 is not supported in `sliding_mul_sym_ops` and `sliding_mul_sym_uct_ops`.
