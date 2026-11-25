# BitNet Simple - Ternary Unpack + Matmul Demo

**Status**: Working prototype / Teaching Example

This is a TT-Metalium example that implements BitNet 1.58 ternary-weight inference on
Blackhole Tensix cores: 2-bit ternary weights are unpacked **on-chip by the SFPU**, fused
with an Int8×ternary matmul, and the work is distributed across many cores over the NoC.

## What this demo does

For a matrix product `C[M×N] = A[M×K] × W[N×K]ᵀ` expressed in 32×32 tiles:

- Activations `A` travel as **Int8** tiles (1 KB each), one tile per 32×32 block.
- Weights `W` travel **packed**: 16 trits per `uint32`, i.e. one 4 KB packed tile expands to
  16 Int8 tiles (16× compression of the weight transport).
- The compute kernel **unpacks** packed words into Int8 tiles with the SFPU
  (`ckernel_sfpu_custom_ternary_unpack.h`) and immediately **multiplies** them by the
  activations with the FPU (`matmul_tiles`), accumulating in Int32.
- Results are written to DRAM as Int32 tiles; the host validates every element against a
  reference matmul computed on the CPU.

All of this is verified end-to-end: `PASS! Arbitrary (M=14, K=64, N=26) BitNet Matmul verified
across all 364 output tiles!`

## Ternary encoding

```
2 bits per trit, 16 trits per uint32:
  00 -> 0
  01 -> +1
  10 -> -1
  11 -> 0
```

Decoding is branchless on the SFPU: `diff = ((w >> 2k) & 1) - ((w >> 2k+1) & 1)`.
The value is stored into DEST in the Blackhole integer-FPU sign-magnitude form
(`diff & 0x80000001`), which the host converts back with `tensix_sm_to_int32`.

## Multi-core scaling

Output columns `N_TILES` are split across the whole compute grid with
`tt::tt_metal::split_work_to_cores(compute_grid, N_TILES)`. Each core:

1. reads only its slice of packed weight tiles (`start_n_tile * num_k_blocks` onwards),
2. unpacks all `core_n_tiles * K_TILES` weight tiles into `c_2` in L1,
3. streams all activations and computes `core_n_tiles` output columns for all `M_TILES` rows,
4. writes results to the global output index `m * total_N_tiles + (start_n_tile + n)`.

This both removes the single-core L1 limit (the unpacked-weight buffer `c_2` now only needs
`max_n_tiles_per_core * K_TILES` Int8 tiles instead of `N_TILES * K_TILES`) and parallelizes
the compute across cores. For `M=14, K=64, N=26` on Blackhole: 26 cores, 1 column per core,
`c_2` = 64 KB per core — the single-core version would have needed 1.7 MB, above the
1.5 MB L1 limit.

## Directory structure

```
bitnet_simple/
├── CMakeLists.txt                    # Build configuration
├── bitnet_unpack_demo.cpp            # Host application (main)
├── README.md                         # This file
└── kernels/
    ├── compute/
    │   └── ternary_unpack.cpp        # SFPU unpacking + FPU matmul (fused)
    └── dataflow/
        ├── read_packed.cpp           # Reader kernel (DRAM → L1), per-core N slice
        └── write_unpacked.cpp        # Writer kernel (L1 → DRAM), global output idx
```

## Building

```bash
cmake -B build -S . -GNinja
ninja -C build metal_example_bitnet_simple
```

## Running

### On simulator (no hardware)

```bash
export TT_METAL_HOME=~/tt/tt-metal
export TT_METAL_SIMULATOR=~/tt/sim/libttsim_bh.so
export ARCH_NAME=blackhole
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/programming_examples/metal_example_bitnet_simple
```

### On hardware

```bash
export ARCH_NAME=blackhole
./build/programming_examples/metal_example_bitnet_simple
```

The compute kernel uses `ckernel_sfpu_custom_ternary_unpack.h`, which is Blackhole-only.

## Configuration

Matrix dimensions are compile-time constants in `bitnet_unpack_demo.cpp`:

```cpp
constexpr uint32_t M_TILES = 14;  // rows
constexpr uint32_t K_TILES = 64;  // inner depth; must be a multiple of 16
constexpr uint32_t N_TILES = 26;  // output columns
```

The weight test pattern is selected with:

```cpp
const std::string test_pattern = "balanced";
```

| Pattern | Description |
|---------|-------------|
| `alternating` | -1, 0, +1, 0, ... |
| `sparse` | 75% zeros, 12.5% +1, 12.5% -1 |
| `balanced` | Equal distribution of -1, 0, +1 |
| `all_zeros` | All zeros |
| `all_ones` | All +1 |

## Expected output

```
Multi-core Work Split:
  - Compute grid:         11 x 10
  - Active cores:         26
  - Max N_tiles per core: 1
  - c_2 per-core size:    65536 bytes

Setting up circular buffers...
Creating kernels...
Uploading input data to device...
Executing kernel...
Reading results from device...
Validating device results against host reference...
PASS! Arbitrary (M=14, K=64, N=26) BitNet Matmul verified across all 364 output tiles!
```

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Host (CPU)                                  │
│  split_work_to_cores → per-core runtime args; validates all tiles  │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ PCIe
                               ↓
┌───────────────────────────────┬────────────────────────────────────┐
│  Tensix core k (of num_cores) │  ... (one per N-column slice)      │
├───────────────────────────────┼────────────────────────────────────┤
│ Reader (RISC0) → c_0 packed   │ Compute: SFPU unpack → c_2 → FPU   │
│ Reader (RISC0) → c_1 activ.   │ matmul → c_16 Int32                │
│                               │ Writer (RISC1): c_16 → DRAM        │
└───────────────────────────────┴────────────────────────────────────┘
```

| Kernel | Purpose | Key operations |
|--------|---------|----------------|
| `read_packed.cpp` | DRAM → L1: own weight slice + activations | `noc_async_read`, `cb_push_back` |
| `ternary_unpack.cpp` | SFPU unpack 16 trits/word + FPU matmul | `copy_tile`, `unpack_row_batch`, `matmul_tiles`, `pack_tile` |
| `write_unpacked.cpp` | L1 → DRAM at global tile index | `noc_async_write`, `cb_pop_front` |

## Known limitations

1. **Blackhole-only**: the custom SFPU unpack header exists only for Blackhole
   (`tt_metal/hw/ckernels/blackhole/...`); there is no Wormhole counterpart.
2. **Unpack buffer is fully materialized in L1**: `c_2` holds all `core_n_tiles * K_TILES`
   unpacked weight tiles at once. This is what makes the multi-core split necessary —
   and it caps `N_TILES / num_cores` relative to L1 size.
3. **Activations are re-read per output column**: the reader streams `A` once per
   `core_n_tiles`; a future optimization could cache them or multicast.
4. **Debug build path**: the example is run via `TT_METAL_SLOW_DISPATCH_MODE=1` on the
   simulator; performance is not tuned.

## References

- [TT-Metalium Guide](../../../../METALIUM_GUIDE.md)
- [Programming Examples](../)
- [LLMs in TT-NN](../../../../tech_reports/LLMs/llms.md)
