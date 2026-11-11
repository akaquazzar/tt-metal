# BitNet Simple - Ternary Unpack Kernel Demo (Prototype)

**Status**: Teaching/Demonstration Example

This is a **prototype/skeleton** demonstrating the structure of a TT-Metal application for ternary quantization. The compute kernel is a placeholder and does not perform actual ternary unpacking (see "Known Limitations" below).

## What this demo does

This example demonstrates the TT-Metal programming model structure:

### Working Components
- ✅ **Host Application**: Set up devices, kernels, circular buffers
- ✅ **Dataflow Kernels**: DRAM ↔ L1 data transfer (reader/writer)
- ✅ **Mock Cluster Simulator**: JIT kernel compilation works correctly
- ✅ **Validation Infrastructure**: Host-side reference implementation

### Placeholder Component
- ⏸️ **Compute Kernel**: Currently just copies data without unpacking (see ternary_unpack.cpp)

**End-to-end test** will fail with mismatched results until actual SFPU unpacking is implemented.

## Directory Structure

```
bitnet_simple/
├── CMakeLists.txt                    # Build configuration
├── bitnet_unpack_demo.cpp            # Host application (main)
├── README.md                         # This file
└── kernels/
    ├── compute/
    │   └── ternary_unpack.cpp        # Compute kernel (unpacking logic)
    └── dataflow/
        ├── read_packed.cpp           # Reader kernel (DRAM → L1)
        └── write_unpacked.cpp        # Writer kernel (L1 → DRAM)
```

## Building

### Prerequisites

```bash
# Ensure you have built tt-metal with all dependencies
# See: ../../../../INSTALLING.md
```

### Build the example

```bash
# From the tt-metal root directory
cmake -B build -S . -GNinja
ninja -C build bitnet_unpack_demo

# Or build all programming examples
ninja -C build bitnet_simple
```

## Running

### On Simulator (No Hardware)

```bash
# Enable simulator mode
export ARCH_NAME=blackhole
export TT_METAL_SIMULATOR=true

# Run the demo
cd build/programming_examples
./bitnet_unpack_demo
```

### On Hardware (Tenstorrent Device)

```bash
# Select architecture based on your device
# For Wormhole:
export ARCH_NAME=wormhole

# For Blackhole:
export ARCH_NAME=blackhole

# Run the demo
cd build/programming_examples
./bitnet_unpack_demo
```

## Test Patterns

The demo supports different test patterns via the `test_pattern` variable in `bitnet_unpack_demo.cpp`:

| Pattern | Description | Typical Use |
|---------|-------------|-------------|
| `alternating` | -1, 0, +1, 0, -1, 0, ... | Initial validation |
| `sparse` | 75% zeros, 12.5% +1, 12.5% -1 | Realistic weight distribution |
| `balanced` | Equal distribution of -1, 0, +1 | Edge case testing |
| `all_zeros` | All values = 0 | Performance baseline |
| `all_ones` | All values = +1 | Edge case testing |

**To change the pattern:**

```cpp
// In bitnet_unpack_demo.cpp, line ~187:
const std::string test_pattern = "sparse";  // Change this
```

## Expected Output

```
========================================
BitNet Ternary Unpack Kernel Demo
========================================

Test Configuration:
  - Number of tiles:    4
  - Trits per tile:     1024
  - Total trits:        4096
  - Test pattern:       alternating

Generating test data...
  Packed bytes:        1024

  First 32 trits: [-1, 0, +1, 0, -1, 0, +1, 0, ...]

Creating device...
  Grid size:           8 x 10
  Using core:          {0, 0}

Setting up circular buffers...
Creating kernels...
  Reader kernel:      read_packed.cpp
  Writer kernel:      write_unpacked.cpp
  Compute kernel:     ternary_unpack.cpp

Uploading input data to device...

Executing kernel...

Reading results from device...

Validating results against host reference...

========================================
✓ Test PASSED
========================================
```

## Ternary Encoding

### Packed Format

```
Byte representation:
  Bits 7-6: Trit 3
  Bits 5-4: Trit 2
  Bits 3-2: Trit 1
  Bits 1-0: Trit 0

Decoding:
  00 -> -1.0
  01 ->  0.0
  10 ->  0.0
  11 -> +1.0
```

### Example

```c
// Packed byte: 0b10011010
// Breakdown:
//   Bits 1-0: 10 -> 0.0  (Trit 0)
//   Bits 3-2: 11 -> +1.0 (Trit 1)
//   Bits 5-4: 01 -> 0.0  (Trit 2)
//   Bits 7-6: 10 -> 0.0  (Trit 3)
// Result: [0.0, +1.0, 0.0, 0.0]
```

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Host (CPU)                               │
└─────────────────┬───────────────────────────────────────────┘
                  │ PCIe/NoC
                  ↓
┌─────────────────────────────────────────────────────────────┐
│                  Tensix Core ({0, 0})                       │
├─────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Reader Kernel │    │ Compute Kernel│    │ Writer Kernel│  │
│  │  (RISC 0)    │    │ (Compute)     │    │  (RISC 1)    │  │
│  │              │    │               │    │              │  │
│  │ DRAM → CB    │──→ │  Unpack       │──→ │  CB → DRAM   │  │
│  │    packed    │    │  Ternary      │    │  unpacked    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│       │                    │                    │           │
│       ↓                    ↓                    ↓           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ NoC R0       │    │  L1 (SRAM)   │    │  NoC R1      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                                │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Responsibilities

| Kernel | Purpose | Key Operations |
|--------|---------|----------------|
| `read_packed.cpp` | Read packed data from DRAM | `noc_async_read_page`, `cb_push_back` |
| `ternary_unpack.cpp` | Decode 2-bit trits to float | `extract bits`, `decode to float16` |
| `write_unpacked.cpp` | Write result back to DRAM | `noc_async_write_page`, `cb_pop_front` |

## Known Limitations

This is a **teaching example** that demonstrates TT-Metal API usage, but has the following limitations:

1. **No Actual Unpacking**: The compute kernel (`ternary_unpack.cpp`) is a placeholder that simply copies data. Real implementation would:
   - Extract 2 bits from packed uint8_t
   - Decode to ternary values: 00→-1, 01→0, 10→0, 11→+1
   - Convert to float16/bfloat16 format

2. **No SFPU Vectorization**: Production implementation should use SFPU (Single-Instruction Multiple-Data Floating Point Unit) operations:
   ```cpp
   #include "experimental/llk_sfpu/ckernel_sfpu_custom_ternary_unpack.h"
   // Custom SFPU implementation needed
   ```

3. **Test Will Fail**: The end-to-end test reports mismatches because data is not actually unpacked. This is **expected behavior** for this prototype.

4. **Single Core Only**: Uses only core {0, 0}. Production would distribute work across multiple Tensix cores.

## What This Example Demonstrates

Despite limitations, this example shows:

✓ TT-Metal application structure
✓ Dataflow kernel patterns (reader/writer)
✓ Circular buffer configuration (different data formats)
✓ Kernel compilation with RISC-V compiler (JIT)
✓ Mock cluster simulator usage
✓ TensorAccessorArgs for buffer addressing
✓ Host-side validation infrastructure
✓ Tile-based data flow (32×32 tiles)

## Future Enhancements

For a production-ready BitNet implementation:

### 1. SFPU Unpacking (Required)
```cpp
// In tt_metal/experimental/llk_sfpu/:
// Add: ckernel_sfpu_custom_ternary_unpack.h

inline void unpack_ternary_tile(uint32_t dst) {
    // Load 32-bit words (16 trits each)
    // Extract 2 bits per trit
    // Decode: 00→-1, 01/10→0, 11→+1
    // Scale to float16 range
    // Store to destination tile
}
```

### 2. Multi-Core Scaling
```cpp
auto cores = corerange.x_size() * corerange.y_size();
split_work_to_cores(cores, num_tiles, work_per_core);
// Distribute tile processing across multiple Tensix cores
```

### 3. Fused Operations
- Combine unpacking with accumulation matrix multiplication
- Skip DRAM round-trip for intermediate results

### 4. Sparse Optimization
- Detect zero-run patterns (75% typical in BitNet weights)
- Skip computation for zero tiles

## Troubleshooting

### Build Errors

```bash
# Make sure dependencies are installed
./install_dependencies.sh

# Clean rebuild
rm -rf build
cmake -B build -S . -GNinja
ninja -C build bitnet_unpack_demo
```

### Runtime Errors

```bash
# Check simulator setup
echo $ARCH_NAME  # Should be blackhole or wormhole
echo $TT_METAL_SIMULATOR  # Should be "true" for simulator

# Check for device
lspci | grep Tenstorrent
```

### Performance Issues

This demo focuses on correctness, not performance. For production:
- Tile-sized operations (already implemented)
- Double buffering in circular buffers
- Optimize memory layout for DRAM bandwidth

## References

- [TT-Metalium Guide](../../../../METALIUM_GUIDE.md)
- [Programming Examples](../)
- [LLMs in TT-NN](../../../../tech_reports/LLMs/llms.md) - For context on why ternary encoding matters

## Contributing

This is an educational example. For production contributions, see the main TT-Metal repository.
