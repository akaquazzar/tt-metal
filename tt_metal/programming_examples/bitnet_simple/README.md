# BitNet Simple - Ternary Unpack Kernel Demo

Minimal example demonstrating ternary quantization unpacking on Tenstorrent hardware.

## What this demo does

This example implements a ternary unpacking operation for BitNet-style quantization:
- Input: Ternary values {-1, 0, +1} packed in 2 bits (4 trits per byte)
- Output: Unpacked float16 values

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

## Future Enhancements

This is a minimal skeleton. Production optimizations would include:

1. **Vectorized Unpacking**: Use SFPU instructions for parallel processing
2. **Fused Operations**: Combine unpacking with downstream operations (accumulate)
3. **Multi-Core Scaling**: Distribute work across multiple Tensix cores
4. **Sparse Optimization**: Skip zero values during unpacking
5. **Custom Data Format**: Support alternative ternary encodings

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
