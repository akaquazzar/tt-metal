// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BitNet Ternary Unpack Kernel Demo
 *
 * This demo implements a minimal ternary unpacking kernel for BitNet-style quantization.
 * Ternary representation: -1, 0, +1 packed into 2 bits.
 *
 * Packed format:
 *   Each uint8_t packs 4 ternary values (2 bits each: 00=-1, 01=0, 10=0, 11=+1)
 *   Encoding: 00->-1, 01->0, 10->0, 11->+1 (specific to different packing schemes)
 *
 * All-zero optimization:
 *   Since 75% of values in typical weights are 0 (01 or 10), we can test
 *   how the unpacker handles sparse ternary data efficiently.
 *
 * For development/testing:
 *   - Uses simulator (ARCH_NAME=blackhole TT_METAL_SIMULATOR=true)
 *   - Demonstrates full kernel lifecycle: dispatch, execution, validation
 *   - Shows how to format input data for device kernels
 */

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <iomanip>
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

/**
 * Ternary encoding/decoding helpers (for host-side comparison)
 *
 * Packed representation: 2 bits per ternary value
 *   Bits 1:0 -> Trit 0
 *   Bits 3:2 -> Trit 1
 *   Bits 5:4 -> Trit 2
 *   Bits 7:6 -> Trit 3
 *
 * Decoding:
 *   00 -> -1.0
 *   01 ->  0.0
 *   10 ->  0.0
 *   11 -> +1.0
 */

constexpr float decode_trit(uint8_t packed, uint8_t trit_idx) {
    // Extract 2 bits for the specified trit
    constexpr uint8_t TERNARY_MASK = 0x3;  // 0b11
    uint8_t bits = (packed >> (trit_idx * 2)) & TERNARY_MASK;

    // Decode to float value
    switch (bits) {
        case 0x0: return -1.0f;  // 00 -> -1
        case 0x1: return  0.0f;  // 01 -> 0
        case 0x2: return  0.0f;  // 10 -> 0
        case 0x3: return  1.0f;  // 11 -> +1
        default:
            return 0.0f;  // Should never happen
    }
}

/**
 * Host-side reference implementation of ternary unpacking
 * Used to validate the device kernel result
 */
std::vector<float> unpack_ternary_host(const std::vector<uint8_t>& packed, uint32_t num_trits) {
    std::vector<float> unpacked;
    unpacked.reserve(num_trits);

    for (uint32_t i = 0; i < num_trits; i++) {
        uint32_t byte_idx = i / 4;
        uint32_t trit_idx = i % 4;

        float value = decode_trit(packed[byte_idx], trit_idx);
        unpacked.push_back(value);
    }

    return unpacked;
}

/**
 * Generate test data with various ternary value distributions
 */
std::vector<uint8_t> generate_test_data(uint32_t num_trits, const std::string& pattern) {
    std::vector<uint8_t> packed((num_trits + 3) / 4, 0);

    for (uint32_t i = 0; i < num_trits; i++) {
        float value;
        if (pattern == "sparse") {
            // 75% zeros, 12.5% +1, 12.5% -1 (typical BitNet distribution)
            int rand_val = rand() % 8;
            value = (rand_val == 0) ? -1.0f : (rand_val == 1) ? 1.0f : 0.0f;
        } else if (pattern == "balanced") {
            // Equal distribution: 33% each
            int rand_val = rand() % 3;
            value = (rand_val == 0) ? -1.0f : (rand_val == 1) ? 0.0f : 1.0f;
        } else if (pattern == "alternating") {
            // Test pattern: -1, 0, +1, 0, -1, 0, +1, 0, ...
            value = (i % 4 == 0) ? -1.0f : (i % 4 == 1) ? 0.0f : (i % 4 == 2) ? 1.0f : 0.0f;
        } else if (pattern == "all_zeros") {
            value = 0.0f;
        } else if (pattern == "all_ones") {
            value = 1.0f;
        } else {
            value = 0.0f;
        }

        // Pack trit into uint8_t
        uint32_t byte_idx = i / 4;
        uint32_t trit_idx = i % 4;

        uint8_t bits;
        if (value < -0.5f) {
            bits = 0x0;  // 00 -> -1
        } else if (value > 0.5f) {
            bits = 0x3;  // 11 -> +1
        } else {
            bits = 0x1;  // 01 -> 0
        }

        packed[byte_idx] |= (bits << (trit_idx * 2));
    }

    return packed;
}

/**
 * Print ternary data in readable format
 */
void print_ternary_data(const std::vector<float>& data, const std::string& label, uint32_t max_print = 64) {
    std::cout << label << ": [";
    for (uint32_t i = 0; i < std::min(data.size(), static_cast<size_t>(max_print)); i++) {
        if (abs(data[i]) < 0.5f) {
            std::cout << "0";
        } else if (data[i] > 0) {
            std::cout << "+1";
        } else {
            std::cout << "-1";
        }
        if (i < std::min(data.size(), static_cast<size_t>(max_print)) - 1) {
            std::cout << ", ";
        }
    }
    if (data.size() > max_print) {
        std::cout << ", ... (" << data.size() << " total)";
    }
    std::cout << "]" << std::endl;
}

/**
 * Main entry point for BitNet ternary unpack demo
 */
int main() {
    bool pass = true;

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "BitNet Ternary Unpack Kernel Demo" << std::endl;
        std::cout << "========================================" << std::endl << std::endl;

        // Test configuration
        constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        constexpr uint32_t NUM_TILES = 4;  // Process 4 tiles (4 * 1024 = 4096 elements)
        constexpr uint32_t NUM_TRITS = NUM_TILES * TILE_SIZE;  // 4096 ternary values
        const std::string test_pattern = "alternating";  // Options: sparse, balanced, alternating, all_zeros, all_ones

        std::cout << "Test Configuration:" << std::endl;
        std::cout << "  - Number of tiles:    " << NUM_TILES << std::endl;
        std::cout << "  - Trits per tile:     " << TILE_SIZE << std::endl;
        std::cout << "  - Total trits:        " << NUM_TRITS << std::endl;
        std::cout << "  - Test pattern:       " << test_pattern << std::endl;
        std::cout << std::endl;

        // Generate test data
        std::cout << "Generating test data..." << std::endl;
        std::vector<uint8_t> packed_input = generate_test_data(NUM_TRITS, test_pattern);
        std::cout << "  Packed bytes:        " << packed_input.size() << std::endl;

        // Host-side reference result
        std::vector<float> host_reference = unpack_ternary_host(packed_input, NUM_TRITS);

        // Print first few values for verification
        std::vector<float> first_32(host_reference.begin(), std::min(host_reference.begin() + 32, host_reference.end()));
        print_ternary_data(first_32, "  First 32 trits");
        std::cout << std::endl;

        // Create device (will use simulator if enabled)
        std::cout << "Creating device..." << std::endl;
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        // Initialize program
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram(*mesh_device);

        // Core configuration (use single core for simplicity)
        constexpr CoreCoord core = {0, 0};
        auto core_grid = mesh_device->compute_with_storage_grid_size();

        std::cout << "  Grid size:           " << core_grid.x << " x " << core_grid.y << std::endl;
        std::cout << "  Using core:          {" << core.x << ", " << core.y << "}" << std::endl;
        std::cout << std::endl;

        // Constants
        constexpr uint32_t tiles_per_cb = 2;
        constexpr uint32_t trits_per_byte = 4;
        constexpr uint32_t packed_bytes_per_tile = TILE_SIZE / trits_per_byte;  // 256 bytes per tile
        constexpr uint32_t packed_tile_size = packed_bytes_per_tile * sizeof(uint8_t);
        constexpr uint32_t unpacked_tile_size = TILE_SIZE * sizeof(bfloat16);

        // Create buffers
        // Input buffer: packed ternary data (uint8_t)
        const uint32_t packed_size_bytes = NUM_TILES * packed_tile_size;
        distributed::DeviceLocalBufferConfig packed_dram_config{
            .page_size = packed_tile_size,
            .buffer_type = BufferType::DRAM
        };
        distributed::ReplicatedBufferConfig packed_buffer_config{.size = packed_size_bytes};

        auto src_packed_buffer = distributed::MeshBuffer::create(
            packed_buffer_config,
            packed_dram_config,
            mesh_device.get()
        );

        // Output buffer: unpacked float16 data
        const uint32_t unpacked_size_bytes = NUM_TILES * unpacked_tile_size;
        distributed::DeviceLocalBufferConfig unpacked_dram_config{
            .page_size = unpacked_tile_size,
            .buffer_type = BufferType::DRAM
        };
        distributed::ReplicatedBufferConfig unpacked_buffer_config{.size = unpacked_size_bytes};

        auto dst_unpacked_buffer = distributed::MeshBuffer::create(
            unpacked_buffer_config,
            unpacked_dram_config,
            mesh_device.get()
        );

        // Create circular buffers
        std::cout << "Setting up circular buffers..." << std::endl;

        // Input circular buffer (packed ternary data - uint8_t)
        tt::CBIndex src_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                tiles_per_cb * packed_tile_size,
                {{static_cast<uint8_t>(src_cb_index), tt::DataFormat::UInt8}}
            ).set_page_size(src_cb_index, packed_tile_size)
        );

        // Output circular buffer (unpacked float16 data)
        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                tiles_per_cb * unpacked_tile_size,
                {{static_cast<uint8_t>(dst_cb_index), tt::DataFormat::Float16_b}}
            ).set_page_size(dst_cb_index, unpacked_tile_size)
        );

        // Create kernels
        std::cout << "Creating kernels..." << std::endl;

        // Reader kernel: only needs TensorAccessorArgs for buffer addressing
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src_packed_buffer).append_to(reader_compile_time_args);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/read_packed.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args
            }
        );

        // Writer kernel: only needs TensorAccessorArgs for buffer addressing
        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_unpacked_buffer).append_to(writer_compile_time_args);

        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/write_unpacked.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args
            }
        );

        // Compute kernel: no compile args (uses CB indices directly)
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/compute/ternary_unpack.cpp",
            core,
            ComputeConfig{}
        );

        std::cout << "  Reader kernel:      read_packed.cpp" << std::endl;
        std::cout << "  Writer kernel:      write_unpacked.cpp" << std::endl;
        std::cout << "  Compute kernel:     ternary_unpack.cpp" << std::endl;
        std::cout << std::endl;

        // Upload input data
        std::cout << "Uploading input data to device..." << std::endl;
        distributed::EnqueueWriteMeshBuffer(cq, src_packed_buffer, packed_input, false);
        std::cout << std::endl;

        // Set runtime arguments (buffer addresses and tile counts)
        SetRuntimeArgs(program, reader, core, {src_packed_buffer->address(), NUM_TILES});
        SetRuntimeArgs(program, writer, core, {dst_unpacked_buffer->address(), NUM_TILES});
        SetRuntimeArgs(program, compute, core, {NUM_TILES});

        // Execute program
        std::cout << "Executing kernel..." << std::endl;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        std::cout << std::endl;

        // Read back results
        std::cout << "Reading results from device..." << std::endl;
        std::vector<bfloat16> device_result;
        distributed::EnqueueReadMeshBuffer(cq, device_result, dst_unpacked_buffer, true);
        std::cout << std::endl;

        // Validate results
        std::cout << "Validating results against host reference..." << std::endl;
        uint32_t errors = 0;
        for (uint32_t i = 0; i < NUM_TRITS; i++) {
            float expected = host_reference[i];
            float actual = static_cast<float>(device_result[i]);

            // Allow small tolerance for float16 precision
            if (std::abs(expected - actual) > 0.1f) {
                if (errors < 10) {  // Only print first 10 errors
                    std::cout << "  Mismatch at index " << i
                              << ": expected " << expected
                              << ", got " << actual << std::endl;
                }
                errors++;
            }
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        if (errors == 0) {
            std::cout << "✓ Test PASSED" << std::endl;
            pass = true;
        } else {
            std::cout << "✗ Test FAILED" << std::endl;
            std::cout << "  Total errors: " << errors << " / " << NUM_TRITS << std::endl;
            pass = false;
        }
        std::cout << "========================================" << std::endl;

        // Close device
        mesh_device->close();

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception!" << std::endl;
        std::cerr << e.what() << std::endl;
        pass = false;
    }

    return pass ? 0 : 1;
}
