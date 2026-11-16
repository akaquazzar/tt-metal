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
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

constexpr int8_t decode_trit_u32(uint32_t packed_word, uint32_t trit_idx) {
    constexpr uint32_t TERNARY_MASK = 0x3;
    uint32_t bits = (packed_word >> (trit_idx * 2)) & TERNARY_MASK;

    switch (bits) {
        case 0x0: return 0;   // 00 -> 0
        case 0x1: return 1;   // 01 -> +1
        case 0x2: return -1;  // 10 -> -1
        default: return 0;
    }
}

std::vector<int8_t> unpack_ternary_host(const std::vector<uint32_t>& packed, uint32_t num_in_tiles) {
    constexpr uint32_t TILE_SIZE = 1024;
    constexpr uint32_t TRITS_PER_WORD = 16;

    const uint32_t total_trits = num_in_tiles * TILE_SIZE * TRITS_PER_WORD;
    std::vector<int8_t> unpacked(total_trits, 0);

    for (uint32_t in_tile = 0; in_tile < num_in_tiles; ++in_tile) {
        for (uint32_t elem_idx = 0; elem_idx < TILE_SIZE; ++elem_idx) {
            uint32_t word_idx = in_tile * TILE_SIZE + elem_idx;
            uint32_t packed_word = packed[word_idx];

            for (uint32_t k = 0; k < TRITS_PER_WORD; ++k) {
                int8_t val = decode_trit_u32(packed_word, k);

                uint32_t out_tile = in_tile * TRITS_PER_WORD + k;
                uint32_t out_idx = out_tile * TILE_SIZE + elem_idx;

                unpacked[out_idx] = val;
            }
        }
    }

    return unpacked;
}

std::vector<uint32_t> generate_test_data(uint32_t num_trits, const std::string& pattern) {
    std::vector<uint32_t> packed((num_trits + 15) / 16, 0);

    for (uint32_t i = 0; i < num_trits; i++) {
        int8_t value = 0;
        if (pattern == "sparse") {
            int rand_val = rand() % 8;
            value = (rand_val == 0) ? -1 : (rand_val == 1) ? 1 : 0;
        } else if (pattern == "balanced") {
            int rand_val = rand() % 3;
            value = (rand_val == 0) ? -1 : (rand_val == 1) ? 0 : 1;
        } else if (pattern == "alternating") {
            // Паттерн: -1, 0, +1, 0, -1, 0, +1, 0...
            value = (i % 4 == 0) ? -1 : (i % 4 == 1) ? 0 : (i % 4 == 2) ? 1 : 0;
        } else if (pattern == "all_zeros") {
            value = 0;
        } else if (pattern == "all_ones") {
            value = 1;
        }

        uint32_t word_idx = i / 16;
        uint32_t trit_idx = i % 16;

        uint32_t bits = 0x0;
        if (value > 0) {
            bits = 0x1;  // 01 -> +1
        } else if (value < 0) {
            bits = 0x2;  // 10 -> -1
        }

        packed[word_idx] |= (bits << (trit_idx * 2));
    }

    return packed;
}


void print_ternary_data(const std::vector<int8_t>& data, const std::string& label, uint32_t max_print = 64) {
    std::cout << label << ": [";
    size_t limit = std::min(data.size(), static_cast<size_t>(max_print));
    for (size_t i = 0; i < limit; i++) {
        if (data[i] == 0) {
            std::cout << "0";
        } else if (data[i] > 0) {
            std::cout << "+1";
        } else {
            std::cout << "-1";
        }
        if (i < limit - 1) {
            std::cout << ", ";
        }
    }
    if (data.size() > max_print) {
        std::cout << ", ... (" << data.size() << " total)";
    }
    std::cout << "]" << std::endl;
}

int main() {
    bool pass = true;

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "BitNet Ternary Unpack Kernel Demo" << std::endl;
        std::cout << "========================================" << std::endl << std::endl;

        constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
        constexpr uint32_t TRITS_PER_WORD = 16;
        constexpr uint32_t NUM_IN_TILES = 16;
        constexpr uint32_t NUM_OUT_TILES = NUM_IN_TILES * TRITS_PER_WORD;          // 16
        constexpr uint32_t NUM_TRITS = NUM_IN_TILES * TILE_SIZE * TRITS_PER_WORD;
        const std::string test_pattern = "balanced";

        std::cout << "Test Configuration:" << std::endl;
        std::cout << "  - Input tiles (UInt32):  " << NUM_IN_TILES << std::endl;
        std::cout << "  - Output tiles (Int8):   " << NUM_OUT_TILES << std::endl;
        std::cout << "  - Total trits:           " << NUM_TRITS << std::endl;
        std::cout << "  - Test pattern:          " << test_pattern << std::endl << std::endl;

        std::cout << "Generating test data..." << std::endl;
        std::vector<uint32_t> packed_input = generate_test_data(NUM_TRITS, test_pattern);
        std::cout << "  Packed words (uint32_t): " << packed_input.size() << std::endl;
        std::cout << "  Packed bytes:            " << packed_input.size() * sizeof(uint32_t) << std::endl;

        std::vector<int8_t> host_reference = unpack_ternary_host(packed_input, NUM_IN_TILES);

        std::vector<int8_t> first_32(
            host_reference.begin(), std::min(host_reference.begin() + 32, host_reference.end()));
        print_ternary_data(first_32, "  First 32 trits (Host Tile 0)");
        std::cout << std::endl;

        std::cout << "Creating device..." << std::endl;
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram(*mesh_device);

        constexpr CoreCoord core = {0, 0};
        auto core_grid = mesh_device->compute_with_storage_grid_size();
        std::cout << "  Grid size:           " << core_grid.x << " x " << core_grid.y << std::endl;
        std::cout << "  Using core:          {" << core.x << ", " << core.y << "}" << std::endl << std::endl;

        constexpr uint32_t in_tiles_per_cb = 2;
        constexpr uint32_t out_tiles_per_cb = 16;  // Вмещает все 16 выходных тайлов без блокировки
        constexpr uint32_t packed_tile_size = TILE_SIZE * sizeof(uint32_t);  // 4096 байт
        constexpr uint32_t unpacked_tile_size = TILE_SIZE * sizeof(int8_t);  // 1024 байта

        const uint32_t packed_size_bytes = NUM_IN_TILES * packed_tile_size;
        distributed::DeviceLocalBufferConfig packed_dram_config{
            .page_size = packed_tile_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig packed_buffer_config{.size = packed_size_bytes};
        auto src_packed_buffer =
            distributed::MeshBuffer::create(packed_buffer_config, packed_dram_config, mesh_device.get());

        const uint32_t unpacked_size_bytes = NUM_OUT_TILES * unpacked_tile_size;
        distributed::DeviceLocalBufferConfig unpacked_dram_config{
            .page_size = unpacked_tile_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig unpacked_buffer_config{.size = unpacked_size_bytes};
        auto dst_unpacked_buffer =
            distributed::MeshBuffer::create(unpacked_buffer_config, unpacked_dram_config, mesh_device.get());

        // Circular Buffers в L1 SRAM
        std::cout << "Setting up circular buffers..." << std::endl;
        tt::CBIndex src_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                in_tiles_per_cb * packed_tile_size, {{static_cast<uint8_t>(src_cb_index), tt::DataFormat::UInt32}})
                .set_page_size(src_cb_index, packed_tile_size));

        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                out_tiles_per_cb * unpacked_tile_size, {{static_cast<uint8_t>(dst_cb_index), tt::DataFormat::Int8}})
                .set_page_size(dst_cb_index, unpacked_tile_size));

        std::cout << "Creating kernels..." << std::endl;
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src_packed_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/read_packed.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_unpacked_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/write_unpacked.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/compute/ternary_unpack.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});

        std::cout << "Uploading input data to device..." << std::endl;
        distributed::EnqueueWriteMeshBuffer(cq, src_packed_buffer, packed_input, false);
        SetRuntimeArgs(program, reader, core, {src_packed_buffer->address(), NUM_IN_TILES});
        SetRuntimeArgs(program, writer, core, {dst_unpacked_buffer->address(), NUM_OUT_TILES});
        SetRuntimeArgs(program, compute, core, {NUM_IN_TILES});
        // Запуск
        std::cout << "Executing kernel..." << std::endl;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::cout << "Reading results from device..." << std::endl;
        std::vector<int8_t> device_result;
        distributed::EnqueueReadMeshBuffer(cq, device_result, dst_unpacked_buffer, true);
        std::vector<int8_t> dev_first_32(
            device_result.begin(), std::min(device_result.begin() + 32, device_result.end()));
        print_ternary_data(dev_first_32, "  First 32 trits (Device Tile 0)");
        std::cout << std::endl;
        std::cout << "Validating device results against host reference..." << std::endl;
        if (device_result.size() != host_reference.size()) {
            std::cerr << "FAIL: Size mismatch! Expected " << host_reference.size() << ", got " << device_result.size()
                      << std::endl;
            pass = false;
        } else if (device_result != host_reference) {
            std::cerr << "FAIL: Data mismatch!" << std::endl;
            pass = false;
            uint32_t errors_printed = 0;
            for (size_t i = 0; i < host_reference.size(); ++i) {
                if (device_result[i] != host_reference[i]) {
                    std::cerr << "  Mismatch at index " << i << " (Out Tile " << (i / TILE_SIZE) << ", Elem "
                              << (i % TILE_SIZE) << "): "
                              << "expected " << static_cast<int>(host_reference[i]) << ", got "
                              << static_cast<int>(device_result[i]) << std::endl;
                    if (++errors_printed >= 10) {
                        break;
                    }
                }
            }
        } else {
            std::cout << "PASS: All " << device_result.size() << " unpacked int8 weights match bit-for-bit!"
                      << std::endl;
        }
        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        pass = false;
    }
    return pass ? 0 : 1;
}