#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/distributed.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
constexpr uint32_t TRITS_PER_WORD = 16;
// Полный объём L1 worker-ядра Blackhole; используется как верхняя оценка для CB
constexpr uint32_t L1_SIZE_BYTES = 1536 * 1024;

// Описание одного теста (все размерности — в 32x32 тайлах)
struct BitnetTestConfig {
    std::string name;
    uint32_t m_tiles;
    uint32_t k_tiles;     // Должно быть кратно TRITS_PER_WORD
    uint32_t n_tiles;
    std::string pattern;  // sparse | balanced | alternating | all_zeros | all_ones
    uint32_t seed;
};

struct BitnetTestResult {
    bool pass = false;
    uint32_t mismatches = 0;
    uint32_t num_cores = 0;
    uint32_t max_n_tiles_per_core = 0;
    double elapsed_ms = 0.0;
};

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

bool is_known_pattern(const std::string& pattern) {
    return pattern == "sparse" || pattern == "balanced" || pattern == "alternating" || pattern == "all_zeros" ||
           pattern == "all_ones";
}

std::vector<uint32_t> generate_test_data(uint32_t num_trits, const std::string& pattern, uint32_t seed) {
    std::vector<uint32_t> packed((num_trits + 15) / 16, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> sparse_dist(0, 7);
    std::uniform_int_distribution<uint32_t> balanced_dist(0, 2);

    for (uint32_t i = 0; i < num_trits; i++) {
        int8_t value = 0;
        if (pattern == "sparse") {
            uint32_t rand_val = sparse_dist(rng);
            value = (rand_val == 0) ? -1 : (rand_val == 1) ? 1 : 0;
        } else if (pattern == "balanced") {
            uint32_t rand_val = balanced_dist(rng);
            value = (rand_val == 0) ? -1 : (rand_val == 1) ? 0 : 1;
        } else if (pattern == "alternating") {
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

[[maybe_unused]] void print_ternary_data(
    const std::vector<int8_t>& data, const std::string& label, uint32_t max_print = 64) {
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

void matmul_tilized_int8_accumulate(
    const int8_t* tile_a, const int8_t* tile_b, std::vector<int32_t>& acc_tile, size_t acc_offset) {
    auto get_tilized_elem = [](const int8_t* tile, uint32_t r, uint32_t c) -> int32_t {
        uint32_t face_id = (r / 16) * 2 + (c / 16);
        uint32_t idx = face_id * 256 + (r % 16) * 16 + (c % 16);
        return static_cast<int32_t>(tile[idx]);
    };

    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < 32; ++k) {
                sum += get_tilized_elem(tile_a, r, k) * get_tilized_elem(tile_b, k, c);
            }
            uint32_t out_face_id = (r / 16) * 2 + (c / 16);
            uint32_t out_idx = out_face_id * 256 + (r % 16) * 16 + (c % 16);
            acc_tile[acc_offset + out_idx] += sum;
        }
    }
}

// DEST хранит Int32 в sign-magnitude формате целочисленного FPU Blackhole
int32_t tensix_sm_to_int32(int32_t raw_val) {
    uint32_t u = static_cast<uint32_t>(raw_val);
    int32_t mag = static_cast<int32_t>(u & 0x7FFFFFFFu);
    return (u & 0x80000000u) ? -mag : mag;
}

// ============================================================================
// Один полный прогон: генерация данных, host-reference, запуск ядер на
// устройстве и валидация результата. Вся "боевая" нагрузка живёт здесь.
// ============================================================================
BitnetTestResult run_bitnet_matmul_test(distributed::MeshDevice& mesh_device, const BitnetTestConfig& test) {
    BitnetTestResult result;
    const auto start_time = std::chrono::steady_clock::now();

    const uint32_t M_TILES = test.m_tiles;
    const uint32_t K_TILES = test.k_tiles;
    const uint32_t N_TILES = test.n_tiles;

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Test: " << test.name << " (M=" << M_TILES << ", K=" << K_TILES << ", N=" << N_TILES
              << ", pattern=" << test.pattern << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Валидация конфигурации ---
    if (M_TILES == 0 || K_TILES == 0 || N_TILES == 0) {
        std::cerr << "FAIL: M, K, N must be non-zero" << std::endl;
        return result;
    }
    if (K_TILES % TRITS_PER_WORD != 0) {
        std::cerr << "FAIL: K_tiles must be a multiple of " << TRITS_PER_WORD << std::endl;
        return result;
    }
    if (!is_known_pattern(test.pattern)) {
        std::cerr << "FAIL: unknown test pattern '" << test.pattern << "'" << std::endl;
        return result;
    }

    const uint32_t NUM_K_BLOCKS = K_TILES / TRITS_PER_WORD;
    const uint32_t NUM_ACT_TILES = M_TILES * K_TILES;
    const uint32_t NUM_PACKED_WEIGHT_TILES = N_TILES * NUM_K_BLOCKS;
    const uint32_t NUM_OUT_TILES = M_TILES * N_TILES;
    const uint32_t TOTAL_TRITS = NUM_PACKED_WEIGHT_TILES * TILE_SIZE * TRITS_PER_WORD;

    std::cout << "Matrix Dimensions (in 32x32 tiles):" << std::endl;
    std::cout << "  - M_tiles (rows):              " << M_TILES << " (" << M_TILES * 32 << ")" << std::endl;
    std::cout << "  - K_tiles (inner depth):       " << K_TILES << " (" << K_TILES * 32 << ")" << std::endl;
    std::cout << "  - N_tiles (output cols):       " << N_TILES << " (" << N_TILES * 32 << ")" << std::endl;
    std::cout << "  - Input Activations (Int8):    " << NUM_ACT_TILES << " tiles" << std::endl;
    std::cout << "  - Packed Weights (UInt32):     " << NUM_PACKED_WEIGHT_TILES << " tiles" << std::endl;
    std::cout << "  - Output Matrix C (Int32):     " << NUM_OUT_TILES << " tiles" << std::endl << std::endl;

    // --- Генерация данных и эталонный расчёт на хосте ---
    std::cout << "Generating test data..." << std::endl;
    // Сжатые веса хранятся в порядке [n * NUM_K_BLOCKS + kb]
    std::vector<uint32_t> packed_weights = generate_test_data(TOTAL_TRITS, test.pattern, test.seed);

    // Активации хранятся построчно [m * K_TILES + k]
    // Заполним разными значениями по строкам/тайлам для строгой проверки индексации
    std::vector<int8_t> activations(NUM_ACT_TILES * TILE_SIZE, 0);
    for (uint32_t m = 0; m < M_TILES; ++m) {
        for (uint32_t k = 0; k < K_TILES; ++k) {
            int8_t val = static_cast<int8_t>(((m + k) % 3) + 1);  // значения 1, 2, 3
            std::fill_n(activations.data() + (m * K_TILES + k) * TILE_SIZE, TILE_SIZE, val);
        }
    }

    std::vector<int8_t> unpacked_weights_ref = unpack_ternary_host(packed_weights, NUM_PACKED_WEIGHT_TILES);
    std::vector<int32_t> host_matmul_ref(NUM_OUT_TILES * TILE_SIZE, 0);

    for (uint32_t m = 0; m < M_TILES; ++m) {
        for (uint32_t n = 0; n < N_TILES; ++n) {
            const size_t out_tile_offset = (m * N_TILES + n) * TILE_SIZE;
            for (uint32_t k = 0; k < K_TILES; ++k) {
                const size_t act_tile_offset = (m * K_TILES + k) * TILE_SIZE;
                // Веса для столбца n и индекса k находятся в блоке (n * NUM_K_BLOCKS), смещение k
                const size_t weight_tile_offset = (n * K_TILES + k) * TILE_SIZE;

                matmul_tilized_int8_accumulate(
                    activations.data() + act_tile_offset,
                    unpacked_weights_ref.data() + weight_tile_offset,
                    host_matmul_ref,
                    out_tile_offset);
            }
        }
    }

    // --- Распределение работы ---
    constexpr uint32_t packed_tile_size = TILE_SIZE * sizeof(uint32_t);  // 4096 байт
    constexpr uint32_t int8_tile_size = TILE_SIZE * sizeof(int8_t);      // 1024 байта
    constexpr uint32_t int32_tile_size = TILE_SIZE * sizeof(int32_t);    // 4096 байт

    auto compute_grid = mesh_device.compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, n_tiles_group_1, n_tiles_group_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, N_TILES);

    // Максимальное число столбцов на одно ядро (определяет размер буфера c_2 в L1!)
    const uint32_t max_n_tiles_per_core = std::max(n_tiles_group_1, n_tiles_group_2);

    result.num_cores = num_cores;
    result.max_n_tiles_per_core = max_n_tiles_per_core;

    const uint64_t c2_bytes = static_cast<uint64_t>(max_n_tiles_per_core) * K_TILES * int8_tile_size;
    const uint64_t cb_total_bytes =
        2ull * packed_tile_size + 8ull * int8_tile_size + c2_bytes + 2ull * int32_tile_size;

    std::cout << "Multi-core Work Split:" << std::endl;
    std::cout << "  - Compute grid:         " << compute_grid.x << " x " << compute_grid.y << std::endl;
    std::cout << "  - Active cores:         " << num_cores << std::endl;
    std::cout << "  - Max N_tiles per core: " << max_n_tiles_per_core << std::endl;
    std::cout << "  - c_2 per-core size:    " << c2_bytes << " bytes" << std::endl;
    std::cout << "  - Est. L1 CB usage:     " << cb_total_bytes << " / " << L1_SIZE_BYTES << " bytes"
              << std::endl << std::endl;

    if (cb_total_bytes > L1_SIZE_BYTES) {
        std::cerr << "FAIL: estimated L1 usage " << cb_total_bytes << " B exceeds the " << L1_SIZE_BYTES
                  << " B Blackhole L1; increase the core count or reduce N_tiles" << std::endl;
        return result;
    }

    // --- Device-программа ---
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device.shape());
    Program program = CreateProgram(mesh_device);

    auto src_packed_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = NUM_PACKED_WEIGHT_TILES * packed_tile_size},
        distributed::DeviceLocalBufferConfig{.page_size = packed_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    auto src_act_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = NUM_ACT_TILES * int8_tile_size},
        distributed::DeviceLocalBufferConfig{.page_size = int8_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    auto dst_out_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = NUM_OUT_TILES * int32_tile_size},
        distributed::DeviceLocalBufferConfig{.page_size = int32_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    // Circular Buffers выделяются на all_cores, а размер c_2 зависит только от max_n_tiles_per_core!
    std::cout << "Setting up circular buffers..." << std::endl;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * packed_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}})
            .set_page_size(tt::CBIndex::c_0, packed_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(8 * int8_tile_size, {{tt::CBIndex::c_1, tt::DataFormat::Int8}})
            .set_page_size(tt::CBIndex::c_1, int8_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(c2_bytes, {{tt::CBIndex::c_2, tt::DataFormat::Int8}})
            .set_page_size(tt::CBIndex::c_2, int8_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * int32_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Int32}})
            .set_page_size(tt::CBIndex::c_16, int32_tile_size));

    std::cout << "Creating kernels..." << std::endl;
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src_packed_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src_act_buffer).append_to(reader_compile_time_args);

    auto reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/read_packed.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_out_buffer).append_to(writer_compile_time_args);

    auto writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/write_unpacked.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    auto compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/compute/ternary_unpack.cpp",
        all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});

    // Раздаём индивидуальные RuntimeArgs каждому активному ядру
    auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
    uint32_t current_n_offset = 0;

    for (const auto& core : cores) {
        uint32_t core_n_tiles = 0;
        if (core_group_1.contains(core)) {
            core_n_tiles = n_tiles_group_1;
        } else if (core_group_2.contains(core)) {
            core_n_tiles = n_tiles_group_2;
        } else {
            TT_THROW("Core not in any work group!");
        }

        SetRuntimeArgs(
            program,
            reader,
            core,
            {src_packed_buffer->address(),
             src_act_buffer->address(),
             M_TILES,
             K_TILES,
             core_n_tiles,
             current_n_offset});
        SetRuntimeArgs(program, compute, core, {M_TILES, K_TILES, core_n_tiles});
        SetRuntimeArgs(
            program, writer, core, {dst_out_buffer->address(), M_TILES, core_n_tiles, current_n_offset, N_TILES});

        current_n_offset += core_n_tiles;
    }

    std::cout << "Uploading input data to device..." << std::endl;
    distributed::EnqueueWriteMeshBuffer(cq, src_packed_buffer, packed_weights, false);
    distributed::EnqueueWriteMeshBuffer(cq, src_act_buffer, activations, false);

    std::cout << "Executing kernel..." << std::endl;
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::cout << "Reading results from device..." << std::endl;
    std::vector<int32_t> device_result;
    distributed::EnqueueReadMeshBuffer(cq, device_result, dst_out_buffer, true);

    // --- Валидация ---
    std::cout << "Validating device results against host reference..." << std::endl;
    if (device_result.size() != host_matmul_ref.size()) {
        std::cerr << "FAIL: Size mismatch! Expected " << host_matmul_ref.size() << ", got " << device_result.size()
                  << std::endl;
        result.mismatches = static_cast<uint32_t>(host_matmul_ref.size());
    } else {
        uint32_t errors_printed = 0;
        for (size_t i = 0; i < host_matmul_ref.size(); ++i) {
            int32_t actual = tensix_sm_to_int32(device_result[i]);
            if (actual != host_matmul_ref[i]) {
                ++result.mismatches;
                if (errors_printed < 33) {
                    std::cerr << "  Mismatch at index " << i << " (Elem " << i << "): "
                              << "expected " << host_matmul_ref[i] << ", got " << actual << std::endl;
                    ++errors_printed;
                }
            }
        }
    }

    result.pass = (result.mismatches == 0);
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();

    if (result.pass) {
        std::cout << "PASS! (" << M_TILES << ", " << K_TILES << ", " << N_TILES << ") verified across all "
                  << NUM_OUT_TILES << " output tiles on " << num_cores << " cores (" << result.elapsed_ms << " ms)"
                  << std::endl;
    } else {
        std::cout << "FAIL! " << result.mismatches << " mismatched elements (" << result.elapsed_ms << " ms)"
                  << std::endl;
    }
    std::cout << std::endl;

    return result;
}

// ============================================================================
// Набор тестов. Добавляйте новые конфигурации сюда.
// ============================================================================
std::vector<BitnetTestConfig> make_default_test_suite() {
    return {
        // Минимальный размер: один выходной тайл
        {"minimal_1x16x1", 1, 16, 1, "alternating", 1},

        // Небольшая матрица: быстрый smoke-тест
        {"small_2x32x3", 2, 32, 3, "balanced", 2},

        // Целевой сценарий: 26 столбцов N раскладываются по 26 ядрам
        {"multicore_14x64x26", 14, 64, 26, "balanced", 3},

        // Неравномерное деление: столбцов N больше, чем ядер (core_group_2)
        {"uneven_split_1x64x200", 1, 64, 200, "sparse", 4},

        // Проверка всех паттернов весов на малой матрице
        {"pattern_all_zeros", 2, 32, 3, "all_zeros", 5},
        {"pattern_all_ones", 2, 32, 3, "all_ones", 6},
        {"pattern_alternating", 2, 32, 3, "alternating", 7},
        {"pattern_sparse", 2, 32, 3, "sparse", 8},
    };
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> requested(argv + 1, argv + argc);
    const std::vector<BitnetTestConfig> suite = make_default_test_suite();

    std::vector<BitnetTestConfig> tests;
    if (requested.empty()) {
        tests = suite;
    } else {
        for (const auto& name : requested) {
            auto it = std::find_if(
                suite.begin(), suite.end(), [&](const BitnetTestConfig& t) { return t.name == name; });
            if (it == suite.end()) {
                std::cerr << "Unknown test: " << name << std::endl << "Available tests:" << std::endl;
                for (const auto& t : suite) {
                    std::cerr << "  " << t.name << std::endl;
                }
                return 2;
            }
            tests.push_back(*it);
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "BitNet Arbitrary (M, K, N) Fused Matmul Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Tests to run: " << tests.size() << std::endl << std::endl;

    uint32_t passed = 0;
    uint32_t failed = 0;

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        for (const auto& test : tests) {
            BitnetTestResult result;
            try {
                result = run_bitnet_matmul_test(*mesh_device, test);
            } catch (const std::exception& e) {
                std::cerr << "Exception in test '" << test.name << "': " << e.what() << std::endl;
                result.pass = false;
            }
            if (result.pass) {
                ++passed;
            } else {
                ++failed;
            }
        }

        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Fatal exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "TEST SUMMARY: " << passed << "/" << (passed + failed) << " passed";
    if (failed != 0) {
        std::cout << ", " << failed << " FAILED";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return failed == 0 ? 0 : 1;
}
