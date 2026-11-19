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

// Вспомогательная функция на хосте: перемножение двух тайлов 32x32 (A и B),
// хранящихся в формате фейсов Tenstorrent (4 фейса 16x16), с накоплением в acc (Int32)
static void matmul_tilized_int8_accumulate(
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

int main() {
    bool pass = true;

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "BitNet Arbitrary (M, K, N) Fused Matmul" << std::endl;
        std::cout << "========================================" << std::endl << std::endl;

        // Произвольные размеры матриц в тайлах 32x32:
        // C[M_tiles x N_tiles] = A[M_tiles x K_tiles] * B[K_tiles x N_tiles]
        constexpr uint32_t TILE_SIZE = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;  // 1024
        constexpr uint32_t TRITS_PER_WORD = 16;

        constexpr uint32_t M_TILES = 2;   // Например, 64 токена (Prefill); для Decode ставим 1
        constexpr uint32_t K_TILES = 32;  // Должно быть кратно 16 (например, K = 1024)
        constexpr uint32_t N_TILES = 3;   // Например, ширина выхода N = 96

        static_assert(K_TILES % TRITS_PER_WORD == 0, "K_TILES must be a multiple of 16");

        constexpr uint32_t NUM_K_BLOCKS = K_TILES / TRITS_PER_WORD;           // Сжатых блоков по глубине K (2)
        constexpr uint32_t NUM_ACT_TILES = M_TILES * K_TILES;                 // 64 тайла активаций (Int8)
        constexpr uint32_t NUM_PACKED_WEIGHT_TILES = N_TILES * NUM_K_BLOCKS;  // 6 сжатых тайлов весов (UInt32)
        constexpr uint32_t NUM_OUT_TILES = M_TILES * N_TILES;                 // 6 выходных тайлов (Int32)
        constexpr uint32_t TOTAL_TRITS = NUM_PACKED_WEIGHT_TILES * TILE_SIZE * TRITS_PER_WORD;

        const std::string test_pattern = "balanced";

        std::cout << "Matrix Dimensions (in 32x32 tiles):" << std::endl;
        std::cout << "  - M_tiles (rows):              " << M_TILES << " (" << M_TILES * 32 << ")" << std::endl;
        std::cout << "  - K_tiles (inner depth):       " << K_TILES << " (" << K_TILES * 32 << ")" << std::endl;
        std::cout << "  - N_tiles (output cols):       " << N_TILES << " (" << N_TILES * 32 << ")" << std::endl;
        std::cout << "  - Input Activations (Int8):    " << NUM_ACT_TILES << " tiles" << std::endl;
        std::cout << "  - Packed Weights (UInt32):     " << NUM_PACKED_WEIGHT_TILES << " tiles" << std::endl;
        std::cout << "  - Output Matrix C (Int32):     " << NUM_OUT_TILES << " tiles" << std::endl << std::endl;

        // 1. Генерация входных данных
        std::cout << "Generating test data..." << std::endl;
        // Сжатые веса хранятся в порядке [n * NUM_K_BLOCKS + kb]
        std::vector<uint32_t> packed_weights = generate_test_data(TOTAL_TRITS, test_pattern);

        // Активации хранятся построчно [m * K_TILES + k]
        // Заполним разными значениями по строкам/тайлам для строгой проверки индексации
        std::vector<int8_t> activations(NUM_ACT_TILES * TILE_SIZE, 0);
        for (uint32_t m = 0; m < M_TILES; ++m) {
            for (uint32_t k = 0; k < K_TILES; ++k) {
                int8_t val = static_cast<int8_t>(((m + k) % 3) + 1);  // значения 1, 2, 3
                std::fill_n(activations.data() + (m * K_TILES + k) * TILE_SIZE, TILE_SIZE, val);
            }
        }

        // 2. Эталонный расчёт на хосте
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

        // 3. Инициализация устройства
        std::cout << "Creating device..." << std::endl;
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram(*mesh_device);

        constexpr CoreCoord core = {0, 0};

        // 4. Выделение DRAM буферов строго под вычисленные размеры
        constexpr uint32_t packed_tile_size = TILE_SIZE * sizeof(uint32_t);  // 4096 байт
        constexpr uint32_t int8_tile_size = TILE_SIZE * sizeof(int8_t);      // 1024 байта
        constexpr uint32_t int32_tile_size = TILE_SIZE * sizeof(int32_t);    // 4096 байт

        auto src_packed_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = NUM_PACKED_WEIGHT_TILES * packed_tile_size},
            distributed::DeviceLocalBufferConfig{.page_size = packed_tile_size, .buffer_type = BufferType::DRAM},
            mesh_device.get());

        auto src_act_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = NUM_ACT_TILES * int8_tile_size},
            distributed::DeviceLocalBufferConfig{.page_size = int8_tile_size, .buffer_type = BufferType::DRAM},
            mesh_device.get());

        auto dst_out_buffer = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = NUM_OUT_TILES * int32_tile_size},
            distributed::DeviceLocalBufferConfig{.page_size = int32_tile_size, .buffer_type = BufferType::DRAM},
            mesh_device.get());

        // 5. Circular Buffers в L1 SRAM
        std::cout << "Setting up circular buffers..." << std::endl;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(2 * packed_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::UInt32}})
                .set_page_size(tt::CBIndex::c_0, packed_tile_size));

        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(8 * int8_tile_size, {{tt::CBIndex::c_1, tt::DataFormat::Int8}})
                .set_page_size(tt::CBIndex::c_1, int8_tile_size));

        // 16 тайлов в c_2 для хранения одного распакованного блока K=16
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(16 * int8_tile_size, {{tt::CBIndex::c_2, tt::DataFormat::Int8}})
                .set_page_size(tt::CBIndex::c_2, int8_tile_size));

        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(2 * int32_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Int32}})
                .set_page_size(tt::CBIndex::c_16, int32_tile_size));

        // 6. Создание ядер
        std::cout << "Creating kernels..." << std::endl;
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*src_packed_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*src_act_buffer).append_to(reader_compile_time_args);

        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bitnet_simple/kernels/dataflow/read_packed.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_out_buffer).append_to(writer_compile_time_args);

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

        // 7. Передача данных и runtime-аргументов (M_TILES, K_TILES, N_TILES)
        std::cout << "Uploading input data to device..." << std::endl;
        distributed::EnqueueWriteMeshBuffer(cq, src_packed_buffer, packed_weights, false);
        distributed::EnqueueWriteMeshBuffer(cq, src_act_buffer, activations, false);

        SetRuntimeArgs(
            program,
            reader,
            core,
            {src_packed_buffer->address(), src_act_buffer->address(), M_TILES, K_TILES, N_TILES});
        SetRuntimeArgs(program, compute, core, {M_TILES, K_TILES, N_TILES});
        SetRuntimeArgs(program, writer, core, {dst_out_buffer->address(), NUM_OUT_TILES});
        // 8. Запуск на симуляторе
        std::cout << "Executing kernel..." << std::endl;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        // 9. Чтение и проверка всех (M_TILES * N_TILES) выходных тайлов
        std::cout << "Reading results from device..." << std::endl;
        std::vector<int32_t> device_result;
        distributed::EnqueueReadMeshBuffer(cq, device_result, dst_out_buffer, true);
        std::cout << "Validating device results against host reference..." << std::endl;
        if (device_result.size() != host_matmul_ref.size()) {
            std::cerr << "FAIL: Size mismatch! Expected " << host_matmul_ref.size() << ", got " << device_result.size()
                      << std::endl;
            pass = false;
        } else if (device_result != host_matmul_ref) {
            std::cerr << "FAIL: Data mismatch!" << std::endl;
            pass = false;
            uint32_t errors_printed = 0;
            for (size_t i = 0; i < host_matmul_ref.size(); ++i) {
                if (device_result[i] != host_matmul_ref[i]) {
                    std::cerr << "  Mismatch at index " << i << " (Out Tile " << (i / TILE_SIZE) << ", Elem "
                              << (i % TILE_SIZE) << "): expected " << host_matmul_ref[i] << ", got " << device_result[i]
                              << std::endl;
                    if (++errors_printed >= 10) {
                        break;
                    }
                }
            }
        }
        if (pass) {
            std::cout << "PASS! Arbitrary (M=" << M_TILES << ", K=" << K_TILES << ", N=" << N_TILES
                      << ") BitNet Matmul verified across all " << NUM_OUT_TILES << " output tiles!" << std::endl;
        }
        mesh_device->close();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        pass = false;
    }
    return pass ? 0 : 1;
}