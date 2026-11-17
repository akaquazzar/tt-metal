#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_custom_ternary_unpack.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_packed   = tt::CBIndex::c_0;  // UInt32 (4096 байт)
    constexpr uint32_t cb_act      = tt::CBIndex::c_1;  // Int8   (1024 байта)
    constexpr uint32_t cb_unpacked = tt::CBIndex::c_2;  // Int8   (1024 байта, L1-only, размер 16 тайлов)
    constexpr uint32_t cb_out      = tt::CBIndex::c_16; // Int32  (4096 байт)

    constexpr uint32_t batch_starts[4] = {0, 4, 8, 12};
    constexpr uint32_t batch_size = 4;
    constexpr uint32_t total_k_tiles = 16;

    // =========================================================================
    // ЭТАП 1: Распаковка всех 16 тайлов весов в L1 (cb_unpacked)
    // =========================================================================
    compute_kernel_hw_startup(cb_packed, cb_unpacked);
    //compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    copy_init(cb_packed);

    cb_wait_front(cb_packed, 1);

    for (uint32_t b = 0; b < 4; ++b) {
        const uint32_t start_k = batch_starts[b];

        tile_regs_acquire();
        copy_tile(cb_packed, 0, 0);

        MATH(
            for (uint32_t row = 0; row < 32; ++row) {
                ckernel::sfpu::unpack_row_batch(row, start_k, batch_size);
            }
        );

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_unpacked, batch_size);
        for (uint32_t i = 0; i < batch_size; ++i) {
            pack_tile(i + 1, cb_unpacked);
        }
        cb_push_back(cb_unpacked, batch_size);

        tile_regs_release();
    }

    cb_pop_front(cb_packed, 1);

    // =========================================================================
    // ЭТАП 2: Матричное умножение 16 тайлов активаций на 16 распакованных весов
    // =========================================================================
    reconfig_data_format(cb_act, cb_unpacked);
    pack_reconfig_data_format(cb_out);
    matmul_init(cb_act, cb_unpacked);

    tile_regs_acquire();

    for (uint32_t k = 0; k < total_k_tiles; k += batch_size) {
        cb_wait_front(cb_act, batch_size);
        cb_wait_front(cb_unpacked, batch_size);

        for (uint32_t i = 0; i < batch_size; ++i) {
            matmul_tiles(cb_act, cb_unpacked, i, i, 5);
        }

        cb_pop_front(cb_act, batch_size);
        cb_pop_front(cb_unpacked, batch_size);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, 1);
    pack_tile(5, cb_out);
    cb_push_back(cb_out, 1);

    tile_regs_release();
    DEVICE_PRINT("---kernel_main\n");
}
