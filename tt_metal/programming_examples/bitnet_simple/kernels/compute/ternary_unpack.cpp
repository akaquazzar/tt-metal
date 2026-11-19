#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_custom_ternary_unpack.h"
#endif

void kernel_main() {
    const uint32_t M_tiles = get_arg_val<uint32_t>(0);
    const uint32_t K_tiles = get_arg_val<uint32_t>(1); // Кратно 16
    const uint32_t N_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_packed   = tt::CBIndex::c_0;
    constexpr uint32_t cb_act      = tt::CBIndex::c_1;
    constexpr uint32_t cb_unpacked = tt::CBIndex::c_2;
    constexpr uint32_t cb_out      = tt::CBIndex::c_16;

    constexpr uint32_t batch_starts[4] = {0, 4, 8, 12};
    constexpr uint32_t batch_size = 4;
    constexpr uint32_t trits_per_word = 16;
    constexpr uint32_t acc_dst_idx = 5;

    const uint32_t num_k_blocks = K_tiles / trits_per_word;

    compute_kernel_hw_startup(cb_packed, cb_unpacked);

    for (uint32_t m = 0; m < M_tiles; ++m) {
        for (uint32_t n = 0; n < N_tiles; ++n) {
            // 1. Сначала распаковываем ВСЕ блоки весов по глубине K для столбца (m, n) в cb_unpacked
            reconfig_data_format(cb_packed, cb_packed);
            pack_reconfig_data_format(cb_unpacked);
            copy_init(cb_packed);

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
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
            }

            // 2. Теперь ОДИН РАЗ инициализируем Matmul и накапливаем всю глубину K_tiles в DST!
            reconfig_data_format(cb_act, cb_unpacked);
            pack_reconfig_data_format(cb_out);
            matmul_init(cb_act, cb_unpacked);

            tile_regs_acquire();

            for (uint32_t k = 0; k < K_tiles; k += batch_size) {
                cb_wait_front(cb_act, batch_size);
                cb_wait_front(cb_unpacked, batch_size);

                for (uint32_t i = 0; i < batch_size; ++i) {
                    matmul_tiles(cb_act, cb_unpacked, i, i, acc_dst_idx);
                }

                cb_pop_front(cb_act, batch_size);
                cb_pop_front(cb_unpacked, batch_size);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, 1);
            pack_tile(acc_dst_idx, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_release();
        }
    }
}


