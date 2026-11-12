#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "experimental/llk_sfpu/ckernel_sfpu_custom_ternary_unpack.h"
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    constexpr uint32_t batch_starts[3] = {0, 7, 14};
    constexpr uint32_t batch_counts[3] = {7, 7, 2};

    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        cb_wait_front(cb_in, 1);

        // Проходим по всем 16 тритам за 3 итерации, не выходя за лимит 8 тайлов DST
        for (uint32_t b = 0; b < 3; ++b) {
            const uint32_t start_k = batch_starts[b];
            const uint32_t count = batch_counts[b];

            tile_regs_acquire();
            
            // Копируем входной сжатый тайл в слот DST[0] (Read-Only)
            copy_tile(cb_in, 0, 0);

            MATH(
                for (uint32_t row = 0; row < 32; row++) {
                    ckernel::sfpu::unpack_row_batch(row, start_k, count);
                }
            );

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, count);
            for (uint32_t i = 0; i < count; ++i) {
                pack_tile(i + 1, cb_out);
            }
            cb_push_back(cb_out, count);

            tile_regs_release();
        }

        cb_pop_front(cb_in, 1);
    }
}
