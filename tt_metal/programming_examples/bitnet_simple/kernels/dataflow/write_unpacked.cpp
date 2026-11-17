#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t M_tiles  = get_arg_val<uint32_t>(1);
    const uint32_t N_tiles  = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t out_tile_bytes = 4096; // Int32 (1024 * 4)

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_gen = TensorAccessor(out_args, out_addr, out_tile_bytes);

    for (uint32_t n = 0; n < N_tiles; ++n) {
        for (uint32_t m = 0; m < M_tiles; ++m) {
            cb_wait_front(cb_out, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_out);

            uint32_t out_tile_idx = m * N_tiles + n;
            noc_async_write(l1_read_addr, out_gen.get_noc_addr(out_tile_idx), out_tile_bytes);
            noc_async_write_barrier();

            cb_pop_front(cb_out, 1);
        }
    }
}
