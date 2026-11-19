#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr        = get_arg_val<uint32_t>(0);
    const uint32_t total_out_tiles = get_arg_val<uint32_t>(1); // M_tiles * N_tiles

    DEVICE_PRINT("+++kernel_main total_out_tiles: {}\n", total_out_tiles);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t out_tile_bytes = 4096;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_gen = TensorAccessor(out_args, out_addr, out_tile_bytes);

    for (uint32_t out_t = 0; out_t < total_out_tiles; ++out_t) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write(l1_read_addr, out_gen.get_noc_addr(out_t), out_tile_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
    DEVICE_PRINT("---kernel_main\n");
}
