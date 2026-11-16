#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t total_out_tiles = get_arg_val<uint32_t>(1); // На хосте передаётся NUM_OUT_TILES (16)

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 1024;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst_gen = TensorAccessor(dst_args, dst_addr, tile_bytes);

    for (uint32_t i = 0; i < total_out_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        uint64_t dst_noc_addr = dst_gen.get_noc_addr(i);
        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
    DEVICE_PRINT("=== Exit from WRITE ===\n");
}
