#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t tile_bytes = 4096;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src_gen = TensorAccessor(src_args, src_addr, tile_bytes);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        uint64_t src_noc_addr = src_gen.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, 4096);
        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
    }

    DEVICE_PRINT("=== Exit from READ ===\n");
}
