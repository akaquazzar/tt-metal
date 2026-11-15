#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t tile_bytes = 4096;

    const InterleavedAddrGenFast<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::UInt32
    };

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        noc_async_read_tile(i, src_gen, l1_write_addr);
        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
    }

    DEVICE_PRINT("=== Exit from READ ===\n");
}
