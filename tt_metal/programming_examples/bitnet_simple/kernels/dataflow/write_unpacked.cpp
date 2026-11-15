#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_in_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t tile_bytes = 1024;
    const uint32_t total_out_tiles = n_in_tiles * 16;

    const InterleavedAddrGenFast<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = DataFormat::Int8
    };

    for (uint32_t i = 0; i < total_out_tiles; ++i) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        noc_async_write_tile(i, dst_gen, l1_read_addr);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
