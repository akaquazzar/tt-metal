#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    DEVICE_PRINT("+++kernel_main");
    const uint32_t packed_weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t act_addr            = get_arg_val<uint32_t>(1);
    const uint32_t num_k_tiles         = get_arg_val<uint32_t>(2); // 16

    constexpr uint32_t cb_packed = tt::CBIndex::c_0;
    constexpr uint32_t cb_act    = tt::CBIndex::c_1;

    constexpr uint32_t packed_tile_bytes = 4096;
    constexpr uint32_t act_tile_bytes    = 1024;
    constexpr uint32_t batch_size        = 4;

    constexpr auto weights_args = TensorAccessorArgs<0>();
    const auto weights_gen = TensorAccessor(weights_args, packed_weights_addr, packed_tile_bytes);

    constexpr auto act_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const auto act_gen = TensorAccessor(act_args, act_addr, act_tile_bytes);

    cb_reserve_back(cb_packed, 1);
    uint32_t l1_weights_addr = get_write_ptr(cb_packed);
    noc_async_read(weights_gen.get_noc_addr(0), l1_weights_addr, packed_tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_packed, 1);

    for (uint32_t k = 0; k < num_k_tiles; k += batch_size) {
        cb_reserve_back(cb_act, batch_size);
        uint32_t l1_act_addr = get_write_ptr(cb_act);

        for (uint32_t j = 0; j < batch_size; ++j) {
            noc_async_read(act_gen.get_noc_addr(k + j), l1_act_addr + j * act_tile_bytes, act_tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_act, batch_size);
    }
    DEVICE_PRINT("---kernel_main\n");
}
