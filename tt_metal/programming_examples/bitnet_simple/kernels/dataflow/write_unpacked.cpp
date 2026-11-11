// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    uint32_t output_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // Compile-time: Circular buffer index for output
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Address generator for output buffer (interleaved)
    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output_accessor = TensorAccessor(output_args, output_buffer_addr);

    // Main loop: Wait for each tile in circular buffer and write to DRAM
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Wait for tile to be available in circular buffer
        cb_wait_front(cb_out, 1);

        // Get read pointer for the circular buffer
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write tile from L1 to DRAM
        noc_async_write_page(tile_idx, output_accessor, l1_read_addr);
        noc_async_write_barrier();

        // Pop the tile from circular buffer
        cb_pop_front(cb_out, 1);
    }
}
