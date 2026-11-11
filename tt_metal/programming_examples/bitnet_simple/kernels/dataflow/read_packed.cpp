// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    // Runtime arguments
    uint32_t packed_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    // Compile-time: Circular buffer indices
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;

    // Address generator for input buffer (interleaved)
    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_accessor = TensorAccessor(input_args, packed_buffer_addr);

    // Constants
    constexpr uint32_t TILE_SIZE = 32 * 32;  // 1024 elements
    constexpr uint32_t TRITS_PER_BYTE = 4;
    constexpr uint32_t PACKED_BYTES_PER_TILE = TILE_SIZE / TRITS_PER_BYTE;  // 256 bytes per tile
    constexpr uint32_t packed_tile_size = PACKED_BYTES_PER_TILE * sizeof(uint8_t);

    // Main loop: Read each tile from DRAM and push to circular buffer
    for (uint32_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        // Reserve space in circular buffer for one packed tile
        cb_reserve_back(cb_in0, 1);

        // Get write pointer for the circular buffer
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        // Read packed tile from DRAM to L1
        noc_async_read_page(tile_idx, input_accessor, l1_write_addr);

        // Wait for read to complete
        noc_async_read_barrier();

        // Push the tile to circular buffer
        cb_push_back(cb_in0, 1);
    }
}
