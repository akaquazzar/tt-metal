// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel::sfpu {

/**
 * Распаковка 16 тритов из одного UInt32 элемента в 16 банков регистров.
 *
 * Branchless формула декодирования:
 *   plus_one  = (packed >> (k*2))   & 0x01
 *   minus_one = (packed >> (k*2+1)) & 0x01
 *   result    = plus_one - minus_one
 *
 * Результат:
 *   00 -> 0-0 = 0
 *   01 -> 1-0 = +1
 *   10 -> 0-1 = -1
 *   11 -> 1-1 = 0
 */
sfpi_inline void unpack_row_batch(uint32_t row_idx, uint32_t start_k, uint32_t count) {
    sfpi::vInt packed = sfpi::dst_reg[row_idx];

for (uint32_t i = 0; i < count; ++i) {
    uint32_t k = start_k + i;
    sfpi::vInt plus_one  = (packed >> (k * 2)) & 0x01;
    sfpi::vInt minus_one = (packed >> (k * 2 + 1)) & 0x01;
    sfpi::dst_reg[(i + 1) * 32 + row_idx] = plus_one - minus_one;
}
}


}  // namespace ckernel::sfpu
