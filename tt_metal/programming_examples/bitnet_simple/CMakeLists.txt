# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_bitnet_simple)

add_executable(metal_example_bitnet_simple)
target_sources(
    metal_example_bitnet_simple PRIVATE
        # Host code
        bitnet_unpack_demo.cpp
        # Note: Kernel files are compiled at runtime by TT-Metal, not via CMake
)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_bitnet_simple PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_bitnet_simple TT::CommonPCH)
endif()
